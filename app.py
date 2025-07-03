import streamlit as st
import pandas as pd
import json
import csv
import logging
import chardet
import io
import os
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_encoding(file_bytes: bytes) -> str:
    """Detect the encoding of the uploaded file."""
    try:
        result = chardet.detect(file_bytes)
        encoding = result['encoding']
        confidence = result['confidence']
        logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
        return encoding if encoding else 'utf-8'
    except Exception as e:
        logger.warning(f"Failed to detect encoding: {e}. Using utf-8 as fallback.")
        return 'utf-8'

def safe_json_parse(json_string: str) -> Optional[Dict[Any, Any]]:
    """Safely parse JSON string with error handling."""
    if pd.isna(json_string) or not json_string.strip():
        return None
    
    try:
        # Handle cases where the JSON might be enclosed in extra quotes
        json_string = json_string.strip()
        if json_string.startswith('"') and json_string.endswith('"'):
            json_string = json_string[1:-1]
        
        # Unescape escaped quotes
        json_string = json_string.replace('\\"', '"')
        
        # First try standard JSON parsing
        return json.loads(json_string)
        
    except json.JSONDecodeError:
        # Try to handle Python dict-like strings with single quotes
        try:
            # Replace single quotes with double quotes for JSON compliance
            # But be careful not to replace quotes inside strings
            import ast
            # Use ast.literal_eval for Python dict-like strings
            parsed_data = ast.literal_eval(json_string)
            return parsed_data
        except (ValueError, SyntaxError):
            # Try a more aggressive approach for malformed JSON-like strings
            try:
                # Replace single quotes with double quotes, handling nested structures
                json_string_fixed = json_string.replace("'", '"')
                # Handle None values which are not valid JSON
                json_string_fixed = json_string_fixed.replace('None', 'null')
                # Handle True/False values
                json_string_fixed = json_string_fixed.replace('True', 'true')
                json_string_fixed = json_string_fixed.replace('False', 'false')
                return json.loads(json_string_fixed)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error after all attempts: {e} for string: {json_string[:100]}...")
                return None
    except Exception as e:
        logger.warning(f"Unexpected error parsing JSON: {e}")
        return None

def extract_patient_data(json_data: Dict[Any, Any]) -> Dict[str, Any]:
    """Extract patient information from JSON data."""
    if not json_data:
        return {}
    
    # Handle nested structure - check if data is wrapped in 'result'
    if 'result' in json_data and isinstance(json_data['result'], dict):
        patient_data = json_data['result']
    else:
        patient_data = json_data
    
    extracted = {}
    
    # Basic patient information - check main level and profile_data/rawjson
    extracted['Patient_ID'] = patient_data.get('id', patient_data.get('patient_id', 'N/A'))
    
    # Name from multiple sources
    name = patient_data.get('full_name', patient_data.get('name', ''))
    if not name and 'profile_data' in patient_data and 'rawjson' in patient_data['profile_data']:
        rawjson = patient_data['profile_data']['rawjson']
        first_name = rawjson.get('patient_first_name', '')
        last_name = rawjson.get('patient_last_name', '')
        name = f"{first_name} {last_name}".strip()
    extracted['Name'] = name if name else 'N/A'
    
    # Date of Birth
    dob = patient_data.get('birthday', patient_data.get('date_of_birth', ''))
    if not dob and 'profile_data' in patient_data and 'rawjson' in patient_data['profile_data']:
        dob = patient_data['profile_data']['rawjson'].get('patient_dob', '')
        # Clean up the datetime format
        if 'T' in str(dob):
            dob = str(dob).split('T')[0]
    extracted['Date_of_Birth'] = dob if dob else 'N/A'
    
    # Medicare ID
    medicare_id = patient_data.get('medicare_id', '')
    if not medicare_id and 'profile_data' in patient_data and 'rawjson' in patient_data['profile_data']:
        medicare_id = patient_data['profile_data']['rawjson'].get('patient_medicare_id', '')
    extracted['Medicare_ID'] = medicare_id if medicare_id else 'N/A'
    
    # Address concatenation
    address_parts = []
    
    # Try main level first
    address = patient_data.get('address', '')
    city = patient_data.get('city', '')
    state = patient_data.get('state', '')
    zip_code = patient_data.get('zip_code', '')
    country = patient_data.get('country_name', 'USA')
    
    # If not found, check profile_data/rawjson
    if not address and 'profile_data' in patient_data and 'rawjson' in patient_data['profile_data']:
        rawjson = patient_data['profile_data']['rawjson']
        address = rawjson.get('patient_address', '')
        city = rawjson.get('patient_city', '')
        state = rawjson.get('patient_state', '')
        zip_code = rawjson.get('patient_zip_code', '')
        country = rawjson.get('patient_country', 'USA')
    
    if address:
        address_parts.append(str(address))
    if city:
        address_parts.append(str(city))
    if state:
        address_parts.append(str(state))
    if zip_code:
        address_parts.append(str(zip_code))
    if country:
        address_parts.append(str(country))
    
    extracted['Complete_Address'] = ', '.join(address_parts) if address_parts else 'N/A'
    
    # Contact information
    phone = patient_data.get('phone', '')
    email = patient_data.get('email', '')
    
    if not phone and 'profile_data' in patient_data and 'rawjson' in patient_data['profile_data']:
        phone = patient_data['profile_data']['rawjson'].get('patient_phone', '')
    if not email and 'profile_data' in patient_data and 'rawjson' in patient_data['profile_data']:
        email = patient_data['profile_data']['rawjson'].get('patient_email', '')
    
    extracted['Phone'] = phone if phone else 'N/A'
    
    # Medications - check pmi_data structure based on sample
    medications = []
    if 'pmi_data' in patient_data and isinstance(patient_data['pmi_data'], dict):
        pmi_data = patient_data['pmi_data']
        patient_medications = pmi_data.get('patient_medications', [])
        if isinstance(patient_medications, list):
            for med in patient_medications:
                if isinstance(med, dict) and 'Name' in med:
                    medications.append(med['Name'])
                elif isinstance(med, str):
                    medications.append(med)
    
    extracted['Medications'] = '; '.join(medications) if medications else 'N/A'
    

    
    # PCP-related fields from pmi_data with pcp_ prefix
    pcp_data = {}
    if 'pmi_data' in patient_data and isinstance(patient_data['pmi_data'], dict):
        pmi_data = patient_data['pmi_data']
        
        # Extract PCP fields with pcp_ prefix
        extracted['PCP_NPI'] = pmi_data.get('pcp_npi', 'N/A')
        extracted['PCP_First_Name'] = pmi_data.get('pcp_first_name', 'N/A')
        extracted['PCP_Last_Name'] = pmi_data.get('pcp_last_name', 'N/A')
        extracted['PCP_Address'] = pmi_data.get('pcp_address', 'N/A')
        extracted['PCP_City'] = pmi_data.get('pcp_city', 'N/A')
        extracted['PCP_State'] = pmi_data.get('pcp_state', 'N/A')
        extracted['PCP_Postal_Code'] = pmi_data.get('pcp_postal_code', 'N/A')
        extracted['PCP_Phone'] = pmi_data.get('pcp_phone', 'N/A')
        extracted['PCP_Fax_Number'] = pmi_data.get('pcp_fax_number', 'N/A')
        extracted['PCP_Email'] = pmi_data.get('pcp_email', 'N/A')
        extracted['PCP_Comment'] = pmi_data.get('pcp_pcp_comment', 'N/A')
        extracted['PCP_Confirm_Response'] = pmi_data.get('pcp_confirm_response', 'N/A')
        
        # Handle tracker which might be a dict
        tracker = pmi_data.get('pcp_tracker', {})
        if isinstance(tracker, dict):
            tracker_parts = []
            for key, value in tracker.items():
                tracker_parts.append(f"{key}: {value}")
            extracted['PCP_Tracker'] = '; '.join(tracker_parts) if tracker_parts else 'N/A'
        else:
            extracted['PCP_Tracker'] = str(tracker) if tracker else 'N/A'
    else:
        # Set all PCP fields to N/A if pmi_data not found
        extracted['PCP_NPI'] = 'N/A'
        extracted['PCP_First_Name'] = 'N/A'
        extracted['PCP_Last_Name'] = 'N/A'
        extracted['PCP_Address'] = 'N/A'
        extracted['PCP_City'] = 'N/A'
        extracted['PCP_State'] = 'N/A'
        extracted['PCP_Postal_Code'] = 'N/A'
        extracted['PCP_Phone'] = 'N/A'
        extracted['PCP_Fax_Number'] = 'N/A'
        extracted['PCP_Email'] = 'N/A'
        extracted['PCP_Comment'] = 'N/A'
        extracted['PCP_Confirm_Response'] = 'N/A'
        extracted['PCP_Tracker'] = 'N/A'
    
    return extracted

def analyze_medications_with_deepseek(medications: str, api_key: str) -> Dict[str, str]:
    """
    Analyze medications using DeepSeek API to determine if patient is diabetic or needs braces.
    
    Args:
        medications: Semicolon-separated list of medications
        api_key: DeepSeek API key
    
    Returns:
        Dictionary with 'is_diabetic' and 'need_braces' analysis results
    """
    if not medications or medications == 'N/A':
        return {'is_diabetic': 'No medications data available', 'need_braces': 'No medications data available'}
    
    # Prepare the prompt for DeepSeek
    prompt = f"""
    Analyze the following list of medications and determine:
    1. Is the patient diabetic? (Yes/No/Uncertain)
    2. Does the patient need braces or orthopedic support? (Yes/No/Uncertain)
    
    Medications: {medications}
    
    Please provide your analysis in the following JSON format:
    {{
        "is_diabetic": "Yes/No/Uncertain",
        "need_braces": "Yes/No/Uncertain",
        "reasoning": "Brief explanation for your conclusions"
    }}
    
    Consider:
    - Diabetes medications (metformin, insulin, sulfonylureas, etc.)
    - Orthopedic/arthritis medications (NSAIDs, corticosteroids, etc.)
    - Medications that might indicate mobility issues
    """
    
    try:
        # DeepSeek API endpoint
        url = "https://api.deepseek.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Try to parse the JSON response
        try:
            # Extract JSON from the response (in case there's extra text)
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = json.loads(content)
            
            return {
                'is_diabetic': analysis.get('is_diabetic', 'Analysis failed'),
                'need_braces': analysis.get('need_braces', 'Analysis failed'),
                'reasoning': analysis.get('reasoning', 'No reasoning provided')
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return {
                'is_diabetic': 'Analysis completed but parsing failed',
                'need_braces': 'Analysis completed but parsing failed',
                'reasoning': content[:200] + '...' if len(content) > 200 else content
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"DeepSeek API request failed: {e}")
        return {
            'is_diabetic': f'API Error: {str(e)}',
            'need_braces': f'API Error: {str(e)}',
            'reasoning': 'Failed to connect to DeepSeek API'
        }
    except Exception as e:
        logger.error(f"DeepSeek analysis failed: {e}")
        return {
            'is_diabetic': f'Analysis Error: {str(e)}',
            'need_braces': f'Analysis Error: {str(e)}',
            'reasoning': 'Unexpected error during analysis'
        }

def process_csv_file(uploaded_file, enable_ai_analysis=False, deepseek_api_key=None) -> tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Process the uploaded CSV file and extract patient data."""
    
    # Read file bytes for encoding detection
    file_bytes = uploaded_file.read()
    encoding = detect_encoding(file_bytes)
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    # Statistics tracking
    stats = {
        'total_rows': 0,
        'processed_rows': 0,
        'skipped_rows': 0,
        'invalid_status_message': 0,
        'json_parse_errors': 0,
        'empty_data': 0
    }
    
    skip_reasons = []
    processed_data = []
    
    try:
        # Try to read with detected encoding
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            logger.warning(f"Failed to read with {encoding}, trying utf-8 with error handling")
            # Create a string with error handling and then use StringIO
            text_data = file_bytes.decode('utf-8', errors='replace')
            df = pd.read_csv(io.StringIO(text_data))
        
        stats['total_rows'] = len(df)
        logger.info(f"Total rows in CSV: {stats['total_rows']}")
        
        # Check if required columns exist
        required_columns = ['status', 'message', 'data']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        for idx, (index, row) in enumerate(df.iterrows()):
            row_num = idx + 1
            
            # Check status and message validity
            status = row['status'] if 'status' in row else None
            message = row['message'] if 'message' in row else None
            data = row['data'] if 'data' in row else None
            
            # Convert status to boolean if it's a string
            if isinstance(status, str):
                status = status.lower() in ['true', '1', 'yes']
            
            try:
                status_valid = bool(status) if status is not None else False
            except:
                status_valid = False
            if not status_valid or message != "Success":
                stats['skipped_rows'] += 1
                stats['invalid_status_message'] += 1
                reason = f"Row {row_num}: Invalid status ({status}) or message ({message})"
                skip_reasons.append(reason)
                logger.debug(reason)
                continue
            
            # Check if data is empty or null
            data_is_empty = False
            try:
                data_is_empty = pd.isna(data) or not str(data).strip()
            except:
                data_is_empty = data is None or str(data).strip() == ''
            
            if data_is_empty:
                stats['skipped_rows'] += 1
                stats['empty_data'] += 1
                reason = f"Row {row_num}: Empty or null data field"
                skip_reasons.append(reason)
                logger.debug(reason)
                continue
            
            # Parse JSON data
            json_data = safe_json_parse(str(data))
            if json_data is None:
                stats['skipped_rows'] += 1
                stats['json_parse_errors'] += 1
                reason = f"Row {row_num}: JSON parsing error"
                skip_reasons.append(reason)
                logger.debug(reason)
                continue
            
            # Extract patient data
            try:
                patient_data = extract_patient_data(json_data)
                if patient_data:
                    # Add AI analysis if enabled
                    if enable_ai_analysis and deepseek_api_key:
                        try:
                            ai_analysis = analyze_medications_with_deepseek(
                                patient_data.get('Medications', ''), 
                                deepseek_api_key
                            )
                            patient_data['is_diabetic_AI'] = ai_analysis['is_diabetic']
                            patient_data['need_braces_AI'] = ai_analysis['need_braces']
                            patient_data['ai_reasoning'] = ai_analysis['reasoning']
                        except Exception as ai_error:
                            logger.warning(f"AI analysis failed for row {row_num}: {ai_error}")
                            patient_data['is_diabetic_AI'] = 'AI Analysis Failed'
                            patient_data['need_braces_AI'] = 'AI Analysis Failed'
                            patient_data['ai_reasoning'] = f'Error: {str(ai_error)}'
                    else:
                        patient_data['is_diabetic_AI'] = 'AI Analysis Disabled'
                        patient_data['need_braces_AI'] = 'AI Analysis Disabled'
                        patient_data['ai_reasoning'] = 'AI analysis was not enabled'
                    
                    processed_data.append(patient_data)
                    stats['processed_rows'] += 1
                else:
                    stats['skipped_rows'] += 1
                    reason = f"Row {row_num}: No extractable patient data"
                    skip_reasons.append(reason)
                    logger.debug(reason)
            except Exception as e:
                stats['skipped_rows'] += 1
                reason = f"Row {row_num}: Data extraction error - {str(e)}"
                skip_reasons.append(reason)
                logger.error(reason)
        
        # Create output DataFrame
        if processed_data:
            output_df = pd.DataFrame(processed_data)
        else:
            # Create empty DataFrame with expected columns (removed Gender, Status, Verified_Status, Email, Family_History)
            expected_columns = [
                'Patient_ID', 'Name', 'Date_of_Birth', 'Medicare_ID',
                'Complete_Address', 'Phone', 'Medications', 
                'PCP_NPI', 'PCP_First_Name', 'PCP_Last_Name', 'PCP_Address', 'PCP_City', 
                'PCP_State', 'PCP_Postal_Code', 'PCP_Phone', 'PCP_Fax_Number', 'PCP_Email',
                'PCP_Comment', 'PCP_Confirm_Response', 'PCP_Tracker',
                'is_diabetic_AI', 'need_braces_AI', 'ai_reasoning'
            ]
            output_df = pd.DataFrame(data=None)
        
        return output_df, stats, skip_reasons
        
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise e

def main():
    st.title("Medical Data Extraction Tool")
    st.markdown("Upload a CSV file containing patient data to extract and standardize patient information.")
    
    # DeepSeek API Key input
    st.sidebar.header("🔑 DeepSeek API Configuration")
    deepseek_api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        type="password",
        help="Enter your DeepSeek API key to enable medication analysis"
    )
    
    # Test API connection button
    if deepseek_api_key and st.sidebar.button("Test API Connection"):
        with st.spinner("Testing DeepSeek API connection..."):
            test_result = analyze_medications_with_deepseek("metformin 500mg", deepseek_api_key)
            if "API Error" in test_result['is_diabetic'] or "Analysis Error" in test_result['is_diabetic']:
                st.sidebar.error("❌ API connection failed. Please check your API key.")
            else:
                st.sidebar.success("✅ API connection successful!")
                with st.sidebar.expander("Test Result"):
                    st.write(f"**Diabetes:** {test_result['is_diabetic']}")
                    st.write(f"**Braces:** {test_result['need_braces']}")
                    st.write(f"**Reasoning:** {test_result['reasoning']}")
    
    # Enable/disable AI analysis
    enable_ai_analysis = st.sidebar.checkbox(
        "Enable AI Medication Analysis",
        value=False,
        help="Analyze medications to determine diabetes and orthopedic conditions"
    )
    
    if enable_ai_analysis and not deepseek_api_key:
        st.sidebar.error("⚠️ Please enter your DeepSeek API key to enable AI analysis.")
        enable_ai_analysis = False
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with columns: status, message, and data (containing JSON strings)"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Get filename without extension for output naming
        input_filename = os.path.splitext(uploaded_file.name)[0]
        output_filename = f"{input_filename}_output.csv"
        
        try:
            # Show different spinner messages based on whether AI analysis is enabled
            if enable_ai_analysis:
                with st.spinner("Processing file with AI medication analysis..."):
                    # Process the CSV file
                    output_df, stats, skip_reasons = process_csv_file(uploaded_file, enable_ai_analysis, deepseek_api_key)
            else:
                with st.spinner("Processing file..."):
                    # Process the CSV file
                    output_df, stats, skip_reasons = process_csv_file(uploaded_file, enable_ai_analysis, deepseek_api_key)
            
            # Display processing statistics
            st.subheader("Processing Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", stats['total_rows'])
            with col2:
                st.metric("Processed Rows", stats['processed_rows'])
            with col3:
                st.metric("Skipped Rows", stats['skipped_rows'])
            
            # Detailed statistics
            st.subheader("Detailed Statistics")
            st.write(f"**Invalid status/message:** {stats['invalid_status_message']}")
            st.write(f"**JSON parsing errors:** {stats['json_parse_errors']}")
            st.write(f"**Empty data fields:** {stats['empty_data']}")
            
            # Show warnings if processing rate is low
            if stats['total_rows'] > 0:
                processing_rate = stats['processed_rows'] / stats['total_rows']
                if processing_rate < 0.5 and stats['total_rows'] >= 50:
                    st.warning(f"⚠️ Low processing rate: {processing_rate:.1%} of rows were processed. "
                             f"This may indicate data quality issues or file truncation.")
            
            # Display skip reasons if any
            if skip_reasons:
                with st.expander(f"View Skip Reasons ({len(skip_reasons)} items)", expanded=False):
                    for reason in skip_reasons[:100]:  # Limit display to first 100 reasons
                        st.text(reason)
                    if len(skip_reasons) > 100:
                        st.text(f"... and {len(skip_reasons) - 100} more reasons")
            
            # Display preview of extracted data
            if not output_df.empty:
                # Show processing completion message
                if enable_ai_analysis:
                    st.success(f"✅ Successfully processed {stats['processed_rows']} patient records with AI analysis!")
                else:
                    st.success(f"✅ Successfully processed {stats['processed_rows']} patient records!")
                
                st.subheader("Extracted Data Preview")
                
                # Show AI analysis summary if enabled
                if enable_ai_analysis and 'is_diabetic_AI' in output_df.columns:
                    st.subheader("🤖 AI Analysis Summary")
                    
                    # Count AI analysis results
                    diabetic_counts = output_df['is_diabetic_AI'].value_counts()
                    braces_counts = output_df['need_braces_AI'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Diabetes Analysis Results:**")
                        for result, count in diabetic_counts.items():
                            st.write(f"- {result}: {count} patients")
                    
                    with col2:
                        st.write("**Orthopedic/Braces Analysis Results:**")
                        for result, count in braces_counts.items():
                            st.write(f"- {result}: {count} patients")
                    
                    # Show sample reasoning
                    if 'ai_reasoning' in output_df.columns:
                        sample_reasoning = output_df[output_df['ai_reasoning'] != 'AI Analysis Disabled']['ai_reasoning'].iloc[0] if len(output_df[output_df['ai_reasoning'] != 'AI Analysis Disabled']) > 0 else "No AI analysis performed"
                        with st.expander("Sample AI Reasoning", expanded=False):
                            st.write(sample_reasoning)
                
                st.subheader("📊 Complete Data Preview")
                st.dataframe(output_df.head(10), use_container_width=True)
                
                # Convert DataFrame to CSV for download
                csv_buffer = io.StringIO()
                output_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                # Download button
                st.download_button(
                    label=f"Download {output_filename}",
                    data=csv_data,
                    file_name=output_filename,
                    mime="text/csv",
                    key="download_csv"
                )
                
            else:
                st.error("❌ No valid patient data could be extracted from the file.")
                st.info("Please check the file format and ensure it contains valid patient data with the required structure.")
            
            # Console-style logging output
            with st.expander("Console Logs", expanded=False):
                st.code(f"""
Processing Summary:
==================
Total rows in CSV: {stats['total_rows']}
Processed rows: {stats['processed_rows']}
Skipped rows: {stats['skipped_rows']}

Breakdown of skipped rows:
- Invalid status/message: {stats['invalid_status_message']}
- JSON parsing errors: {stats['json_parse_errors']}
- Empty data fields: {stats['empty_data']}

Output file: {output_filename}
Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            logger.error(f"File processing error: {e}")
            
            # Show detailed error information
            with st.expander("Error Details", expanded=True):
                st.code(str(e))
    
    else:
        st.info("👆 Please upload a CSV file to begin processing.")
        
        # Show expected file format
        with st.expander("Expected File Format", expanded=False):
            st.markdown("""
            **Required CSV Columns:**
            - `status`: Boolean or string indicating record validity
            - `message`: String that should be "Success" for valid records
            - `data`: JSON string containing patient information
            
            **Valid rows criteria:**
            - status = True (or "true", "1", "yes")
            - message = "Success"
            - data contains valid JSON with patient information
            
            **Extracted Fields:**
            - Patient ID, Name, Date of Birth, Medicare ID
            - Complete Address (concatenated from address components)
            - Phone, Email
            - Medications (semicolon-separated list)
            - Family History (semicolon-separated with detailed information)
            - All PCP-related fields (NPI, names, contact information, etc.)
            - AI Analysis Results (if enabled):
              - is_diabetic_AI: Diabetes assessment based on medications
              - need_braces_AI: Orthopedic/braces assessment based on medications
              - ai_reasoning: Explanation for AI conclusions
            """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check the console logs for more details.")
        logger.error(f"Application startup error: {e}")
