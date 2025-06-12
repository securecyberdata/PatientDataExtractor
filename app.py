import streamlit as st
import pandas as pd
import json
import csv
import logging
import chardet
import io
import os
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
        
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error: {e} for string: {json_string[:100]}...")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error parsing JSON: {e}")
        return None

def extract_patient_data(json_data: Dict[Any, Any]) -> Dict[str, Any]:
    """Extract patient information from JSON data."""
    if not json_data:
        return {}
    
    extracted = {}
    
    # Basic patient information
    extracted['Patient_ID'] = json_data.get('patient_id', '')
    extracted['Name'] = json_data.get('name', '')
    extracted['Date_of_Birth'] = json_data.get('date_of_birth', '')
    extracted['Gender'] = json_data.get('gender', '')
    extracted['Medicare_ID'] = json_data.get('medicare_id', '')
    
    # Address concatenation
    address_parts = []
    address = json_data.get('address', '')
    city = json_data.get('city', '')
    state = json_data.get('state', '')
    zip_code = json_data.get('zip_code', '')
    country = json_data.get('country', 'USA')  # Default to USA
    
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
    
    extracted['Complete_Address'] = ', '.join(address_parts)
    
    # Contact information
    extracted['Phone'] = json_data.get('phone', '')
    extracted['Email'] = json_data.get('email', '')
    extracted['Status'] = json_data.get('status', '')
    extracted['Verified_Status'] = json_data.get('verified_status', '')
    
    # Medications (semicolon-separated)
    medications = json_data.get('medications', [])
    if isinstance(medications, list):
        extracted['Medications'] = '; '.join([str(med) for med in medications if med])
    else:
        extracted['Medications'] = str(medications) if medications else ''
    
    # Family History (semicolon-separated)
    family_history = json_data.get('family_history', [])
    if isinstance(family_history, list):
        family_history_formatted = []
        for history in family_history:
            if isinstance(history, dict):
                history_parts = []
                for key in ['relation', 'sex', 'status', 'age_of_deceased', 'maternal_paternal', 'health_issues']:
                    value = history.get(key, '')
                    if value:
                        history_parts.append(f"{key}: {value}")
                if history_parts:
                    family_history_formatted.append(' | '.join(history_parts))
            else:
                family_history_formatted.append(str(history))
        extracted['Family_History'] = '; '.join(family_history_formatted)
    else:
        extracted['Family_History'] = str(family_history) if family_history else ''
    
    # PCP-related fields
    pcp_data = json_data.get('pcp', {}) or {}
    extracted['PCP_NPI'] = pcp_data.get('npi', '')
    extracted['PCP_First_Name'] = pcp_data.get('first_name', '')
    extracted['PCP_Last_Name'] = pcp_data.get('last_name', '')
    extracted['PCP_Address'] = pcp_data.get('address', '')
    extracted['PCP_City'] = pcp_data.get('city', '')
    extracted['PCP_State'] = pcp_data.get('state', '')
    extracted['PCP_Postal_Code'] = pcp_data.get('postal_code', '')
    extracted['PCP_Phone'] = pcp_data.get('phone', '')
    extracted['PCP_Fax_Number'] = pcp_data.get('fax_number', '')
    extracted['PCP_Email'] = pcp_data.get('email', '')
    extracted['PCP_Comment'] = pcp_data.get('comment', '')
    extracted['PCP_Confirm_Response'] = pcp_data.get('confirm_response', '')
    extracted['PCP_Tracker'] = pcp_data.get('tracker', '')
    
    return extracted

def process_csv_file(uploaded_file) -> tuple[pd.DataFrame, Dict[str, int], List[str]]:
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
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8', errors='replace')
        
        stats['total_rows'] = len(df)
        logger.info(f"Total rows in CSV: {stats['total_rows']}")
        
        # Check if required columns exist
        required_columns = ['status', 'message', 'data']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        for index, row in df.iterrows():
            row_num = index + 1
            
            # Check status and message validity
            status = row.get('status')
            message = row.get('message')
            data = row.get('data')
            
            # Convert status to boolean if it's a string
            if isinstance(status, str):
                status = status.lower() in ['true', '1', 'yes']
            
            if not status or message != "Success":
                stats['skipped_rows'] += 1
                stats['invalid_status_message'] += 1
                reason = f"Row {row_num}: Invalid status ({status}) or message ({message})"
                skip_reasons.append(reason)
                logger.debug(reason)
                continue
            
            # Check if data is empty or null
            if pd.isna(data) or not str(data).strip():
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
            # Create empty DataFrame with expected columns
            columns = [
                'Patient_ID', 'Name', 'Date_of_Birth', 'Gender', 'Medicare_ID',
                'Complete_Address', 'Phone', 'Email', 'Status', 'Verified_Status',
                'Medications', 'Family_History', 'PCP_NPI', 'PCP_First_Name',
                'PCP_Last_Name', 'PCP_Address', 'PCP_City', 'PCP_State',
                'PCP_Postal_Code', 'PCP_Phone', 'PCP_Fax_Number', 'PCP_Email',
                'PCP_Comment', 'PCP_Confirm_Response', 'PCP_Tracker'
            ]
            output_df = pd.DataFrame(columns=columns)
        
        return output_df, stats, skip_reasons
        
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise e

def main():
    st.title("Medical Data Extraction Tool")
    st.markdown("Upload a CSV file containing patient data to extract and standardize patient information.")
    
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
            with st.spinner("Processing file..."):
                # Process the CSV file
                output_df, stats, skip_reasons = process_csv_file(uploaded_file)
            
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
                st.subheader("Extracted Data Preview")
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
                
                st.success(f"✅ Successfully processed {stats['processed_rows']} patient records!")
                
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
            - Patient ID, Name, Date of Birth, Gender, Medicare ID
            - Complete Address (concatenated from address components)
            - Phone, Email, Status, Verified Status
            - Medications (semicolon-separated list)
            - Family History (semicolon-separated with detailed information)
            - All PCP-related fields (NPI, names, contact information, etc.)
            """)

if __name__ == "__main__":
    main()
