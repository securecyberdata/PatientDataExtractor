import streamlit as st
import json
import csv
import logging
import chardet
import io
import os
import requests
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Google Drive imports
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    import pickle
    GOOGLE_DRIVE_AVAILABLE = True
    print("✅ Google Drive packages imported successfully")
except ImportError as e:
    GOOGLE_DRIVE_AVAILABLE = False
    print(f"❌ Google Drive packages not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting for Anthropic API (50 requests per minute)
class AnthropicRateLimiter:
    def __init__(self, max_requests_per_minute=50):
        self.max_requests = max_requests_per_minute
        self.request_times = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we're approaching the rate limit"""
        with self.lock:
            current_time = time.time()
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            if len(self.request_times) >= self.max_requests:
                # Calculate how long to wait
                oldest_request = min(self.request_times)
                wait_time = 60 - (current_time - oldest_request) + 0.1  # Add small buffer
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    current_time = time.time()
            
            # Add current request time
            self.request_times.append(current_time)

# Global rate limiter instance
anthropic_rate_limiter = AnthropicRateLimiter()

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
    if json_string is None or not json_string.strip():
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

def authenticate_google_drive():
    """
    Authenticate with Google Drive API.
    Returns credentials if successful, None otherwise.
    """
    if not GOOGLE_DRIVE_AVAILABLE:
        return None
    
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = None
    
    # Check if we have stored credentials
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Check if credentials.json exists
            if not os.path.exists('credentials.json'):
                st.error("❌ Google Drive credentials not found. Please upload your credentials.json file.")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def upload_to_google_drive(file_content: str, filename: str, folder_name: str = "PatientDataExtractor") -> str:
    """
    Upload a file to Google Drive.
    
    Args:
        file_content: The content of the file to upload
        filename: Name of the file
        folder_name: Name of the folder in Google Drive (will be created if doesn't exist)
    
    Returns:
        URL of the uploaded file or error message
    """
    if not GOOGLE_DRIVE_AVAILABLE:
        return "Google Drive API not available"
    
    try:
        # Authenticate
        creds = authenticate_google_drive()
        if not creds:
            return "Authentication failed"
        
        # Build the service
        service = build('drive', 'v3', credentials=creds)
        
        # Create or find the folder
        folder_id = None
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query).execute()
        files = results.get('files', [])
        
        if files:
            folder_id = files[0]['id']
        else:
            # Create the folder
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = folder.get('id')
        
        # Prepare file metadata
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        # Create file content
        file_content_io = io.BytesIO(file_content.encode('utf-8'))
        
        # Try different MIME types for CSV files - Google Drive is picky about CSV uploads
        mime_types_to_try = [
            'text/plain',  # Most compatible
            'application/csv',
            'text/csv',
            'application/vnd.ms-excel'
        ]
        
        upload_success = False
        last_error = None
        
        for mime_type in mime_types_to_try:
            try:
                logger.info(f"Attempting upload with MIME type: {mime_type}")
                media = MediaIoBaseUpload(file_content_io, mimetype=mime_type, resumable=True)
                
                # Upload the file
                file = service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id, webViewLink'
                ).execute()
                
                logger.info(f"Upload successful with MIME type: {mime_type}")
                upload_success = True
                break
                
            except Exception as e:
                last_error = e
                # Reset the BytesIO object for the next attempt
                file_content_io.seek(0)
                continue
        
        if not upload_success:
            raise Exception(f"Failed to upload with any MIME type. Last error: {last_error}")
        
        return file.get('webViewLink', 'Upload successful but link not available')
        
    except Exception as e:
        logger.error(f"Google Drive upload failed: {e}")
        return f"Upload failed: {str(e)}"

def analyze_medications_with_deepseek(medications: str, api_key: str) -> Dict[str, str]:
    """
    Analyze medications using DeepSeek API to determine if patient is diabetic or needs braces.
    
    Args:
        medications: Semicolon-separated list of medications
        api_key: DeepSeek API key
    
    Returns:
        Dictionary with 'is_diabetic' and 'need_braces' analysis results
    """
    if not medications or medications == 'N/A' or medications.strip() == '':
        return {
            'is_diabetic': 'No medications data available', 
            'need_braces': 'No medications data available',
            'reasoning': 'No medication information provided for analysis'
        }
    
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
    
    Important guidelines:
    - Answer "No" if no relevant medications are found
    - Answer "Yes" only if clear evidence of diabetes/orthopedic medications
    - Answer "Uncertain" if medications are unclear or insufficient
    - Consider diabetes medications: metformin, insulin, sulfonylureas, DPP-4 inhibitors, GLP-1 agonists
    - Consider orthopedic medications: NSAIDs, corticosteroids, pain medications, mobility aids
    - Provide clear reasoning for your conclusions
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
        
        response = requests.post(url, headers=headers, json=data, timeout=15)
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

def analyze_medications_with_anthropic(medications: str, api_key: str) -> Dict[str, str]:
    """
    Analyze medications using Anthropic Claude API to determine if patient is diabetic or needs braces.
    
    Args:
        medications: Semicolon-separated list of medications
        api_key: Anthropic API key
    
    Returns:
        Dictionary with 'is_diabetic' and 'need_braces' analysis results
    """
    if not medications or medications == 'N/A' or medications.strip() == '':
        return {
            'is_diabetic': 'No medications data available', 
            'need_braces': 'No medications data available',
            'reasoning': 'No medication information provided for analysis'
        }
    
    # Prepare the prompt for Claude
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
    
    Important guidelines:
    - Answer "No" if no relevant medications are found
    - Answer "Yes" only if clear evidence of diabetes/orthopedic medications
    - Answer "Uncertain" if medications are unclear or insufficient
    - Consider diabetes medications: metformin, insulin, sulfonylureas, DPP-4 inhibitors, GLP-1 agonists
    - Consider orthopedic medications: NSAIDs, corticosteroids, pain medications, mobility aids
    - Provide clear reasoning for your conclusions
    """
    
    try:
        # Import Anthropic client
        logger.info("Attempting to import Anthropic SDK...")
        from anthropic import Anthropic
        logger.info("Anthropic SDK imported successfully")
        
        # Apply rate limiting for Anthropic API
        logger.info("Applying rate limiting for Anthropic API...")
        anthropic_rate_limiter.wait_if_needed()
        
        # Initialize Anthropic client with proper configuration
        logger.info("Initializing Anthropic client...")
        try:
            # Try simple initialization first
            client = Anthropic(api_key=api_key)
            logger.info("Anthropic client initialized successfully with simple method")
        except Exception as init_error:
            logger.warning(f"Simple initialization failed: {init_error}")
            # Try alternative initialization method
            client = Anthropic()
            client.api_key = api_key
            logger.info("Anthropic client initialized successfully with alternative method")
        
        # Call Claude API with updated model name
        logger.info("Making API call to Anthropic...")
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        logger.info("Anthropic API call completed successfully")
        
        # Extract content safely
        logger.info("Extracting response content...")
        if hasattr(response, 'content') and response.content:
            if isinstance(response.content, list) and len(response.content) > 0:
                content = response.content[0].text
                logger.info("Content extracted from list response")
            elif hasattr(response.content, 'text'):
                content = response.content.text
                logger.info("Content extracted from text response")
            else:
                content = str(response.content)
                logger.info("Content extracted as string")
        else:
            raise Exception("No content received from Anthropic API")
        
        logger.info(f"Response content length: {len(content)}")
        
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
            
    except ImportError as e:
        logger.error(f"Anthropic SDK not available: {e}")
        return {
            'is_diabetic': 'Anthropic SDK not installed',
            'need_braces': 'Anthropic SDK not installed',
            'reasoning': 'Please install anthropic package: pip install anthropic'
        }
    except Exception as e:
        logger.error(f"Anthropic analysis failed: {e}")
        return {
            'is_diabetic': f'Analysis Error: {str(e)}',
            'need_braces': f'Analysis Error: {str(e)}',
            'reasoning': 'Unexpected error during Anthropic analysis'
        }

def process_csv_file(uploaded_file, enable_ai_analysis=False, deepseek_api_key=None, anthropic_api_key=None) -> tuple[List[Dict], Dict[str, int], List[str]]:
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
    ai_analysis_count = 0
    
    try:
        # Try to read with detected encoding
        try:
            text_data = file_bytes.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            logger.warning(f"Failed to read with {encoding}, trying utf-8 with error handling")
            text_data = file_bytes.decode('utf-8', errors='replace')
        
        # Increase CSV field size limit to handle large JSON strings
        import sys
        maxInt = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt/10)
        
        # Parse CSV using standard csv module
        csv_reader = csv.DictReader(io.StringIO(text_data))
        rows = list(csv_reader)
        
        stats['total_rows'] = len(rows)
        logger.info(f"Total rows in CSV: {stats['total_rows']}")
        
        # Check if required columns exist
        required_columns = ['status', 'message', 'data']
        if rows:
            missing_columns = [col for col in required_columns if col not in rows[0].keys()]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        for idx, row in enumerate(rows):
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
                
                # Always create a patient record, even if extraction failed
                if not patient_data:
                    patient_data = {
                        'Patient_ID': f'Row_{row_num}',
                        'Name': 'N/A',
                        'Date_of_Birth': 'N/A',
                        'Medicare_ID': 'N/A',
                        'Complete_Address': 'N/A',
                        'Phone': 'N/A',
                        'Medications': 'N/A'
                    }
                
                # Add AI analysis if enabled - ALWAYS add AI results
                if enable_ai_analysis and (deepseek_api_key or anthropic_api_key):
                    try:
                        # Add a small delay to avoid rate limiting
                        import time
                        time.sleep(0.1)
                        
                        # Determine which APIs are available and set appropriate headers
                        available_apis = []
                        if deepseek_api_key:
                            available_apis.append('DeepSeek')
                        if anthropic_api_key:
                            available_apis.append('Anthropic')
                        
                        # Perform analysis based on available APIs
                        if len(available_apis) == 1:
                            # Single API - use simple headers
                            if deepseek_api_key:
                                analysis = analyze_medications_with_deepseek(
                                    patient_data.get('Medications', ''), 
                                    deepseek_api_key
                                )
                                patient_data['is_diabetic_AI'] = analysis['is_diabetic']
                                patient_data['need_braces_AI'] = analysis['need_braces']
                                patient_data['ai_reasoning'] = analysis['reasoning']
                            else:  # Anthropic only
                                logger.info(f"Processing Anthropic analysis for row {row_num} (Rate limited)")
                                analysis = analyze_medications_with_anthropic(
                                    patient_data.get('Medications', ''), 
                                    anthropic_api_key
                                )
                                patient_data['is_diabetic_AI'] = analysis['is_diabetic']
                                patient_data['need_braces_AI'] = analysis['need_braces']
                                patient_data['ai_reasoning'] = analysis['reasoning']
                        
                        elif len(available_apis) == 2:
                            # Both APIs available - create consensus analysis
                            deepseek_analysis = analyze_medications_with_deepseek(
                                patient_data.get('Medications', ''), 
                                deepseek_api_key
                            )
                            
                            logger.info(f"Processing Anthropic analysis for row {row_num} (Rate limited)")
                            anthropic_analysis = analyze_medications_with_anthropic(
                                patient_data.get('Medications', ''), 
                                anthropic_api_key
                            )
                            
                            # Consensus logic
                            deepseek_diabetic = deepseek_analysis['is_diabetic']
                            anthropic_diabetic = anthropic_analysis['is_diabetic']
                            deepseek_braces = deepseek_analysis['need_braces']
                            anthropic_braces = anthropic_analysis['need_braces']
                            
                            diabetic_consensus = 'Uncertain'
                            if deepseek_diabetic == anthropic_diabetic:
                                diabetic_consensus = deepseek_diabetic
                            elif 'Yes' in [deepseek_diabetic, anthropic_diabetic]:
                                diabetic_consensus = 'Yes (Partial Consensus)'
                            elif 'No' in [deepseek_diabetic, anthropic_diabetic]:
                                diabetic_consensus = 'No (Partial Consensus)'
                            
                            braces_consensus = 'Uncertain'
                            if deepseek_braces == anthropic_braces:
                                braces_consensus = deepseek_braces
                            elif 'Yes' in [deepseek_braces, anthropic_braces]:
                                braces_consensus = 'Yes (Partial Consensus)'
                            elif 'No' in [deepseek_braces, anthropic_braces]:
                                braces_consensus = 'No (Partial Consensus)'
                            
                            patient_data['is_diabetic_AI'] = diabetic_consensus
                            patient_data['need_braces_AI'] = braces_consensus
                            patient_data['ai_reasoning'] = f"DeepSeek: {deepseek_diabetic}, Anthropic: {anthropic_diabetic} | Braces - DeepSeek: {deepseek_braces}, Anthropic: {anthropic_braces}"
                        
                        # Log successful analysis
                        ai_analysis_count += 1
                        logger.info(f"AI analysis completed for row {row_num} using {', '.join(available_apis)} (Total: {ai_analysis_count})")
                        
                    except Exception as ai_error:
                        logger.warning(f"AI analysis failed for row {row_num}: {ai_error}")
                        # Use a simple fallback analysis based on medications
                        medications = patient_data.get('Medications', '').lower()
                        if 'metformin' in medications or 'insulin' in medications or 'glucophage' in medications:
                            patient_data['is_diabetic_AI'] = 'Yes (Fallback)'
                        elif 'nsaid' in medications or 'ibuprofen' in medications or 'naproxen' in medications:
                            patient_data['need_braces_AI'] = 'Yes (Fallback)'
                        else:
                            patient_data['is_diabetic_AI'] = 'No (Fallback)'
                            patient_data['need_braces_AI'] = 'No (Fallback)'
                        patient_data['ai_reasoning'] = f'Fallback analysis due to API error: {str(ai_error)}'
                else:
                    patient_data['is_diabetic_AI'] = 'AI Analysis Disabled'
                    patient_data['need_braces_AI'] = 'AI Analysis Disabled'
                    patient_data['ai_reasoning'] = 'AI analysis was not enabled'
                
                processed_data.append(patient_data)
                stats['processed_rows'] += 1
            except Exception as e:
                stats['skipped_rows'] += 1
                reason = f"Row {row_num}: Data extraction error - {str(e)}"
                skip_reasons.append(reason)
                logger.error(reason)
        
        # Log final AI analysis summary
        if enable_ai_analysis:
            logger.info(f"AI Analysis Summary: {ai_analysis_count} successful analyses out of {stats['processed_rows']} patients")
        
        # Final verification - ensure we have results for all patients
        logger.info(f"Final Summary: {len(processed_data)} patients processed, {stats['skipped_rows']} skipped")
        
        # Return processed data as list of dictionaries
        return processed_data, stats, skip_reasons
        
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise e

def main():
    st.title("Medical Data Extraction Tool")
    st.markdown("Upload a CSV file containing patient data to extract and standardize patient information.")
    
    # AI API Configuration
    st.sidebar.header("🔑 AI API Configuration")
    
    # DeepSeek API Key input
    st.sidebar.subheader("🤖 DeepSeek API")
    deepseek_api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        type="password",
        help="Enter your DeepSeek API key to enable medication analysis"
    )
    
    # Test DeepSeek API connection button
    if deepseek_api_key and st.sidebar.button("Test DeepSeek API"):
        with st.spinner("Testing DeepSeek API connection..."):
            test_result = analyze_medications_with_deepseek("metformin 500mg", deepseek_api_key)
            if "API Error" in test_result['is_diabetic'] or "Analysis Error" in test_result['is_diabetic']:
                st.sidebar.error("❌ DeepSeek API connection failed. Please check your API key.")
            else:
                st.sidebar.success("✅ DeepSeek API connection successful!")
                with st.sidebar.expander("DeepSeek Test Result"):
                    st.write(f"**Diabetes:** {test_result['is_diabetic']}")
                    st.write(f"**Braces:** {test_result['need_braces']}")
                    st.write(f"**Reasoning:** {test_result['reasoning']}")
    
    # Anthropic API Key input
    st.sidebar.subheader("🧠 Anthropic Claude API")
    anthropic_api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key to enable Claude medication analysis"
    )
    
    # Test Anthropic API connection button
    if anthropic_api_key and st.sidebar.button("Test Anthropic API"):
        with st.spinner("Testing Anthropic API connection..."):
            try:
                # First test if we can import the SDK
                st.sidebar.info("🔍 Testing Anthropic SDK import...")
                from anthropic import Anthropic
                st.sidebar.info("✅ Anthropic SDK imported successfully")
                
                # Test client initialization
                st.sidebar.info("🔍 Testing client initialization...")
                client = Anthropic(api_key=anthropic_api_key)
                st.sidebar.info("✅ Anthropic client initialized successfully")
                
                # Test a simple API call
                st.sidebar.info("🔍 Testing API call...")
                test_result = analyze_medications_with_anthropic("metformin 500mg", anthropic_api_key)
                
                if "API Error" in test_result['is_diabetic'] or "Analysis Error" in test_result['is_diabetic']:
                    st.sidebar.error(f"❌ Anthropic API connection failed: {test_result['reasoning']}")
                else:
                    st.sidebar.success("✅ Anthropic API connection successful!")
                    with st.sidebar.expander("Anthropic Test Result"):
                        st.write(f"**Diabetes:** {test_result['is_diabetic']}")
                        st.write(f"**Braces:** {test_result['need_braces']}")
                        st.write(f"**Reasoning:** {test_result['reasoning']}")
                        
            except ImportError as e:
                st.sidebar.error(f"❌ Anthropic SDK import failed: {str(e)}")
                st.sidebar.info("💡 Try running: pip install --upgrade anthropic")
            except Exception as e:
                st.sidebar.error(f"❌ Anthropic client initialization failed: {str(e)}")
                st.sidebar.info("💡 Check your API key and internet connection")
    
    # Enable/disable AI analysis
    enable_ai_analysis = st.sidebar.checkbox(
        "Enable AI Medication Analysis",
        value=False,
        help="Analyze medications to determine diabetes and orthopedic conditions using DeepSeek and/or Anthropic"
    )
    
    if enable_ai_analysis and not deepseek_api_key and not anthropic_api_key:
        st.sidebar.error("⚠️ Please enter at least one API key (DeepSeek or Anthropic) to enable AI analysis.")
        enable_ai_analysis = False
    
    # Google Drive Configuration
    st.sidebar.header("☁️ Google Drive Integration")
    
    # Debug: Show Google Drive availability status
    st.sidebar.info(f"Google Drive Status: {'✅ Available' if GOOGLE_DRIVE_AVAILABLE else '❌ Not Available'}")
    
    if GOOGLE_DRIVE_AVAILABLE:
        enable_google_drive = st.sidebar.checkbox(
            "Enable Google Drive Upload",
            value=False,
            help="Automatically save processed files to Google Drive"
        )
        
        if enable_google_drive:
            # Google Drive credentials upload
            credentials_file = st.sidebar.file_uploader(
                "Upload Google Drive Credentials (credentials.json)",
                type="json",
                help="Upload your Google Drive API credentials file"
            )
            
            if credentials_file:
                # Save credentials file
                with open('credentials.json', 'wb') as f:
                    f.write(credentials_file.getvalue())
                st.sidebar.success("✅ Credentials uploaded successfully!")
                
                # Test Google Drive connection
                if st.sidebar.button("Test Google Drive Connection"):
                    with st.sidebar.spinner("Testing Google Drive connection..."):
                        try:
                            creds = authenticate_google_drive()
                            if creds:
                                st.sidebar.success("✅ Google Drive connection successful!")
                            else:
                                st.sidebar.error("❌ Google Drive authentication failed.")
                        except Exception as e:
                            st.sidebar.error(f"❌ Google Drive test failed: {str(e)}")
            else:
                st.sidebar.warning("⚠️ Please upload your Google Drive credentials to enable upload.")
                enable_google_drive = False
    else:
        st.sidebar.warning("⚠️ Google Drive integration not available. Install required packages.")
        enable_google_drive = False
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with columns: status, message, and data (containing JSON strings)"
    )
    
    # Use Streamlit's built-in caching to prevent re-processing
    @st.cache_data
    def process_file_with_cache(file_content, enable_ai_analysis, deepseek_api_key, anthropic_api_key, filename):
        # Create a temporary file-like object
        import io
        temp_file = io.BytesIO(file_content)
        temp_file.name = filename  # Set the name for the function
        return process_csv_file(temp_file, enable_ai_analysis, deepseek_api_key, anthropic_api_key)
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Get filename without extension for output naming
        input_filename = os.path.splitext(uploaded_file.name)[0]
        output_filename = f"{input_filename}_output.csv"
        
        try:
            # Read file content once
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            # Show different spinner messages based on whether AI analysis is enabled
            if enable_ai_analysis:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                with st.spinner("Processing file with AI medication analysis..."):
                    progress_text.text("Starting AI analysis...")
                    # Process the CSV file with caching
                    output_data, stats, skip_reasons = process_file_with_cache(file_content, enable_ai_analysis, deepseek_api_key, anthropic_api_key, uploaded_file.name)
                    progress_text.text("AI analysis completed!")
                    progress_bar.progress(1.0)
            else:
                with st.spinner("Processing file..."):
                    # Process the CSV file with caching
                    output_data, stats, skip_reasons = process_file_with_cache(file_content, enable_ai_analysis, deepseek_api_key, anthropic_api_key, uploaded_file.name)
            
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
            if output_data:
                # Show processing completion message
                if enable_ai_analysis:
                    st.success(f"✅ Successfully processed {stats['processed_rows']} patient records with AI analysis!")
                else:
                    st.success(f"✅ Successfully processed {stats['processed_rows']} patient records!")
                
                st.subheader("Extracted Data Preview")
                
                # Show AI analysis summary if enabled
                if enable_ai_analysis and output_data and 'is_diabetic_AI' in output_data[0]:
                    st.subheader("🤖 AI Analysis Summary")
                    
                    # Count AI analysis results
                    diabetic_counts = {}
                    braces_counts = {}
                    
                    for row in output_data:
                        diabetic = row.get('is_diabetic_AI', 'Unknown')
                        braces = row.get('need_braces_AI', 'Unknown')
                        diabetic_counts[diabetic] = diabetic_counts.get(diabetic, 0) + 1
                        braces_counts[braces] = braces_counts.get(braces, 0) + 1
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Diabetes Analysis:**")
                        for result, count in diabetic_counts.items():
                            st.write(f"- {result}: {count} patients")
                    
                    with col2:
                        st.write("**Orthopedic/Braces Analysis:**")
                        for result, count in braces_counts.items():
                            st.write(f"- {result}: {count} patients")
                    
                    # Show detailed breakdown
                    st.write("**📈 Detailed Breakdown:**")
                    st.write(f"Total patients analyzed: {len(output_data)}")
                    st.write(f"Patients with diabetes indicators: {diabetic_counts.get('Yes', 0) + diabetic_counts.get('Yes (Partial Consensus)', 0)}")
                    st.write(f"Patients with orthopedic needs: {braces_counts.get('Yes', 0) + braces_counts.get('Yes (Partial Consensus)', 0)}")
                    st.write(f"Patients with no clear indicators: {diabetic_counts.get('No', 0) + braces_counts.get('No', 0)}")
                    
                    # Show sample reasoning
                    if 'ai_reasoning' in output_data[0]:
                        sample_reasoning = next((row['ai_reasoning'] for row in output_data if row.get('ai_reasoning') != 'AI Analysis Disabled'), "No AI analysis performed")
                        with st.expander("Sample AI Reasoning", expanded=False):
                            st.write(sample_reasoning)
                
                st.subheader("📊 Complete Data Preview")
                # Display first 10 rows as a table
                if output_data:
                    # Get all unique keys from all dictionaries
                    all_keys = set()
                    for row in output_data:
                        all_keys.update(row.keys())
                    
                    # Create a list of dictionaries with all keys
                    display_data = []
                    for row in output_data[:10]:  # First 10 rows
                        display_row = {}
                        for key in sorted(all_keys):
                            display_row[key] = row.get(key, '')
                        display_data.append(display_row)
                    
                    st.dataframe(display_data, use_container_width=True)
                
                # Convert list of dictionaries to CSV for download
                csv_buffer = io.StringIO()
                if output_data:
                    # Get all unique keys
                    all_keys = set()
                    for row in output_data:
                        all_keys.update(row.keys())
                    
                    # Write CSV
                    writer = csv.DictWriter(csv_buffer, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    for row in output_data:
                        # Ensure all keys are present
                        csv_row = {}
                        for key in sorted(all_keys):
                            csv_row[key] = row.get(key, '')
                        writer.writerow(csv_row)
                
                csv_data = csv_buffer.getvalue()
                
                # Download and Google Drive upload buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label=f"📥 Download {output_filename}",
                        data=csv_data,
                        file_name=output_filename,
                        mime="text/csv",
                        key="download_csv"
                    )
                
                with col2:
                    if enable_google_drive and GOOGLE_DRIVE_AVAILABLE:
                        if st.button("☁️ Upload to Google Drive", key="upload_gdrive"):
                            with st.spinner("Uploading to Google Drive..."):
                                try:
                                    # Add timestamp to filename
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    gdrive_filename = f"{input_filename}_output_{timestamp}.csv"
                                    
                                    # Upload to Google Drive
                                    upload_result = upload_to_google_drive(csv_data, gdrive_filename)
                                    
                                    if "Upload failed" in upload_result or "Authentication failed" in upload_result:
                                        st.error(f"❌ {upload_result}")
                                    else:
                                        st.success("✅ File uploaded to Google Drive successfully!")
                                        st.info(f"📁 File saved as: {gdrive_filename}")
                                        st.info(f"🔗 Access link: {upload_result}")
                                except Exception as e:
                                    st.error(f"❌ Upload failed: {str(e)}")
                    else:
                        st.button("☁️ Upload to Google Drive", disabled=True, help="Enable Google Drive integration in sidebar")
                
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
              - is_diabetic_AI: Diabetes assessment based on medications (from available API(s))
              - need_braces_AI: Orthopedic/braces assessment based on medications (from available API(s))
              - ai_reasoning: Explanation for AI conclusions
            """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check the console logs for more details.")
        logger.error(f"Application startup error: {e}")
