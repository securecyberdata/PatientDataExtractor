# Patient Data Extractor

A Streamlit application for extracting and standardizing patient data from CSV files containing JSON-formatted patient information.

## Features

- **CSV Processing**: Upload and process CSV files with patient data
- **JSON Parsing**: Robust parsing of JSON strings containing patient information
- **Data Standardization**: Extract and standardize patient information into a consistent format
- **AI-Powered Medication Analysis**: Analyze patient medications using DeepSeek and Anthropic Claude APIs to determine:
  - Diabetes status based on medications
  - Need for orthopedic support/braces based on medications
  - Consensus analysis combining both AI models
- **Export Functionality**: Download processed data as CSV files

## Dual AI Analysis Feature

The application now includes AI-powered medication analysis using both DeepSeek and Anthropic Claude APIs:

### Setup
1. **DeepSeek API**: Obtain a DeepSeek API key from [DeepSeek](https://platform.deepseek.com/)
2. **Anthropic API**: Obtain an Anthropic API key from [Anthropic](https://console.anthropic.com/)
3. Enter your API keys in the sidebar (you can use one or both)
4. Enable "AI Medication Analysis" checkbox
5. Test the API connections using the "Test DeepSeek API" and "Test Anthropic API" buttons

### AI Analysis Results
The AI analysis adds three new columns to your output:

- **`is_diabetic_AI`**: Diabetes assessment based on medications (from available API(s))
- **`need_braces_AI`**: Orthopedic/braces assessment based on medications (from available API(s))
- **`ai_reasoning`**: Explanation for AI conclusions

**Note**: The headers are dynamic based on which APIs you configure:
- **DeepSeek only**: Results from DeepSeek analysis
- **Anthropic only**: Results from Anthropic analysis  
- **Both APIs**: Consensus analysis combining both results

### Supported Medications Analysis
The AI analyzes medications for:
- **Diabetes indicators**: metformin, insulin, sulfonylureas, etc.
- **Orthopedic/arthritis indicators**: NSAIDs, corticosteroids, etc.
- **Mobility-related medications**: medications that might indicate mobility issues

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python -m streamlit run app.py
   ```

## Google Drive Setup (Optional)

To enable Google Drive integration:

1. **Create a Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Google Drive API

2. **Create Credentials**:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - Choose "Desktop application"
   - Download the credentials file as `credentials.json`

3. **Upload Credentials**:
   - In the app sidebar, enable "Google Drive Upload"
   - Upload your `credentials.json` file
   - Test the connection

4. **First-time Authentication**:
   - Click "Test Google Drive Connection"
   - A browser window will open for Google authentication
   - Grant permissions to access your Google Drive
   - The app will create a "PatientDataExtractor" folder in your Drive

## Usage

1. Open the application in your web browser
2. (Optional) Enter your DeepSeek and/or Anthropic API keys and enable AI analysis
3. (Optional) Enable Google Drive integration and upload credentials
4. Upload a CSV file with the required format
5. Review the processing results and AI analysis (if enabled)
6. Download the processed data as a CSV file or upload to Google Drive

## Required CSV Format

Your CSV file must contain these columns:
- `status`: Boolean or string indicating record validity
- `message`: String that should be "Success" for valid records  
- `data`: JSON string containing patient information

## Extracted Fields

The application extracts the following patient information:
- Patient ID, Name, Date of Birth, Medicare ID
- Complete Address (concatenated from address components)
- Phone
- Medications (semicolon-separated list)
- All PCP-related fields (NPI, names, contact information, etc.)
- AI Analysis Results (if enabled):
  - is_diabetic_AI: Diabetes assessment based on medications (from available API(s))
  - need_braces_AI: Orthopedic/braces assessment based on medications (from available API(s))
  - ai_reasoning: Explanation for AI conclusions

## Dependencies

- streamlit==1.28.1
- chardet==5.2.0
- requests==2.31.0
- google-auth==2.23.4
- google-auth-oauthlib==1.1.0
- google-auth-httplib2==0.1.1
- google-api-python-client==2.108.0
- anthropic==0.7.8

**Note**: This application uses only standard Python libraries (csv, json) for data processing, eliminating the need for pandas and ensuring maximum compatibility with Streamlit Cloud. 
