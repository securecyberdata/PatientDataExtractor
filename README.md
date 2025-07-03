# Patient Data Extractor

A Streamlit application for extracting and standardizing patient data from CSV files containing JSON-formatted patient information.

## Features

- **CSV Processing**: Upload and process CSV files with patient data
- **JSON Parsing**: Robust parsing of JSON strings containing patient information
- **Data Standardization**: Extract and standardize patient information into a consistent format
- **AI-Powered Medication Analysis**: Analyze patient medications using DeepSeek API to determine:
  - Diabetes status based on medications
  - Need for orthopedic support/braces based on medications
- **Export Functionality**: Download processed data as CSV files

## New AI Analysis Feature

The application now includes AI-powered medication analysis using the DeepSeek API:

### Setup
1. Obtain a DeepSeek API key from [DeepSeek](https://platform.deepseek.com/)
2. Enter your API key in the sidebar
3. Enable "AI Medication Analysis" checkbox
4. Test the API connection using the "Test API Connection" button

### AI Analysis Results
The AI analysis adds three new columns to your output:
- **`is_diabetic_AI`**: Determines if the patient is diabetic based on their medications
- **`need_braces_AI`**: Determines if the patient needs orthopedic support/braces
- **`ai_reasoning`**: Provides explanation for the AI's conclusions

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

## Usage

1. Open the application in your web browser
2. (Optional) Enter your DeepSeek API key and enable AI analysis
3. Upload a CSV file with the required format
4. Review the processing results and AI analysis (if enabled)
5. Download the processed data as a CSV file

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
  - is_diabetic_AI: Diabetes assessment based on medications
  - need_braces_AI: Orthopedic/braces assessment based on medications
  - ai_reasoning: Explanation for AI conclusions

## Dependencies

- streamlit>=1.28.0
- pandas>=2.1.0
- chardet>=5.2.0
- requests>=2.31.0 
