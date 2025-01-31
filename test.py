from googleapiclient.discovery import build
from google.oauth2 import service_account

def read_specific_line(sheet_id, sheet_name, line_number, credentials_file):
    # Authenticate using service account credentials
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    service = build('sheets', 'v4', credentials=credentials)

    # Call the Sheets API
    sheet_range = f"{sheet_name}!A{line_number}:E{line_number}"
    result = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
    values = result.get('values', [])


    return values[0] if values else []


# Example usage:
sheet_id = '1bGWOozPOWes27q2k_kLN7XWPYvNhP7M8nEnq9QHdlpM'
sheet_name = 'Form Responses 1'
line_number = 3
credentials_file = 'service_credentials.json'

line_data = read_specific_line(sheet_id, sheet_name, line_number, credentials_file)
print("Line data:", line_data)
if line_data:
    print('True')
