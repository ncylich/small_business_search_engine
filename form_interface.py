from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2 import service_account
from time import sleep
import pipeline
import os
import io


COLS_DICT = {'Timestamp': 0, 'Email Address': 1, "Please describe in detail the sector you'd like to match the companies to (leave empty if you don't want to)": 2, 'List all key phrases and their weights (do not write redundant key phrases, except for acronyms/abbreviations).\nEach key phrase should be its own line in the following format:\nkey phrase=weight\nOR if you choose to omit weighting, just:\nkey phrase\n': 3, 'Upload UDU Company List - must be ".csv" & contain company, description, contacts, & email columns': 4}


def authenticate_service_v4():
    credentials = service_account.Credentials.from_service_account_file('service_credentials.json')
    service = build('sheets', 'v4', credentials=credentials)
    return service


def authenticate_service_v3_old():
    # Define the scopes
    scopes = ['https://www.googleapis.com/auth/drive.metadata.readonly',
              'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file']

    # Authenticate and create the service
    flow = InstalledAppFlow.from_client_secrets_file('service_credentials.json', scopes)
    creds = flow.run_local_server(port=0)
    service = build('drive', 'v3', credentials=creds)
    return service


def authenticate_service_v3():
    credentials = service_account.Credentials.from_service_account_file('service_credentials.json')
    service = build('drive', 'v3', credentials=credentials)
    return service


def list_files(service):  # service must be v3
    # Call the Drive v3 API to list the files
    results = service.files().list(pageSize=10, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])

    items = [(item['name'], item['id']) for item in items]
    return items if items else []


def get_files_in_folder(service, folder_id):  # service must be v3
    query = f"'{folder_id}' in parents"   # Query to search for files within the specified folder
    # Call the Drive v3 API to list the files
    results = service.files().list(q=query, pageSize=10, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    return items if items else []


def download_file(service, file_id, file_name, dir_path=''):  # service must be v3
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()

    # Create a downloader object to manage download.
    downloader = MediaIoBaseDownload(fh, request)
    done = False

    while not done:
        status, done = downloader.next_chunk()
        print("Download Progress: {0}".format(status.progress() * 100))

    # Write the downloaded content to a file
    fh.seek(0)
    file_name = os.path.join(dir_path, file_name) if dir_path else file_name  # merging the directory path
    with open(file_name, 'wb') as f:
        f.write(fh.read())
        print('File downloaded successfully: {}'.format(file_name))


def upload_file(service, folder_id, file_name, dir_path=""):  # service must be v3
    file_metadata = {'name': os.path.basename(file_name), 'parents': [folder_id]}

    file_name = os.path.join(dir_path, file_name) if dir_path else file_name
    media = MediaFileUpload(file_name, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print('File ID: %s' % file.get('id'))


def delete_file(service, file_id):  # service must be v3
    try:
        service.files().delete(fileId=file_id).execute()
        print("File deleted successfully.")
    except Exception as e:
        print("An error occurred: %s" % e)


def read_specific_line(line_number, service, sheet_id='1bGWOozPOWes27q2k_kLN7XWPYvNhP7M8nEnq9QHdlpM',
                       sheet_name='Form Responses 1'):  # service must be v4
    sheet_range = f"{sheet_name}!A{line_number}:E{line_number}"
    result = service.spreadsheets().values().get(spreadsheetId=sheet_id, range=sheet_range).execute()
    values = result.get('values', [])

    return values[0] if values else []


def get_file_name(file_id, service):  # service must be v3
    file = service.files().get(fileId=file_id).execute()
    return file['name']


def main():
    # Generating path configs
    folder = "interface_temp"

    num_path = os.path.join(folder, "num")
    with open(num_path, 'r') as f:
        num = int(f.read())  # num is the last line read from the Google sheets

    if num < 1:  # handling invalid num values
        with open(num_path, 'w') as f:
            f.write(str(1))
        num = 1
    num += 1

    i_file = f"{num} input.csv"
    o_file = f"{num} results.csv"

    # IDs
    sheet_id = '1bGWOozPOWes27q2k_kLN7XWPYvNhP7M8nEnq9QHdlpM'
    folder_id = '1aaJRpG-lwrMo6I2-YKRZyIpJWHJoreo97jTEzfXV0vRrv9C6tNdinZg1GHQAGLnrGunVB6CW'
    results_id = '1McyRfSkytWjeMS-gtjYiskVPdhkphqud'

    # Authenticating services
    service_v4 = authenticate_service_v4()
    service_v3 = authenticate_service_v3()

    values = []
    while not values:
        values = read_specific_line(num, service_v4, sheet_id=sheet_id)
        sleep(5)

    sector, key_phrases, csv = values[2], values[3], values[4]
    csv_id = csv.split('=')[-1].strip()
    key_phrases = [phrase.split('=') for phrase in key_phrases.split('\n')]
    key_phrases = {pair[0].strip('"'): float(pair[1].strip(',')) if len(pair) == 2 else 1 for pair in key_phrases}

    # Checking if the file is a csv, if not -> cancels -> re-run with next num if this is main script
    file_name = get_file_name(csv_id, service_v3)
    if not file_name.endswith('.csv'):
        with open(num_path, 'w') as f:
            f.write(str(num))
        return

    # Processing the file, if error, outputs, and proceeds to next num
    try:
        # Downloading the file
        download_file(service_v3, csv_id, i_file, dir_path=folder)
        print(f'File "{file_name}" downloaded successfully.')

        # Processing the file
        pipeline.pipeline(udu_csv=i_file, results_path=o_file, keys=key_phrases, sector=sector, dir_path=folder)

        # Uploading the results
        upload_file(service_v3, results_id, o_file, dir_path=folder)
        print(f'File "{file_name}" processed & uploaded successfully.')

        # Cleaning up
        os.remove(os.path.join(folder, i_file))
        os.remove(os.path.join(folder, o_file))
    except Exception as e:
        print(f'An error occurred while processing the file "{file_name}": {e}')

    # Updating the num
    with open(num_path, 'w') as f:
        f.write(str(num))


if __name__ == "__main__":
    while True:
        main()
        sleep(30)
