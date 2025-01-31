import os.path
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import base64
from googleapiclient.errors import HttpError
import pandas as pd

# NOT NEEDED - REDUNDANT

INPUT_FILE = 'George Contacts.csv'
OUTPUT_FILE = 'George Emails 2'


# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


# authenticates the user and returns the Gmail API service
def authenticate_gmail_api():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('gmail_token.json'):
        creds = Credentials.from_authorized_user_file('gmail_token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'gmail_credentials.json', SCOPES)
            creds = flow.run_local_server(port=8888)
        # Save the credentials for the next run
        with open('gmail_token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service


# lists the messages in the user's mailbox given a query
def list_messages(service, user_id, query=''):
    try:
        response = service.users().messages().list(userId=user_id, q=query).execute()
        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])

        while 'nextPageToken' in response:
            page_token = response['nextPageToken']
            response = service.users().messages().list(userId=user_id, q=query,
                                                       pageToken=page_token).execute()
            messages.extend(response['messages'])

        return messages
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None


# returns the email pure text content given a message id
def get_email_string(service, user_id, msg_id):
    try:
        message = service.users().messages().get(userId=user_id, id=msg_id, format='full').execute()

        # Extract the subject
        headers = message['payload']['headers']
        subject = next(header['value'] for header in headers if header['name'] == 'Subject')

        # Extract the plain text part
        parts = message['payload'].get('parts', [])
        text = ''

        def extract_text(parts):
            for part in parts:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                elif 'parts' in part:
                    return extract_text(part['parts'])
            return ''

        text = extract_text(parts)

        # Combine subject and text into one string
        email_content = f"Subject: {subject}\n\n{text}"
        return email_content
    except HttpError as error:
        print(f'An error occurred: {error}')
        return None


# lists the messages sent from the email addresses in the email list
def list_messages_from(service, user_id, email_list):
    messages = []
    for email in email_list:
        query_from = f'from:{email}'
        messages_from = list_messages(service, user_id, query_from)
        messages_from = [] if messages_from is None else messages_from
        messages.append(messages_from)
    return messages


# lists the messages sent to the email addresses in the email list
def list_messages_to(service, user_id, email_list):
    messages = []
    for email in email_list:
        query_to = f'to:{email}'
        messages_to = list_messages(service, user_id, query_to)
        messages_to = [] if messages_to is None else messages_to
        messages.append(messages_to)
    return messages


# saves the messages sent to and from the email addresses in the email to a csv file
def df_messages(service, email_list, user_id='me', output_file='brandon_labels.csv'):
    messages_from = list_messages_from(service, user_id, email_list)
    messages_to = list_messages_to(service, user_id, email_list)

    data = []
    for email, from_msgs, to_msgs in zip(email_list, messages_from, messages_to):
        for i, m in enumerate(from_msgs):
            data.append({
                'email': email,
                'direction': 'from',
                'message_index': i + 1,
                'message_content': get_email_string(service, user_id, m['id'])
            })
        for i, m in enumerate(to_msgs):
            data.append({
                'email': email,
                'direction': 'to',
                'message_index': i + 1,
                'message_content': get_email_string(service, user_id, m['id'])
            })

    df = pd.DataFrame(data)
    if output_file:
        df.to_csv(f'{output_file}.csv', index=False)
        df.to_excel(f'{output_file}.xlsx', index=False)
    return df


def main(input_file, output_file='brandon_labels.csv'):
    # Authenticate and build the service
    service = authenticate_gmail_api()

    # Define the email addresses you are interested in
    input = pd.read_csv(input_file)
    email_addresses = [email.strip() for email in input['Email'].tolist() if email.strip()]
    email_addresses = list(set(email_addresses))

    if output_file:
        df_messages(service, email_addresses, user_id='me', output_file=output_file)
    else:
        # Fetch emails sent to the email addresses
        messages_to = list_messages_to(service, 'me', email_addresses)
        print(f'Emails sent to {email_addresses}:')
        for msg in messages_to:
            for m in msg:
                print(get_email_string(service, 'me', m['id']))
        # Fetch emails replied from the email addresses
        messages_from = list_messages_from(service, 'me', email_addresses)
        print(f'\nEmails replied from {email_addresses}:')
        for msg in messages_from:
            for m in msg:
                print(get_email_string(service, 'me', m['id']))


if __name__ == '__main__':
    main(INPUT_FILE, OUTPUT_FILE)
