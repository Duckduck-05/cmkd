import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import pytz

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ID of the folder in Google Drive from which you want to retrieve information
FOLDER_ID = "115T_hgRjKTWIueGDSxvsQpp_FaGro-uN"

# Path where you want to download files from Google Drive
DOWNLOAD_PATH = '/workdir/carrot/ravvdess/vid_features'

# Set to True to download subfolders (recursive), False to only download files in the current folder (sequential/flat)
RECURSIVE_MODE = False


def download_files_recursively(service, folder_id, local_path):
    """Downloads content from a specific folder on Google Drive, respecting RECURSIVE_MODE and pagination."""
    
    page_token = None
    while True:
        # Added orderBy='name' to ensure files are processed sequentially by name
        # Added pageToken to ensure we get ALL files, not just the first 100
        try:
            results = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false", 
                fields="nextPageToken, files(id, name, mimeType)",
                orderBy="name",
                pageToken=page_token
            ).execute()
        except HttpError as error:
            print(f"An error occurred accessing folder {folder_id}: {error}")
            break

        items = results.get("files", [])
        page_token = results.get("nextPageToken")

        if not items and page_token is None:
            # Only print if truly empty on the first/only page
            # print("No files found in this folder.")
            pass

        for item in items:
            item_name = item["name"]
            item_id = item["id"]
            item_type = item["mimeType"]

            # If it's a folder
            if item_type == "application/vnd.google-apps.folder":
                # Check the recursive flag before proceeding
                if RECURSIVE_MODE:
                    new_local_path = os.path.join(local_path, item_name)
                    os.makedirs(new_local_path, exist_ok=True)
                    print(f"Downloading content of folder: {new_local_path}")
                    download_files_recursively(service, item_id, new_local_path)
                else:
                    print(f"Skipping folder: {item_name}")
            else:
                # Download the file into the specified local folder
                local_file_path = os.path.join(local_path, item_name)
                
                # Basic check to skip if file exists (optional, keeping it simple as per request)
                # if os.path.exists(local_file_path):
                #     print(f"File already exists: {item_name}")
                #     continue

                print(f"Downloading file: {local_file_path}")
                request = service.files().get_media(fileId=item_id)
                with open(local_file_path, "wb") as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        print(f"Downloading {item_name} at {int(status.progress() * 100)}%.")
        
        # If there are no more pages, break the loop
        if not page_token:
            break


def convert_expiry_to_paris_time(expiry_utc):
    utc_timezone = pytz.utc
    paris_timezone = pytz.timezone('Europe/Paris')
    if hasattr(expiry_utc, 'replace'): 
        expiry_utc = utc_timezone.localize(expiry_utc) if expiry_utc.tzinfo is None else expiry_utc
        expiry_paris = expiry_utc.astimezone(paris_timezone)
        return expiry_paris
    return expiry_utc


def main():
    """Recursively retrieves all content from a specific folder on Google Drive."""
    creds = None
    token_path = "token.json"

    # Check if the token.json file exists and contains valid credentials
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # If no valid credentials are found, allow the user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                print("Token refreshed:")
                print(f"Access Token: {creds.token}")
                print(f"Refresh Token: {creds.refresh_token}")
                print("Expiry:", convert_expiry_to_paris_time(creds.expiry))
            except Exception as e:
                print(f"Error refreshing token: {e}")
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
                print("New authorization:")
                print(f"Access Token: {creds.token}")
                print(f"Refresh Token: {creds.refresh_token}")
                print("Expiry:", convert_expiry_to_paris_time(creds.expiry))
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
            print("New authorization:")
            print(f"Access Token: {creds.token}")
            print(f"Refresh Token: {creds.refresh_token}")
            print("Expiry:", convert_expiry_to_paris_time(creds.expiry))

        # Save the credentials for the next execution
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    else:
        print("Existing token:")
        print(f"Access Token: {creds.token}")
        print(f"Refresh Token: {creds.refresh_token}")
        print("Expiry:", convert_expiry_to_paris_time(creds.expiry))

    try:
        # Initialize the Google Drive service
        service = build("drive", "v3", credentials=creds)

        # Recursively download all content from the specified folder
        download_files_recursively(service, FOLDER_ID, DOWNLOAD_PATH)
    except HttpError as error:
        # Handle errors from the Google Drive API
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()