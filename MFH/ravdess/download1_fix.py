import os
import time
import socket
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import pytz

# --- CẤU HÌNH ---
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_ID = "115T_hgRjKTWIueGDSxvsQpp_FaGro-uN"
DOWNLOAD_PATH = '/workdir/carrot/ravvdess/vid_features'
RECURSIVE_MODE = False

# Tăng timeout cho socket lên 10 phút để tránh BrokenPipeError sớm
socket.setdefaulttimeout(600)

def download_files_recursively(service, folder_id, local_path):
    """Downloads content from a specific folder on Google Drive, with Resume and Retry capability."""
    
    page_token = None
    while True:
        try:
            # Thêm retry cho việc lấy danh sách file
            results = None
            for attempt in range(5):
                try:
                    results = service.files().list(
                        q=f"'{folder_id}' in parents and trashed=false", 
                        fields="nextPageToken, files(id, name, mimeType)",
                        orderBy="name",
                        pageToken=page_token
                    ).execute()
                    break
                except (socket.error, HttpError, Exception) as e:
                    print(f"[LIST ERROR] Lỗi lấy danh sách (lần {attempt+1}): {e}. Đợi 5s...")
                    time.sleep(5)
            
            if not results:
                print("Không thể lấy danh sách file sau nhiều lần thử. Dừng.")
                break

        except Exception as error:
            print(f"An error occurred accessing folder {folder_id}: {error}")
            break

        items = results.get("files", [])
        page_token = results.get("nextPageToken")

        if not items and page_token is None:
            pass

        for item in items:
            item_name = item["name"]
            item_id = item["id"]
            item_type = item["mimeType"]

            # Xử lý thư mục (Folder)
            if item_type == "application/vnd.google-apps.folder":
                if RECURSIVE_MODE:
                    new_local_path = os.path.join(local_path, item_name)
                    os.makedirs(new_local_path, exist_ok=True)
                    print(f"Checking folder: {new_local_path}")
                    download_files_recursively(service, item_id, new_local_path)
                else:
                    print(f"Skipping folder: {item_name}")
            
            # Xử lý File
            else:
                local_file_path = os.path.join(local_path, item_name)
                
                # --- LOGIC 1: SKIP EXISTING (Tiếp tục tải) ---
                if os.path.exists(local_file_path):
                    # Bạn có thể thêm kiểm tra kích thước file tại đây nếu cần thiết
                    # Hiện tại chỉ kiểm tra tên file tồn tại để bỏ qua cho nhanh
                    print(f"[SKIP] File đã tồn tại: {item_name}")
                    continue

                # --- LOGIC 2: RETRY DOWNLOAD (Xử lý BrokenPipe) ---
                print(f"[START] Downloading: {item_name}")
                max_retries = 5
                success = False
                
                for attempt in range(max_retries):
                    try:
                        request = service.files().get_media(fileId=item_id)
                        with open(local_file_path, "wb") as fh:
                            downloader = MediaIoBaseDownload(fh, request)
                            done = False
                            while not done:
                                status, done = downloader.next_chunk()
                                # In progress trên cùng 1 dòng
                                print(f"Downloading {item_name}: {int(status.progress() * 100)}%", end='\r')
                        
                        print(f"\n[DONE] Hoàn thành: {item_name}")
                        success = True
                        break # Tải xong thì thoát vòng lặp retry
                    
                    except (socket.error, HttpError, Exception) as e:
                        print(f"\n[ERROR] Lỗi khi tải {item_name}: {e}")
                        print(f"Đang thử lại ({attempt + 1}/{max_retries})...")
                        time.sleep(3 * (attempt + 1)) # Đợi tăng dần: 3s, 6s, 9s...

                if not success:
                    print(f"\n[FAIL] Không thể tải file {item_name} sau {max_retries} lần thử. Bỏ qua.")
                    # Xóa file lỗi để lần sau chạy lại tải lại từ đầu
                    if os.path.exists(local_file_path):
                        os.remove(local_file_path)

        # Nếu không còn trang tiếp theo thì dừng
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
                print("Token refreshed.")
            except Exception as e:
                print(f"Error refreshing token: {e}")
                # Nếu refresh lỗi, thử xóa token cũ đi
                if os.path.exists(token_path):
                    os.remove(token_path)
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next execution
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    
    print(f"Access Token Valid. Expiry: {convert_expiry_to_paris_time(creds.expiry)}")

    try:
        # Initialize the Google Drive service
        service = build("drive", "v3", credentials=creds)
        
        # Đảm bảo thư mục gốc tồn tại
        os.makedirs(DOWNLOAD_PATH, exist_ok=True)

        # Recursively download all content from the specified folder
        download_files_recursively(service, FOLDER_ID, DOWNLOAD_PATH)
        
    except HttpError as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    main()