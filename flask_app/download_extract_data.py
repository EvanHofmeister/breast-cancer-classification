import os
import requests
import zipfile

def download_and_extract_zip(url, data_directory):
    base_data_directory = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', data_directory))
    if not os.path.exists(base_data_directory):
        os.makedirs(base_data_directory)

    response = requests.get(url)
    if response.status_code == 200:
        zip_file_path = os.path.join(base_data_directory, "downloaded_data.zip")
        with open(zip_file_path, 'wb') as file:
            file.write(response.content)
        print(f"Zip file downloaded to {zip_file_path}")

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(base_data_directory)
        print(f"Extracted to {base_data_directory}")

        os.remove(zip_file_path)
        print(f"Removed zip file {zip_file_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    download_url = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"
    download_directory = "data"
    download_and_extract_zip(download_url, download_directory)
