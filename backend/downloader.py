import os
import requests
import logging
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_fixtures(league_name, year):
    # URL of the CSV file page
    data_url = f"https://fixturedownload.com/download/csv/{league_name}-{year}"
    logging.info(f"Fetching download page from URL: {data_url}")

    # Specify the download location
    download_path = f"backend/files/input/{league_name}/{year}.csv"
    logging.info(f"Checking file at path: {download_path}")

    # Fetch the page content
    response = requests.get(data_url)
    response.raise_for_status()

    # Parse the HTML to find the direct CSV link
    soup = BeautifulSoup(response.text, 'html.parser')
    link = soup.find('a', text="click here to download")
    if link:
        csv_url = "https://fixturedownload.com" + link['href']
        csv_response = requests.get(csv_url, stream=True)
        csv_response.raise_for_status()

        # Get the size of the new file from the headers (if provided)
        new_file_size = int(csv_response.headers.get('Content-Length', 0))

        # Check if the file exists and compare sizes
        if os.path.exists(download_path):
            existing_file_size = os.path.getsize(download_path)
            if existing_file_size >= new_file_size:
                logging.info(f"File already exists and is up-to-date (size: {existing_file_size} bytes).")
                return
            else:
                logging.info(f"Existing file is smaller (size: {existing_file_size} bytes). Re-downloading.")

        # Save the file to the specified path
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        with open(download_path, "wb") as file:
            file.write(csv_response.content)
        logging.info(f"File successfully downloaded to: {download_path} (size: {new_file_size} bytes).")
    else:
        logging.warning("Download link not found.")
