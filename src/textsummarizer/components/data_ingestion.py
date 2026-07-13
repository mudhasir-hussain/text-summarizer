import os
import urllib.request as request
import zipfile
import shutil
from textsummarizer.logging import logger
from textsummarizer.utils.common import get_size
from textsummarizer.entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            # Check if we have the zip file in the root directory already
            local_zip_backup = Path("summarizer-data.zip")
            if local_zip_backup.exists():
                logger.info(f"Found local zip backup at {local_zip_backup}. Copying to destination...")
                shutil.copy(local_zip_backup, self.config.local_data_file)
                logger.info(f"File copied successfully: {self.config.local_data_file}")
            else:
                logger.info(f"Downloading dataset from {self.config.source_URL}...")
                filename, headers = request.urlretrieve(
                    url = self.config.source_URL,
                    filename = self.config.local_data_file
                )
                logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
