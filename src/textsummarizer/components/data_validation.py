import os
from textsummarizer.logging import logger
from textsummarizer.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True
            
            for file in self.config.ALL_REQUIRED_FILES:
                path = os.path.join("artifacts", "data_ingestion", file)
                if not os.path.exists(path):
                    validation_status = False
                    logger.warning(f"File not found during validation: {path}")
                    break
                    
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Data Validation check complete. Status written to status.txt: {validation_status}")
            return validation_status
        except Exception as e:
            raise e
