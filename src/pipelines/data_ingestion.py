import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Simulates downloading/moving dataset from a staging area to local data/raw directory"""
        logger.info(f"Starting data ingestion from {self.source_dir} to {self.target_dir}")
        if not self.source_dir.exists():
            logger.error(f"Source directory {self.source_dir} not found.")
            return False
            
        copied_files = 0
        for item in self.source_dir.glob('*.*'):
            if item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy2(item, self.target_dir / item.name)
                copied_files += 1
                
        logger.info(f"Successfully ingested {copied_files} images.")
        return True

if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion("staging_data", "data/raw")
    ingestion.load_data()
