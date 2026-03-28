import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import logging
import os

logger = logging.getLogger(__name__)

def generate_drift_report(reference_data_path: str, current_data_path: str, output_path: str = "reports/drift_report.html"):
    """
    Generates an Evidently data drift report comparing reference data to current data.
    """
    logger.info(f"Generating drift report comparing '{reference_data_path}' to '{current_data_path}'")
    try:
        if not os.path.exists(reference_data_path):
            logger.warning(f"Reference data missing at {reference_data_path}. Cannot generate report.")
            return False
        if not os.path.exists(current_data_path):
            logger.warning(f"Current data missing at {current_data_path}. Cannot generate report.")
            return False
            
        reference = pd.read_csv(reference_data_path)
        current = pd.read_csv(current_data_path)
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report.save_html(output_path)
        logger.info(f"Data drift report generated successfully: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error generating data drift report: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Example test usage:
    # generate_drift_report("data/reference_logs.csv", "data/inference_logs.csv")
