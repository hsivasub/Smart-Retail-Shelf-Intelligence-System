import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShelfAnomalyDetector:
    def __init__(self, contamination: float = 0.05, model_path: str = None):
        """
        Anomaly detection for misplaced products using Isolation Forest.
        Features expected: [x_center, y_center, width, height, class_id_encoded, confidence]
        """
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination, random_state=42)
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading anomaly detection model from {model_path}")
            self.model = joblib.load(model_path)
            self.is_trained = True
        else:
            self.is_trained = False

    def train(self, features: np.ndarray, model_save_path: str = "models/anomaly/iso_forest.joblib"):
        """
        Train the Isolation Forest on historical bounding box features to learn the 'normal' layout.
        """
        logger.info(f"Training Isolation Forest on {features.shape[0]} historical detections...")
        
        import mlflow
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment("Shelf_Anomaly_Detection")
        
        with mlflow.start_run(run_name="IsolationForest_Training"):
            mlflow.log_param("contamination", self.contamination)
            mlflow.log_param("n_samples", features.shape[0])
            
            self.model.fit(features)
            self.is_trained = True
            
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            joblib.dump(self.model, model_save_path)
            
            mlflow.sklearn.log_model(self.model, "iso_forest_model")
            logger.info(f"Model saved to {model_save_path} and logged to MLflow")

    def detect_misplaced_items(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomalies. 
        Returns an array where 1 = normal, -1 = anomaly (misplaced).
        """
        if not self.is_trained:
            logger.error("Model is not trained yet!")
            return None
        
        predictions = self.model.predict(features)
        return predictions

    def shelf_health_score(self, total_slots: int, empty_slots: int, misplaced_items: int) -> float:
        """
        Calculates a robust proprietary shelf health score (0-100).
        """
        if total_slots == 0:
            return 0.0
            
        # Penalize empty slots linearly, penalize misplaced items slightly less
        # Assuming maximum penalty pushes score to 0
        empty_penalty = (empty_slots / total_slots) * 60  # Max 60% penalty
        misplaced_penalty = (misplaced_items / total_slots) * 40 # Max 40% penalty
        
        health_score = 100.0 - empty_penalty - misplaced_penalty
        return max(0.0, round(health_score, 2))


if __name__ == "__main__":
    # Example usage
    # Dummy features: 100 bounding boxes with [x, y, w, h, class, conf]
    dummy_data = np.random.rand(100, 6)
    
    detector = ShelfAnomalyDetector(contamination=0.1)
    detector.train(dummy_data)
    
    # Test on new dummy shelf
    test_shelf = np.random.rand(10, 6)
    anomalies = detector.detect_misplaced_items(test_shelf)
    
    misplaced_count = sum(anomalies == -1)
    health = detector.shelf_health_score(total_slots=10, empty_slots=2, misplaced_items=misplaced_count)
    
    logger.info(f"Detected {misplaced_count} misplaced items.")
    logger.info(f"Calculated Shelf Health Score: {health}/100")
