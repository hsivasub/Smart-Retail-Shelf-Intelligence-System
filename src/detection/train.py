import os
import logging
import mlflow
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_yolo_model(data_yaml_path: str, epochs: int = 50, imgsz: int = 640, batch_size: int = 16):
    """
    Trains YOLOv8 model on custom dataset (SKU110k formatted to YOLO format).
    Classes: 0: product, 1: empty_slot
    """
    if not os.path.exists(data_yaml_path):
        logger.error(f"Dataset configuration {data_yaml_path} not found.")
        return
        
    logger.info("Initializing YOLOv8s model...")
    # Load pre-trained model for transfer learning
    model = YOLO("yolov8s.pt")
    
    # Initialize MLflow experiment
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Shelf_Object_Detection")
    
    logger.info(f"Starting training for {epochs} epochs with image size {imgsz}...")
    
    # Ultralytics natively logs to active MLflow runs if mlflow is installed
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=10,        # Early stopping
        save=True,          # Save best weights
        device="cpu",       # Change to '0' for CUDA GPU
        project="models",   # Directory to save runs
        name="shelf_detection",
        exist_ok=True
    )
    
    logger.info(f"Training completed. Weights saved in models/shelf_detection/weights/best.pt")
    return results

if __name__ == "__main__":
    # Ensure this runs from the root of the project
    data_config = "data/dataset.yaml"
    train_yolo_model(data_yaml_path=data_config, epochs=50, imgsz=640)
