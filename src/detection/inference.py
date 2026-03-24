import os
import cv2
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShelfDetector:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        
    def predict_image(self, image_path: str, conf_threshold: float = 0.5):
        """
        Runs inference on a single image.
        Returns the predictions and the annotated image.
        """
        logger.info(f"Running inference on {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to read image {image_path}")
            return None, None
            
        results = self.model(img, conf=conf_threshold)
        
        # results[0].plot() returns the image array with bounding boxes drawn
        annotated_img = results[0].plot()
        
        return results[0], annotated_img
        
    def save_prediction(self, image_path: str, output_dir: str):
        """
        Runs prediction and saves the annotated image to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"pred_{img_name}")
        
        _, annotated_img = self.predict_image(image_path)
        
        if annotated_img is not None:
            cv2.imwrite(output_path, annotated_img)
            logger.info(f"Saved annotated image to {output_path}")
            return output_path
        return None

if __name__ == "__main__":
    # Example local usage
    model_weights = "models/shelf_detection/weights/best.pt"
    test_image = "data/images/test/sample.jpg" 
    
    # Just a mock check, requires weights to run
    if os.path.exists(model_weights) and os.path.exists(test_image):
        detector = ShelfDetector(model_weights)
        detector.save_prediction(test_image, "data/processed")
    else:
        logger.info("Provide valid weights and test image paths to run locally.")
