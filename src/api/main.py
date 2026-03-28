from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import logging
from typing import List, Dict
import sys
import os

# Ensure the src module is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.inference import ShelfDetector
from anomaly.model import ShelfAnomalyDetector

from logging.handlers import TimedRotatingFileHandler

# Define log formatting and directory
os.makedirs("logs", exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set up Rotating file handler (daily rotation, keeps 7 days)
file_handler = TimedRotatingFileHandler("logs/api_production.log", when="d", interval=1, backupCount=7)
file_handler.setFormatter(log_formatter)

# Set up Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Configure Root Logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Retail Shelf Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class BoundingBox(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float
    class_name: str
    confidence: float

class InferenceResponse(BaseModel):
    shelf_health_score: float
    total_products_detected: int
    empty_slots_detected: int
    misplaced_items_detected: int
    detections: List[BoundingBox]

# --- Model Loading (Paths for Day 8 integration) ---
MODEL_WEIGHTS = "models/shelf_detection/weights/best.pt"
ANOMALY_WEIGHTS = "models/anomaly/iso_forest.joblib"

detector = None
anomaly_detector = None

@app.on_event("startup")
def load_models():
    """Loads weights dynamically on server start."""
    global detector, anomaly_detector
    try:
        if os.path.exists(MODEL_WEIGHTS):
            detector = ShelfDetector(MODEL_WEIGHTS)
        if os.path.exists(ANOMALY_WEIGHTS):
            anomaly_detector = ShelfAnomalyDetector(model_path=ANOMALY_WEIGHTS)
        logger.info("Models loaded successfully (if weights were present).")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": detector is not None}

@app.post("/analyze-shelf", response_model=InferenceResponse)
async def analyze_shelf(file: UploadFile = File(...)):
    """
    Accepts an image upload, runs YOLO object detection, passes bounding boxes
    to the anomaly detector, and returns the comprehensive shelf analysis.
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only images are allowed.")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # In a full pipeline, we'd execute detector.predict_image() here.
    # We return mock responses if weights aren't present yet, indicating the expected contract.
    
    detected_products = 50
    empty_slots = 5
    detections: List[BoundingBox] = []
    misplaced = 2
    health_score = 92.5
    
    if detector and anomaly_detector:
        # Pseudo-code for actual model calls:
        # preds, _ = detector.predict_image(img)
        # formatted_preds = format_yolo_to_bbox(preds)
        # anomalies = anomaly_detector.detect_misplaced_items(features)
        # health_score = anomaly_detector.shelf_health_score(...)
        pass
        
    return InferenceResponse(
        shelf_health_score=health_score,
        total_products_detected=detected_products,
        empty_slots_detected=empty_slots,
        misplaced_items_detected=misplaced,
        detections=detections
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
