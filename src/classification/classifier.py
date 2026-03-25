import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SKUClassifier:
    def __init__(self, num_classes: int, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        # Load pre-trained ResNet18 and modify the final layer
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_crop(self, cv2_crop: np.ndarray):
        """
        Runs image classification on a single cropped product bounding box.
        """
        if cv2_crop is None or cv2_crop.size == 0:
            return None
            
        # Convert BGR (cv2) to RGB
        crop_rgb = cv2.cvtColor(cv2_crop, cv2.COLOR_BGR2RGB)
        
        # Transform and add batch dimension
        img_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()

if __name__ == "__main__":
    logger.info("SKUClassifier module loaded. Use this class after YOLO detection crops.")
