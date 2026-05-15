import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SKUEmbedder:
    def __init__(self, model_type: str = "resnet18", model_path: str = None, device: str = "cpu"):
        """
        SKUEmbedder extracts high-dimensional visual features (embeddings) from product crops.
        """
        self.device = torch.device(device)
        
        # Initialize backbone
        if model_type == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Remove the final fully connected layer to get embeddings (512-dim for ResNet18)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        elif model_type == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Remove classifier
            self.model.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_path and os.path.exists(model_path):
            logger.info(f"Loading weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, cv2_crop: np.ndarray) -> np.ndarray:
        """
        Extracts a feature vector for a given image crop.
        """
        if cv2_crop is None or cv2_crop.size == 0:
            return None
            
        crop_rgb = cv2.cvtColor(cv2_crop, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
            # Flatten the output
            embedding = embedding.view(embedding.size(0), -1)
            
        return embedding.cpu().numpy().flatten()

if __name__ == "__main__":
    # Test extraction
    embedder = SKUEmbedder(device="cpu")
    dummy_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    vec = embedder.get_embedding(dummy_crop)
    logger.info(f"Extracted embedding with shape: {vec.shape}")
