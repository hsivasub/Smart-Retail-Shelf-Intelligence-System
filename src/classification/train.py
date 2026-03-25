import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
import os
import sys

# Add parent directory to path to import classifier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from classifier import SKUClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_sku_classifier(data_dir: str, num_classes: int, epochs: int = 10, batch_size: int = 32):
    """
    Trains the ResNet sku classification model on cropped product images.
    Expects data_dir to contain standard torchvision ImageFolder structure.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Just a sanity structure check
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} not found. Skipping dataset load.")
        return
        
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Init custom model wrapper
    classifier = SKUClassifier(num_classes=num_classes, device=device)
    model = classifier.model
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Train only final layer initially
    
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
        
    os.makedirs("models/classification", exist_ok=True)
    torch.save(model.state_dict(), "models/classification/sku_resnet.pt")
    logger.info("Model saved to models/classification/sku_resnet.pt")

if __name__ == "__main__":
    train_sku_classifier(data_dir="data/crops/train", num_classes=10, epochs=5)
