import cv2
import os
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, input_dir: str, output_dir: str, target_size=(640, 640)):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_images(self):
        logger.info(f"Starting preprocessing. Target size: {self.target_size}")
        if not self.input_dir.exists():
            logger.warning(f"Input directory {self.input_dir} does not exist.")
            return

        processed_count = 0
        for img_path in self.input_dir.glob('*.*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to read image {img_path}")
                continue
                
            # Resize while maintaining aspect ratio (padding)
            img_resized = self._letterbox(img, self.target_size)
            
            # Save processed image
            out_path = self.output_dir / img_path.name
            cv2.imwrite(str(out_path), img_resized)
            processed_count += 1
            
        logger.info(f"Preprocessed {processed_count} images.")

    def _letterbox(self, img, new_shape, color=(114, 114, 114)):
        """Resize image to a 32-pixel-multiple rectangle, preserving aspect ratio (padding)."""
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # stride-multiple padding
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img

if __name__ == "__main__":
    preprocessor = ImagePreprocessor("data/raw", "data/processed")
    preprocessor.process_images()
