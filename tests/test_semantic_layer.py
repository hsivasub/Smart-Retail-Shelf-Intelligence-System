import unittest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from classification.embedding import SKUEmbedder
from anomaly.semantic import SemanticVerifier

class TestSemanticLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = SKUEmbedder(device="cpu")
        cls.verifier = SemanticVerifier(threshold=0.85)

    def test_embedding_extraction(self):
        # Create a dummy image
        dummy_crop = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        embedding = self.embedder.get_embedding(dummy_crop)
        
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 512) # ResNet18 embedding size

    def test_similarity_identical(self):
        v1 = np.random.rand(512).astype(np.float32)
        similarity = self.verifier.compare(v1, v1)
        self.assertAlmostEqual(similarity, 1.0, places=5)
        self.assertFalse(self.verifier.is_mismatch(v1, v1))

    def test_mismatch_detection(self):
        # Two very different vectors
        v1 = np.array([1.0] * 512, dtype=np.float32)
        v2 = np.array([-1.0] * 512, dtype=np.float32)
        
        similarity = self.verifier.compare(v1, v2)
        self.assertTrue(similarity < 0.85)
        self.assertTrue(self.verifier.is_mismatch(v1, v2))

if __name__ == "__main__":
    unittest.main()
