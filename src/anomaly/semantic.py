import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class SemanticVerifier:
    def __init__(self, threshold: float = 0.85):
        """
        SemanticVerifier compares visual embeddings to detect product mismatches.
        A similarity score below the threshold indicates a potential swap.
        """
        self.threshold = threshold

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        Returns a score between 0 and 1.
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        # Reshape to 2D arrays for sklearn
        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(e1, e2)[0][0]
        return float(similarity)

    def is_mismatch(self, current_embedding: np.ndarray, expected_embedding: np.ndarray) -> bool:
        """
        Returns True if the similarity is below the threshold.
        """
        similarity = self.compare(current_embedding, expected_embedding)
        logger.info(f"Semantic similarity: {similarity:.4f} (threshold: {self.threshold})")
        return similarity < self.threshold

if __name__ == "__main__":
    # Test verifier
    verifier = SemanticVerifier(threshold=0.8)
    v1 = np.random.rand(512)
    v2 = np.random.rand(512)
    
    mismatch = verifier.is_mismatch(v1, v2)
    print(f"Mismatch detected: {mismatch}")
    
    # Test identical
    mismatch_self = verifier.is_mismatch(v1, v1)
    print(f"Self-mismatch detected: {mismatch_self}")
