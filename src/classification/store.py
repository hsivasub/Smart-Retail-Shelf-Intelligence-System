import numpy as np
import os
import logging
import json

logger = logging.getLogger(__name__)

class ReferenceVectorStore:
    def __init__(self, storage_path: str = "models/classification/reference_vectors.npz"):
        """
        Manages the storage and retrieval of 'Gold Standard' product embeddings.
        Stored as a mapping: shelf_id_slot_id -> embedding_vector
        """
        self.storage_path = storage_path
        self.vectors = {}
        self.load()

    def add_reference(self, shelf_id: str, slot_id: int, embedding: np.ndarray, sku_id: str = "unknown"):
        """
        Adds or updates a reference embedding for a specific shelf slot.
        """
        key = f"{shelf_id}_{slot_id}"
        self.vectors[key] = {
            "vector": embedding,
            "sku_id": sku_id
        }
        logger.info(f"Added reference for {key} (SKU: {sku_id})")

    def get_reference(self, shelf_id: str, slot_id: int):
        """
        Retrieves the reference embedding and SKU info for a specific slot.
        """
        key = f"{shelf_id}_{slot_id}"
        return self.vectors.get(key)

    def save(self):
        """
        Persists the vectors to a compressed NumPy file.
        """
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        # Separate vectors and metadata for NPZ storage
        save_data = {k: v["vector"] for k, v in self.vectors.items()}
        # Store metadata (SKU IDs) in a sibling JSON file
        metadata = {k: v["sku_id"] for k, v in self.vectors.items()}
        
        np.savez_compressed(self.storage_path, **save_data)
        
        metadata_path = self.storage_path.replace(".npz", "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        logger.info(f"Saved {len(self.vectors)} references to {self.storage_path}")

    def load(self):
        """
        Loads vectors from disk if they exist.
        """
        if not os.path.exists(self.storage_path):
            logger.info("No reference storage found. Starting with empty store.")
            return

        try:
            data = np.load(self.storage_path)
            metadata_path = self.storage_path.replace(".npz", "_metadata.json")
            
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            for key in data.files:
                self.vectors[key] = {
                    "vector": data[key],
                    "sku_id": metadata.get(key, "unknown")
                }
            logger.info(f"Loaded {len(self.vectors)} references from {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to load reference storage: {e}")

if __name__ == "__main__":
    # Test Store
    logging.basicConfig(level=logging.INFO)
    store = ReferenceVectorStore("models/classification/test_refs.npz")
    
    dummy_vec = np.random.rand(512).astype(np.float32)
    store.add_reference("shelf_001", 1, dummy_vec, "Coke_500ml")
    store.save()
    
    # Reload and check
    new_store = ReferenceVectorStore("models/classification/test_refs.npz")
    ref = new_store.get_reference("shelf_001", 1)
    print(f"Retrieved SKU: {ref['sku_id']}, Vector Shape: {ref['vector'].shape}")
