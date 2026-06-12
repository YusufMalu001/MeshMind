import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from memory.extractor import Memory

logger = logging.getLogger("MeshMind.Store")


class MemoryStore:
    def __init__(self, config: Dict[str, Any], collection_name: Optional[str] = None) -> None:
        self.config = config
        
        # Determine the database persistence directory
        results_dir = Path(config.get("results_dir", "./results"))
        db_path = results_dir / "chroma_db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        # Collection name setup
        self.collection_name = collection_name or config.get(
            "chroma_collection", "meshmind_memories"
        )
        
        # Initialize SentenceTransformer model
        embedding_model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
        # CPU-only compatibility is standard for SentenceTransformer; device='cpu' forces it
        self.embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
        
        # Create or get collection with cosine space setting
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized ChromaDB collection: {self.collection_name} in {db_path}")

    def add_memory(self, memory: Memory) -> bool:
        """
        Embed memory.content using SentenceTransformer.
        Store in ChromaDB with full metadata.
        Return success bool.
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(memory.content).tolist()
            
            # Save to ChromaDB
            self.collection.add(
                ids=[memory.memory_id],
                embeddings=[embedding],
                metadatas=[memory.to_dict()],
                documents=[memory.content]
            )
            logger.debug(f"Added memory: {memory.memory_id} to collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add memory {memory.memory_id}: {e}")
            return False

    def get_all_memories(self, user_id: str) -> List[Memory]:
        """
        Query ChromaDB for all memories by user_id.
        Return sorted by created_at desc.
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id}
            )
            
            memories: List[Memory] = []
            if results and "metadatas" in results and results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata:
                        memories.append(Memory.from_dict(metadata))
            
            # Sort by created_at desc
            memories.sort(key=lambda x: x.created_at, reverse=True)
            return memories
        except Exception as e:
            logger.error(f"Failed to retrieve memories for user {user_id}: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Remove from ChromaDB by ID."""
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def update_metadata(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific metadata fields.
        Used by forgetter to update decay_weight and retrieval_count.
        """
        try:
            # Retrieve existing metadata
            existing = self.collection.get(ids=[memory_id])
            if not existing or not existing["metadatas"]:
                logger.warning(f"Memory {memory_id} not found for metadata update.")
                return False
            
            current_metadata = dict(existing["metadatas"][0])
            current_metadata.update(updates)
            
            # Ensure types are correct for metadata (e.g. float and int compatibility)
            if "decay_weight" in current_metadata:
                current_metadata["decay_weight"] = float(current_metadata["decay_weight"])
            if "retrieval_count" in current_metadata:
                current_metadata["retrieval_count"] = int(current_metadata["retrieval_count"])
            if "importance_score" in current_metadata:
                current_metadata["importance_score"] = float(current_metadata["importance_score"])

            self.collection.update(
                ids=[memory_id],
                metadatas=[current_metadata]
            )
            logger.debug(f"Updated metadata for memory {memory_id}: {updates}")
            return True
        except Exception as e:
            logger.error(f"Failed to update metadata for memory {memory_id}: {e}")
            return False

    def get_memory_count(self, user_id: str) -> int:
        """Count total memories for user."""
        try:
            results = self.collection.get(
                where={"user_id": user_id}
            )
            if results and "ids" in results:
                return len(results["ids"])
            return 0
        except Exception as e:
            logger.error(f"Failed to get memory count for user {user_id}: {e}")
            return 0
