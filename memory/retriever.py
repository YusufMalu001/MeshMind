import logging
from typing import List, Dict, Any
from datetime import datetime, timezone

from memory.extractor import Memory
from memory.store import MemoryStore

logger = logging.getLogger("MeshMind.Retriever")


class MemoryRetriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.config = store.config
        self.embedding_model = store.embedding_model
        self.collection = store.collection

    def retrieve(self, query: str, user_id: str, top_k: int = 5) -> List[Memory]:
        """
        Query and rank memory retrieval.
        Step 1: Embed query.
        Step 2: ChromaDB similarity search filtered by user_id.
        Step 3: Get top 20 candidates by cosine similarity.
        Step 4: Re-rank by composite score:
          final_score = (
            0.5 * cosine_similarity +
            0.3 * memory.importance_score +
            0.2 * memory.decay_weight
          )
        Step 5: Return top_k after re-ranking.
        Step 6: Update last_retrieved_at and retrieval_count for returned memories.
        """
        try:
            # Step 1: Embed query
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Step 2 & 3: Query ChromaDB for top 20 candidates by similarity
            results = self.collection.query(
                query_embeddings=[query_embedding],
                where={"user_id": user_id},
                n_results=20
            )
            
            if not results or not results.get("ids") or not results["ids"][0]:
                return []
            
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            candidates: List[Dict[str, Any]] = []
            for i in range(len(ids)):
                memory_obj = Memory.from_dict(metadatas[i])
                distance = distances[i]
                
                # Convert cosine distance to cosine similarity: sim = 1 - distance
                cosine_similarity = max(-1.0, min(1.0, 1.0 - distance))
                
                # Step 4: Re-rank by composite score
                sim_contrib = 0.5 * cosine_similarity
                imp_contrib = 0.3 * memory_obj.importance_score
                dec_contrib = 0.2 * memory_obj.decay_weight
                
                final_score = sim_contrib + imp_contrib + dec_contrib
                
                candidates.append({
                    "memory": memory_obj,
                    "final_score": final_score,
                    "cosine_similarity": cosine_similarity
                })
            
            # Sort by final score desc
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
            
            # Step 5: Return top_k
            top_candidates = candidates[:top_k]
            retrieved_memories: List[Memory] = []
            
            now_iso = datetime.now(timezone.utc).isoformat()
            
            # Step 6: Update metadata in ChromaDB and returned objects
            for cand in top_candidates:
                mem = cand["memory"]
                new_retrieval_count = mem.retrieval_count + 1
                
                # Update attributes locally
                mem.last_retrieved_at = now_iso
                mem.retrieval_count = new_retrieval_count
                
                # Update database
                self.store.update_metadata(mem.memory_id, {
                    "last_retrieved_at": now_iso,
                    "retrieval_count": new_retrieval_count
                })
                
                retrieved_memories.append(mem)
                
            return retrieved_memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories for query '{query}': {e}")
            return []

    def retrieve_with_explanation(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        Same as retrieve() but also returns:
        - why each memory was retrieved (which factor dominated: similarity/importance/recency)
        - retrieval scores breakdown
        """
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                where={"user_id": user_id},
                n_results=20
            )
            
            if not results or not results.get("ids") or not results["ids"][0]:
                return {"memories": [], "explanations": []}
            
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            candidates: List[Dict[str, Any]] = []
            for i in range(len(ids)):
                memory_obj = Memory.from_dict(metadatas[i])
                distance = distances[i]
                
                cosine_similarity = max(-1.0, min(1.0, 1.0 - distance))
                
                sim_contrib = 0.5 * cosine_similarity
                imp_contrib = 0.3 * memory_obj.importance_score
                dec_contrib = 0.2 * memory_obj.decay_weight
                
                final_score = sim_contrib + imp_contrib + dec_contrib
                
                # Determine dominant factor
                contribs = {
                    "similarity": sim_contrib,
                    "importance": imp_contrib,
                    "recency": dec_contrib
                }
                dominant_factor = max(contribs, key=contribs.get)
                
                candidates.append({
                    "memory": memory_obj,
                    "final_score": final_score,
                    "cosine_similarity": cosine_similarity,
                    "sim_contrib": sim_contrib,
                    "imp_contrib": imp_contrib,
                    "dec_contrib": dec_contrib,
                    "dominant_factor": dominant_factor
                })
                
            # Sort by final score desc
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
            
            # Fetch top_k (from config, default to 5)
            top_k = self.config.get("retrieval_top_k", 5)
            top_candidates = candidates[:top_k]
            
            retrieved_memories: List[Memory] = []
            explanations: List[Dict[str, Any]] = []
            
            now_iso = datetime.now(timezone.utc).isoformat()
            
            for cand in top_candidates:
                mem = cand["memory"]
                new_retrieval_count = mem.retrieval_count + 1
                
                mem.last_retrieved_at = now_iso
                mem.retrieval_count = new_retrieval_count
                
                self.store.update_metadata(mem.memory_id, {
                    "last_retrieved_at": now_iso,
                    "retrieval_count": new_retrieval_count
                })
                
                retrieved_memories.append(mem)
                
                explanations.append({
                    "memory_id": mem.memory_id,
                    "content": mem.content,
                    "composite_score": cand["final_score"],
                    "similarity_score": cand["cosine_similarity"],
                    "similarity_contribution": cand["sim_contrib"],
                    "importance_score": mem.importance_score,
                    "importance_contribution": cand["imp_contrib"],
                    "decay_weight": mem.decay_weight,
                    "decay_contribution": cand["dec_contrib"],
                    "dominant_factor": cand["dominant_factor"]
                })
                
            return {
                "memories": retrieved_memories,
                "explanations": explanations
            }
            
        except Exception as e:
            logger.error(f"Failed retrieve_with_explanation for query '{query}': {e}")
            return {"memories": [], "explanations": []}
