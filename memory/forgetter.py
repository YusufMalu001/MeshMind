import logging
from typing import Dict, Any
from datetime import datetime, timezone

from memory.store import MemoryStore

logger = logging.getLogger("MeshMind.Forgetter")


class MemoryForgetter:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.config = store.config

    def strategy_1_no_forgetting(self, user_id: str) -> None:
        """Do nothing — keep all memories forever. Used as baseline condition."""
        pass

    def strategy_2_recency_decay(self, user_id: str) -> None:
        """
        For each memory:
          days_old = (now - created_at).days
          halflife = config.recency_decay_halflife_days
          decay_weight = 0.5 ** (days_old / halflife)
          Update memory metadata with new decay_weight
        Memories are NOT deleted, just down-weighted
        """
        memories = self.store.get_all_memories(user_id)
        now = datetime.now(timezone.utc)
        halflife = self.config.get("recency_decay_halflife_days", 7.0)

        for memory in memories:
            created_dt = datetime.fromisoformat(memory.created_at)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            
            # Using total_seconds to calculate precise floating-point days to support 
            # simulation and fast-running benchmarks where times might have fractional days,
            # but default to .days if we strictly want integer-based. Let's compute:
            delta = now - created_dt
            days_old = delta.total_seconds() / 86400.0
            
            decay_weight = 0.5 ** (days_old / halflife)
            # Bound decay weight between 0.0 and 1.0
            decay_weight = max(0.0, min(1.0, decay_weight))
            
            # Update database
            self.store.update_metadata(memory.memory_id, {"decay_weight": decay_weight})
            # Update memory object attribute
            memory.decay_weight = decay_weight

    def strategy_3_relevance_pruning(self, user_id: str) -> None:
        """
        For each memory:
          if retrieval_count < config.min_retrieval_count_to_keep:
            if (now - created_at).days > 14:
              delete_memory(memory_id)
        Memories that were never useful get deleted
        """
        memories = self.store.get_all_memories(user_id)
        now = datetime.now(timezone.utc)
        min_retrieval = self.config.get("min_retrieval_count_to_keep", 2)

        for memory in memories:
            created_dt = datetime.fromisoformat(memory.created_at)
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            
            delta = now - created_dt
            days_old = delta.total_seconds() / 86400.0
            
            if memory.retrieval_count < min_retrieval and days_old > 14.0:
                self.store.delete_memory(memory.memory_id)

    def strategy_4_hybrid(self, user_id: str) -> None:
        """
        Apply recency_decay first
        Then apply relevance_pruning
        Most aggressive forgetting strategy
        """
        self.strategy_2_recency_decay(user_id)
        self.strategy_3_relevance_pruning(user_id)

    def apply_forgetting(self, user_id: str, strategy: str) -> Dict[str, Any]:
        """
        Run specified strategy
        Return report:
        {
          "strategy": str,
          "memories_before": int,
          "memories_after": int,
          "memories_deleted": int,
          "avg_decay_weight": float
        }
        """
        memories_before = self.store.get_memory_count(user_id)
        
        if strategy == "no_forgetting" or strategy == "no_memory":
            self.strategy_1_no_forgetting(user_id)
        elif strategy == "recency_decay":
            self.strategy_2_recency_decay(user_id)
        elif strategy == "relevance_pruning":
            self.strategy_3_relevance_pruning(user_id)
        elif strategy == "hybrid":
            self.strategy_4_hybrid(user_id)
        else:
            logger.warning(f"Unknown forgetting strategy: {strategy}. Defaulting to no_forgetting.")
            self.strategy_1_no_forgetting(user_id)

        memories_after_list = self.store.get_all_memories(user_id)
        memories_after = len(memories_after_list)
        memories_deleted = memories_before - memories_after
        
        avg_decay = 1.0
        if memories_after_list:
            avg_decay = sum(m.decay_weight for m in memories_after_list) / memories_after
            
        report = {
            "strategy": strategy,
            "memories_before": memories_before,
            "memories_after": memories_after,
            "memories_deleted": memories_deleted,
            "avg_decay_weight": avg_decay
        }
        logger.info(f"Forgetting report for user {user_id} ({strategy}): {report}")
        return report
