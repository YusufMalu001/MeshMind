import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from memory.extractor import Memory, MemoryExtractor
from memory.store import MemoryStore
from memory.retriever import MemoryRetriever
from memory.forgetter import MemoryForgetter

logger = logging.getLogger("MeshMind.Manager")


@dataclass
class MemoryOps:
    new_memories: List[Memory]
    retrieved_memories: List[Memory]


class MemoryManager:
    def __init__(self, config: Dict[str, Any], collection_name: Optional[str] = None) -> None:
        self.config = config
        self.store = MemoryStore(config, collection_name=collection_name)
        self.extractor = MemoryExtractor(config)
        self.retriever = MemoryRetriever(self.store)
        self.forgetter = MemoryForgetter(self.store)

    def process_turn(
        self, 
        user_message: str, 
        conversation_id: str, 
        user_id: str, 
        context: List[str]
    ) -> MemoryOps:
        """
        Orchestrate memory operations for a single conversation turn:
        1. Extract memories from user_message
        2. Store new memories (enforcing maximum memories constraint)
        3. Retrieve relevant memories for the current query/turn
        4. Return MemoryOps containing both new and retrieved memories
        """
        # Convert list of context turns to a single string
        context_str = "\n".join(context)
        
        # 1. Extract memories
        new_memories = self.extractor.extract_memories(
            conversation_turn=user_message,
            context=context_str,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # 2. Store new memories
        max_memories = self.config.get("max_memories_per_user", 100)
        for memory in new_memories:
            current_count = self.store.get_memory_count(user_id)
            if current_count >= max_memories:
                # Get all memories (sorted created_at desc) and delete the oldest (at the end)
                all_memories = self.store.get_all_memories(user_id)
                if all_memories:
                    oldest_memory = all_memories[-1]
                    logger.warning(
                        f"Memory limit ({max_memories}) reached for user {user_id}. "
                        f"Deleting oldest memory: {oldest_memory.memory_id}"
                    )
                    self.store.delete_memory(oldest_memory.memory_id)
            
            self.store.add_memory(memory)
            
        # 3. Retrieve relevant memories for the current query
        top_k = self.config.get("retrieval_top_k", 5)
        retrieved_memories = self.retriever.retrieve(
            query=user_message,
            user_id=user_id,
            top_k=top_k
        )
        
        return MemoryOps(
            new_memories=new_memories,
            retrieved_memories=retrieved_memories
        )

    def get_memory_context(self, query: str, user_id: str) -> str:
        """
        Retrieve relevant memories.
        Format as context string:
        "What I remember about you:
        - [memory 1]
        - [memory 2]
        ..."
        Return formatted string for LLM injection, or empty string if no memories are found.
        """
        top_k = self.config.get("retrieval_top_k", 5)
        retrieved = self.retriever.retrieve(query=query, user_id=user_id, top_k=top_k)
        
        if not retrieved:
            return ""
            
        lines = []
        for memory in retrieved:
            lines.append(f"- {memory.content}")
            
        context_str = "What I remember about you:\n" + "\n".join(lines)
        return context_str
