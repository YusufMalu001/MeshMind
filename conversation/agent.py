import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from groq import Groq

from memory.extractor import Memory
from memory.memory_manager import MemoryManager
from conversation.context_builder import ContextBuilder

logger = logging.getLogger("MeshMind.Agent")


@dataclass
class AgentResponse:
    response: str
    memories_used: List[Memory]
    memory_context_injected: str
    latency_ms: float


class ConversationalAgent:
    def __init__(
        self, 
        config: Dict[str, Any], 
        memory_manager: MemoryManager,
        groq_client: Optional[Groq] = None
    ) -> None:
        self.config = config
        self.memory_manager = memory_manager
        self.groq_client = groq_client or Groq()
        self.context_builder = ContextBuilder(config)
        self.groq_model = config.get("groq_model", "llama-3.1-8b-instant")
        
        # Track histories per conversation in memory
        self.histories: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get or initialize history for a conversation."""
        if conversation_id not in self.histories:
            self.histories[conversation_id] = []
        return self.histories[conversation_id]

    def clear_history(self, conversation_id: str) -> None:
        """Clear conversation history."""
        if conversation_id in self.histories:
            self.histories[conversation_id] = []

    def chat(
        self,
        user_message: str,
        user_id: str,
        conversation_id: str,
        memory_condition: str
    ) -> AgentResponse:
        """
        Processes chat turn based on the memory condition:
        - "no_memory": Just respond without memory context.
        - Other conditions: Get memory context, apply forgetting strategy, and respond.
        """
        # Get historical turns for this conversation (excluding system prompt)
        history = self.get_history(conversation_id)
        
        memories_used: List[Memory] = []
        memory_context = ""
        
        if memory_condition != "no_memory":
            # Retrieve relevant memories for context injection
            top_k = self.config.get("retrieval_top_k", 5)
            memories_used = self.memory_manager.retriever.retrieve(
                query=user_message,
                user_id=user_id,
                top_k=top_k
            )
            
            if memories_used:
                lines = [f"- {m.content}" for m in memories_used]
                memory_context = "What I remember about you:\n" + "\n".join(lines)
            
            # Apply appropriate forgetting strategy for this condition
            # Conditions are: "no_forgetting", "recency_decay", "relevance_pruning", "hybrid"
            self.memory_manager.forgetter.apply_forgetting(user_id, memory_condition)

        # Build prompt using the context builder
        messages = self.context_builder.build_messages(
            user_message=user_message,
            history=history,
            memory_context=memory_context
        )

        # Call Groq API with retries and backoff
        retries = [10, 30, 60]
        response_text = ""
        latency_ms = 0.0
        success = False

        for i, wait_time in enumerate(retries + [0]):
            try:
                start_time = time.time()
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=0.7
                )
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000.0
                
                response_text = response.choices[0].message.content or ""
                success = True
                # Pacing sleep to respect rate limits
                time.sleep(1.0)
                break
            except Exception as e:
                if i < len(retries):
                    logger.warning(
                        f"Groq API call failed: {e}. Retrying in {wait_time}s... (Attempt {i+1}/3)"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Groq API call failed after all retries: {e}")
                    response_text = "I'm having some trouble connecting right now, but I'm here."
                    latency_ms = 0.0

        # Update the conversation history with user and assistant turns
        if success:
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": response_text})
            
        return AgentResponse(
            response=response_text,
            memories_used=memories_used,
            memory_context_injected=memory_context,
            latency_ms=latency_ms
        )
