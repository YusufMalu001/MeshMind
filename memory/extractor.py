import time
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from groq import Groq

# Configure logging
logger = logging.getLogger("MeshMind.Extractor")

@dataclass
class Memory:
    memory_id: str
    user_id: str
    content: str
    memory_type: str  # personal_fact | preference | event | emotional_state
    importance_score: float  # 0.0 - 1.0
    created_at: str  # ISO timestamp
    last_retrieved_at: str  # ISO timestamp
    retrieval_count: int
    conversation_id: str
    decay_weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert Memory to dictionary for JSON or ChromaDB metadata serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Reconstruct Memory object from dictionary/metadata."""
        # Ensure correct types for numeric values from ChromaDB metadata
        return cls(
            memory_id=str(data["memory_id"]),
            user_id=str(data["user_id"]),
            content=str(data["content"]),
            memory_type=str(data["memory_type"]),
            importance_score=float(data["importance_score"]),
            created_at=str(data["created_at"]),
            last_retrieved_at=str(data["last_retrieved_at"]),
            retrieval_count=int(data["retrieval_count"]),
            conversation_id=str(data["conversation_id"]),
            decay_weight=float(data.get("decay_weight", 1.0))
        )


class MemoryExtractor:
    def __init__(self, config: Dict[str, Any], groq_client: Optional[Groq] = None) -> None:
        self.config = config
        self.groq_model = config.get("groq_model", "llama-3.1-8b-instant")
        self.groq_client = groq_client or Groq()

    def extract_memories(
        self, 
        conversation_turn: str, 
        context: str, 
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> List[Memory]:
        """
        Extract factual, personal, and preference-based memories from conversations.
        Calls Groq API with retries and exponential backoff.
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        system_prompt = (
            "You are a memory extraction system. Extract factual, personal, and preference-based "
            "memories from conversations. Return ONLY a JSON array. Each memory must have:\n"
            "- content: one sentence fact about the user\n"
            "- memory_type: personal_fact|preference|event|emotional_state\n"
            "- importance_score: 0.0-1.0 (how useful for future personalization)\n\n"
            "Rules:\n"
            "- Only extract memories about the USER not the AI\n"
            "- Ignore generic chitchat with no factual content\n"
            "- Max 3 memories per turn\n"
            "- Return [] if nothing worth remembering"
        )

        user_prompt = (
            f"Conversation turn: {conversation_turn}\n"
            f"Previous context: {context}\n"
            "Extract memories:"
        )

        retries = [10, 30, 60]
        response_text = ""
        success = False

        for i, wait_time in enumerate(retries + [0]):
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}  # Request JSON response
                )
                response_text = response.choices[0].message.content or ""
                success = True
                break
            except Exception as e:
                if i < len(retries):
                    logger.warning(
                        f"Groq API call failed: {e}. Retrying in {wait_time}s... (Attempt {i+1}/3)"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Groq API call failed after all retries: {e}")
                    return []

        if not success or not response_text.strip():
            return []

        # JSON parsing
        try:
            # Strip markdown code blocks if any
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            parsed_json = json.loads(clean_text)
            
            # The prompt requests a JSON array, but sometimes LLM response format {"type": "json_object"}
            # might wrap the array in a key, e.g. {"memories": [...]}.
            # Let's handle both.
            raw_memories = []
            if isinstance(parsed_json, list):
                raw_memories = parsed_json
            elif isinstance(parsed_json, dict):
                # Search for any list inside the dictionary keys
                for key, val in parsed_json.items():
                    if isinstance(val, list):
                        raw_memories = val
                        break
                else:
                    # If it's a dict but no list inside, maybe it is a single memory item
                    if "content" in parsed_json:
                        raw_memories = [parsed_json]
            
            memories: List[Memory] = []
            now_iso = datetime.now(timezone.utc).isoformat()

            for item in raw_memories[:3]:  # Limit to max 3 memories
                if not isinstance(item, dict) or "content" not in item:
                    continue

                content = item.get("content", "").strip()
                if not content:
                    continue

                memory_type = item.get("memory_type", "personal_fact")
                if memory_type not in ["personal_fact", "preference", "event", "emotional_state"]:
                    memory_type = "personal_fact"

                # Parse LLM-assigned importance score
                try:
                    llm_score = float(item.get("importance_score", 0.5))
                except (ValueError, TypeError):
                    llm_score = 0.5

                # Construct temporary memory object
                temp_mem = Memory(
                    memory_id=str(uuid.uuid4()),
                    user_id=user_id,
                    content=content,
                    memory_type=memory_type,
                    importance_score=llm_score,
                    created_at=now_iso,
                    last_retrieved_at=now_iso,
                    retrieval_count=0,
                    conversation_id=conversation_id,
                    decay_weight=1.0
                )

                # Compute final assigned importance score
                final_importance = self.assign_importance(temp_mem)
                temp_mem.importance_score = final_importance
                
                memories.append(temp_mem)

            return memories

        except Exception as e:
            logger.error(f"Error parsing memory JSON response: {e}. Raw content: {response_text}")
            return []

    def assign_importance(self, memory: Memory) -> float:
        """
        Score based on memory_type:
        personal_fact: 0.8
        preference: 0.7
        event: 0.6
        emotional_state: 0.5
        Multiply by LLM-assigned score, normalize to 0-1
        """
        multipliers = {
            "personal_fact": 0.8,
            "preference": 0.7,
            "event": 0.6,
            "emotional_state": 0.5
        }
        multiplier = multipliers.get(memory.memory_type, 0.5)
        # Multiply and clamp to range 0.0 - 1.0
        score = memory.importance_score * multiplier
        return max(0.0, min(1.0, score))
