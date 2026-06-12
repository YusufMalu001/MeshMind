import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("MeshMind.ContextBuilder")


class ContextBuilder:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def build_system_prompt(self, memory_context: Optional[str] = None) -> str:
        """
        Build the prompt for the conversational agent:
        system: f\"\"\"You are Ira, a warm and personal AI 
        companion. You have a great memory for details 
        about the people you talk to.
        
        {memory_context if memory_context else ''}
        
        Rules:
        - Reference remembered facts naturally when relevant
        - Never say "As an AI I don't have memory"
        - Be warm, personal, and contextually aware
        - If you remember something relevant, use it\"\"\"
        """
        context_section = f"\n{memory_context}\n" if memory_context else ""
        
        prompt = (
            "You are Ira, a warm and personal AI companion. "
            "You have a great memory for details about the people you talk to.\n"
            f"{context_section}"
            "Rules:\n"
            "- Reference remembered facts naturally when relevant\n"
            "- Never say \"As an AI I don't have memory\"\n"
            "- Be warm, personal, and contextually aware\n"
            "- If you remember something relevant, use it"
        )
        return prompt

    def build_messages(
        self, 
        user_message: str, 
        history: List[Dict[str, str]], 
        memory_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Combine system prompt, history, and current user message.
        History is a list of dictionaries: [{"role": "user"|"assistant", "content": "..."}]
        """
        messages = []
        
        # 1. System Prompt (incorporating memory context)
        system_content = self.build_system_prompt(memory_context)
        messages.append({"role": "system", "content": system_content})
        
        # 2. History
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
            
        # 3. Current User Message
        messages.append({"role": "user", "content": user_message})
        
        return messages
