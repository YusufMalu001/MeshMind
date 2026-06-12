import os
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from groq import Groq

from memory.memory_manager import MemoryManager
from conversation.agent import ConversationalAgent
from evaluation.metrics import compute_all_metrics

logger = logging.getLogger("MeshMind.Benchmark")


class BenchmarkRunner:
    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        # Load config.yaml
        import yaml
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.results_dir = Path(self.config.get("results_dir", "./results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load conversations
        conv_path = Path("evaluation/conversations.json")
        with open(conv_path, "r") as f:
            self.conversations = json.load(f)
            
        # Initialize Groq Client
        from dotenv import load_dotenv
        load_dotenv()
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Please check your .env file.")
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Generate unique run_id for this benchmark run
        self.run_id = str(uuid.uuid4())[:8]
        logger.info(f"Initialized Benchmark Runner. Run ID: {self.run_id}")

    def run_condition(self, condition_name: str, force: bool = False) -> Dict[str, Any]:
        """
        Run a specific memory condition across all 20 conversations.
        Resumable: loads existing results from file if available.
        """
        result_file = self.results_dir / f"{condition_name}_results.json"
        
        existing_results: Dict[str, Any] = {}
        if result_file.exists() and not force:
            try:
                with open(result_file, "r") as f:
                    existing_results = json.load(f)
                logger.info(f"Loaded existing results for condition '{condition_name}' from {result_file}")
            except Exception as e:
                logger.warning(f"Could not load existing results for '{condition_name}': {e}. Re-running.")
        
        # Structure for storing results
        condition_results = existing_results.get("conversations", {})
        
        # Filter conversations that need to be run
        pending_conversations = [
            c for c in self.conversations 
            if c["conversation_id"] not in condition_results
        ]
        
        if not pending_conversations:
            logger.info(f"All conversations for condition '{condition_name}' are already completed.")
            return existing_results

        logger.info(
            f"Running condition '{condition_name}': {len(pending_conversations)} / "
            f"{len(self.conversations)} conversations pending."
        )

        for conv in tqdm(pending_conversations, desc=f"Condition: {condition_name}"):
            conv_id = conv["conversation_id"]
            user_id = conv["user_id"]
            ground_truth = conv["ground_truth_relevant_memories"]
            
            # Isolated ChromaDB collection per conversation in this run
            collection_name = f"meshmind_{condition_name}_{self.run_id}_{conv_id}"
            
            # Setup MemoryManager and ConversationalAgent
            memory_manager = MemoryManager(self.config, collection_name=collection_name)
            agent = ConversationalAgent(self.config, memory_manager, self.groq_client)
            
            turns_data = []
            
            for turn in conv["turns"]:
                user_msg = turn["user"]
                
                # 1. Chat using Agent (this retrieves relevant memories, applies forgetting, and generates reply)
                agent_response = agent.chat(
                    user_message=user_msg,
                    user_id=user_id,
                    conversation_id=conv_id,
                    memory_condition=condition_name
                )
                
                # 2. Extract and store memories from the user's turn
                history_turns = agent.get_history(conv_id)
                # Exclude last assistant response for context window during extraction
                history_except_last = history_turns[:-2] if len(history_turns) >= 2 else []
                context_turns = [f"{t['role']}: {t['content']}" for t in history_except_last]
                
                # Extract memories
                new_memories = []
                if condition_name != "no_memory":
                    new_memories = memory_manager.extractor.extract_memories(
                        conversation_turn=user_msg,
                        context="\n".join(context_turns),
                        user_id=user_id,
                        conversation_id=conv_id
                    )
                    
                    # Store memories enforcing the limit
                    max_memories = self.config.get("max_memories_per_user", 100)
                    for mem in new_memories:
                        current_count = memory_manager.store.get_memory_count(user_id)
                        if current_count >= max_memories:
                            all_mems = memory_manager.store.get_all_memories(user_id)
                            if all_mems:
                                memory_manager.store.delete_memory(all_mems[-1].memory_id)
                        memory_manager.store.add_memory(mem)
                
                # 3. Record turn statistics
                turn_record = {
                    "turn_id": turn["turn_id"],
                    "user_message": user_msg,
                    "response": agent_response.response,
                    "memories_used": [m.to_dict() for m in agent_response.memories_used],
                    "memory_context_injected": agent_response.memory_context_injected,
                    "latency_ms": agent_response.latency_ms,
                    "memories_stored": [m.to_dict() for m in new_memories],
                    "memory_count_after_turn": memory_manager.store.get_memory_count(user_id)
                }
                turns_data.append(turn_record)
                
                # 4. Simulate passage of time between turns (shift memories back by 4 days)
                if condition_name != "no_memory" and condition_name != "no_forgetting":
                    days_to_shift = 4
                    all_mems = memory_manager.store.get_all_memories(user_id)
                    for mem in all_mems:
                        created_dt = datetime.fromisoformat(mem.created_at)
                        if created_dt.tzinfo is None:
                            created_dt = created_dt.replace(tzinfo=timezone.utc)
                        shifted_dt = created_dt - timedelta(days=days_to_shift)
                        memory_manager.store.update_metadata(mem.memory_id, {
                            "created_at": shifted_dt.isoformat()
                        })

            # Evaluate metrics for this specific conversation
            logger.info(f"Computing metrics for conversation {conv_id} under condition {condition_name}")
            conv_metrics = compute_all_metrics(
                turns_data=turns_data,
                ground_truth_relevant_memories=ground_truth,
                groq_client=self.groq_client,
                config=self.config
            )
            
            # Cleanup collection after the run to save disk space, but keep metadata for analysis
            try:
                memory_manager.store.client.delete_collection(collection_name)
            except Exception as e:
                logger.warning(f"Failed to delete ChromaDB collection {collection_name}: {e}")

            # Store result
            condition_results[conv_id] = {
                "conversation_id": conv_id,
                "user_id": user_id,
                "category": conv["category"],
                "turns": turns_data,
                "metrics": conv_metrics
            }
            
            # Save checkpoints incrementally (fully resumable)
            temp_results = {
                "condition": condition_name,
                "run_id": self.run_id,
                "conversations": condition_results
            }
            with open(result_file, "w") as f:
                json.dump(temp_results, f, indent=2)
                
        # Aggregate metrics across all conversations for this condition
        logger.info(f"Aggregating metrics for condition '{condition_name}'...")
        all_conv_metrics = [c["metrics"] for c in condition_results.values()]
        
        aggregated_metrics = {}
        if all_conv_metrics:
            for key in all_conv_metrics[0].keys():
                values = [m[key] for m in all_conv_metrics]
                aggregated_metrics[key] = float(sum(values) / len(values))
        
        final_results = {
            "condition": condition_name,
            "run_id": self.run_id,
            "aggregated_metrics": aggregated_metrics,
            "conversations": condition_results
        }
        
        with open(result_file, "w") as f:
            json.dump(final_results, f, indent=2)
            
        logger.info(f"Saved completed results for condition '{condition_name}' to {result_file}")
        return final_results

    def run_all_conditions(self, force: bool = False) -> Dict[str, Any]:
        """
        Run all 4 conditions:
        1. "no_memory"
        2. "no_forgetting"
        3. "recency_decay"
        4. "hybrid"
        Save aggregated results to results/benchmark_results.json.
        """
        conditions = ["no_memory", "no_forgetting", "recency_decay", "hybrid"]
        
        benchmark_results = {}
        
        # Load existing benchmark results if present
        summary_file = self.results_dir / "benchmark_results.json"
        if summary_file.exists() and not force:
            try:
                with open(summary_file, "r") as f:
                    benchmark_results = json.load(f)
                logger.info(f"Loaded existing benchmark summary from {summary_file}")
            except Exception as e:
                logger.warning(f"Could not load benchmark summary: {e}. Re-running all.")

        for cond in conditions:
            if cond not in benchmark_results or force:
                logger.info(f"\n=== Running Memory Condition: {cond.upper()} ===")
                cond_res = self.run_condition(cond, force=force)
                benchmark_results[cond] = cond_res["aggregated_metrics"]
                
                # Checkpoint summary file
                with open(summary_file, "w") as f:
                    json.dump(benchmark_results, f, indent=2)
            else:
                logger.info(f"Condition '{cond}' is already in benchmark results summary. Skipping.")

        logger.info(f"\n=== Benchmark Complete! Summary saved to {summary_file} ===")
        print("\nBenchmark Summary Results:")
        print(json.dumps(benchmark_results, indent=2))
        
        return benchmark_results
