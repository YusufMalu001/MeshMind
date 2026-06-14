import logging
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np

from groq import Groq

logger = logging.getLogger("MeshMind.Metrics")


def is_match(retrieved_str: str, relevant_str: str) -> bool:
    """
    Check if a retrieved memory matches a relevant ground truth memory.
    Uses word intersection and substring containment.
    """
    r_clean = retrieved_str.lower().strip()
    rel_clean = relevant_str.lower().strip()
    
    # Substring matches
    if rel_clean in r_clean or r_clean in rel_clean:
        return True
        
    # Word intersection for words of length > 3
    r_words = set(w.replace(".", "").replace(",", "").replace("!", "") for w in r_clean.split() if len(w) > 3)
    rel_words = set(w.replace(".", "").replace(",", "").replace("!", "") for w in rel_clean.split() if len(w) > 3)
    
    if not r_words or not rel_words:
        return False
        
    # If they share 2 or more significant words (e.g. ["priya", "delhi", "sister"]), it's a match
    intersection_size = len(r_words.intersection(rel_words))
    if intersection_size >= 2:
        return True
        
    return False


def memory_precision(retrieved: List[str], relevant: List[str]) -> float:
    """
    |retrieved ∩ relevant| / |retrieved|
    What fraction of retrieved memories were relevant?
    """
    if not retrieved:
        return 0.0
    matched = 0
    for r in retrieved:
        if any(is_match(r, rel) for rel in relevant):
            matched += 1
    return matched / len(retrieved)


def memory_recall(retrieved: List[str], relevant: List[str]) -> float:
    """
    |retrieved ∩ relevant| / |relevant|
    What fraction of relevant memories were retrieved?
    """
    if not relevant:
        return 1.0
    matched = 0
    for rel in relevant:
        if any(is_match(r, rel) for r in retrieved):
            matched += 1
    return matched / len(relevant)


def evaluate_hallucination(
    response: str, 
    memory_context: str, 
    groq_client: Groq, 
    model: str
) -> bool:
    """
    Use LLM Judge to check if the response contains claims about the user 
    unsupported by the memory context. Returns True if hallucinated, False otherwise.
    """
    system_prompt = (
        "You are an evaluation judge. Determine if the conversational assistant's response "
        "contains any claims, facts, or assumptions about the user that are NOT supported "
        "by the provided memory context. Return ONLY a JSON object with this structure:\n"
        "{\n"
        "  \"hallucinated\": true/false,\n"
        "  \"explanation\": \"your explanation here\"\n"
        "}\n\n"
        "Rules:\n"
        "- If the memory context is empty, any specific claims about the user (e.g., sister's name, favorite food, pets) are unsupported and should be marked as hallucinated (true).\n"
        "- Do not mark polite conversational replies or general questions as hallucinated. Only mark specific claims/facts about the user."
    )
    
    user_prompt = (
        f"Memory Context:\n{memory_context if memory_context else '[Empty]'}\n\n"
        f"Response:\n{response}\n\n"
        "Evaluate:"
    )
    
    retries = [10, 30]
    for i, wait_time in enumerate(retries + [0]):
        try:
            res = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = res.choices[0].message.content or ""
            parsed = json.loads(content)
            # Pacing sleep to respect rate limits
            time.sleep(1.0)
            return bool(parsed.get("hallucinated", False))
        except Exception as e:
            if i < len(retries):
                logger.warning(f"Hallucination judge failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Hallucination judge failed completely: {e}")
                return False
    return False


def evaluate_personalization(
    response: str, 
    memory_context: str, 
    groq_client: Groq, 
    model: str
) -> int:
    """
    Use LLM Judge to rate personalization on a scale of 1-5.
    1=generic, 3=somewhat personal, 5=highly personal.
    """
    system_prompt = (
        "You are an evaluation judge. Rate the personalization of the assistant's response "
        "on a scale of 1 to 5. Check if it naturally and correctly references details about the user "
        "provided in the memory context. Return ONLY a JSON object with this structure:\n"
        "{\n"
        "  \"score\": 1-5,\n"
        "  \"explanation\": \"your explanation here\"\n"
        "}\n\n"
        "Scoring guide:\n"
        "- 1: Generic response, does not use or reference any user-specific details from the memory context.\n"
        "- 3: Somewhat personal, references general topics or a single user-specific detail.\n"
        "- 5: Highly personal, naturally weaves multiple relevant user details from the memory context into the reply."
    )
    
    user_prompt = (
        f"Memory Context:\n{memory_context if memory_context else '[Empty]'}\n\n"
        f"Response:\n{response}\n\n"
        "Rate personalization:"
    )
    
    retries = [10, 30]
    for i, wait_time in enumerate(retries + [0]):
        try:
            res = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = res.choices[0].message.content or ""
            parsed = json.loads(content)
            score = int(parsed.get("score", 1))
            # Pacing sleep to respect rate limits
            time.sleep(1.0)
            return max(1, min(5, score))
        except Exception as e:
            if i < len(retries):
                logger.warning(f"Personalization judge failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Personalization judge failed completely: {e}")
                return 1
    return 1


def evaluate_coherence(
    response: str, 
    conversation_context: str, 
    groq_client: Groq, 
    model: str
) -> int:
    """
    Use LLM Judge to rate response coherence on a scale of 1-5.
    """
    system_prompt = (
        "You are an evaluation judge. Rate the coherence and naturalness of the assistant's response "
        "given the conversation context on a scale of 1 to 5. Return ONLY a JSON object with this structure:\n"
        "{\n"
        "  \"score\": 1-5,\n"
        "  \"explanation\": \"your explanation here\"\n"
        "}\n\n"
        "Scoring guide:\n"
        "- 1: Completely irrelevant or incoherent reply.\n"
        "- 3: Coherent but slightly awkward or repetitive.\n"
        "- 5: Highly coherent, fluent, and naturally engaging conversational reply."
    )
    
    user_prompt = (
        f"Conversation Context:\n{conversation_context}\n\n"
        f"Response:\n{response}\n\n"
        "Rate coherence:"
    )
    
    retries = [10, 30]
    for i, wait_time in enumerate(retries + [0]):
        try:
            res = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = res.choices[0].message.content or ""
            parsed = json.loads(content)
            score = int(parsed.get("score", 5))
            # Pacing sleep to respect rate limits
            time.sleep(1.0)
            return max(1, min(5, score))
        except Exception as e:
            if i < len(retries):
                logger.warning(f"Coherence judge failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Coherence judge failed completely: {e}")
                return 5
    return 5


def compute_all_metrics(
    turns_data: List[Dict[str, Any]], 
    ground_truth_relevant_memories: List[str],
    groq_client: Groq,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute complete metrics dict for one condition.
    - Precision
    - Recall
    - Hallucination Rate
    - Personalization Score
    - Coherence Score
    - Avg Latency
    """
    judge_model = config.get("judge_model", "llama-3.3-70b-versatile")
    
    retrieved_contents = []
    responses = []
    memory_contexts = []
    latencies = []
    coherence_contexts = []
    
    # Process each turn
    for turn in turns_data:
        # Collect data for precision & recall
        # turn["memories_used"] is list of memory dicts/objects
        mems_used = turn.get("memories_used", [])
        mems_str = [m.get("content", m) if isinstance(m, dict) else getattr(m, "content", str(m)) for m in mems_used]
        retrieved_contents.extend(mems_str)
        
        # Collect data for LLM judge
        responses.append(turn.get("response", ""))
        memory_contexts.append(turn.get("memory_context_injected", ""))
        latencies.append(turn.get("latency_ms", 0.0))
        
        # Build context for coherence evaluation (last few turns)
        coherence_contexts.append(turn.get("user_message", ""))

    # 1. Precision & Recall
    prec = memory_precision(retrieved_contents, ground_truth_relevant_memories)
    rec = memory_recall(retrieved_contents, ground_truth_relevant_memories)
    
    # 2. Hallucination rate
    hallucination_results = []
    for resp, mem_ctx in zip(responses, memory_contexts):
        h = evaluate_hallucination(resp, mem_ctx, groq_client, judge_model)
        hallucination_results.append(h)
    hall_rate = float(np.mean(hallucination_results)) if hallucination_results else 0.0
    
    # 3. Personalization score
    personalization_results = []
    for resp, mem_ctx in zip(responses, memory_contexts):
        p = evaluate_personalization(resp, mem_ctx, groq_client, judge_model)
        personalization_results.append(p)
    pers_score = float(np.mean(personalization_results)) if personalization_results else 1.0
    
    # 4. Coherence score
    coherence_results = []
    for idx, resp in enumerate(responses):
        # build running dialogue context up to that point
        conv_ctx = "\n".join(coherence_contexts[:idx+1])
        c = evaluate_coherence(resp, conv_ctx, groq_client, judge_model)
        coherence_results.append(c)
    coh_score = float(np.mean(coherence_results)) if coherence_results else 5.0
    
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    
    return {
        "precision": prec,
        "recall": rec,
        "hallucination_rate": hall_rate,
        "personalization_score": pers_score,
        "coherence_score": coh_score,
        "avg_latency_ms": avg_latency
    }
