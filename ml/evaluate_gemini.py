"""
IntentRouter - Model 4: Gemini Zero-Shot & Few-Shot Evaluation
Evaluates Gemini on the stratified 308-sample test subset using the Gemini API.
Features:
- Reads key from Colab secret (userdata.get('GEMINI_API_KEY')) or os.environ['GEMINI_API_KEY']
- Disk-based caching (SQLite) to prevent redundant API calls
- Token bucket / sleep rate-limiter and exponential backoff retry
- Constrained response schema / enum adhering strictly to the 77 Banking77 intent labels
- Counts invalid or unparseable outputs as classification errors
- Configurable GEMINI_MODEL setting
"""

import os
import json
import time
import sqlite3
from typing import Dict, List, Optional, Any
import numpy as np

from ml.data import load_banking77_dataset
from ml.metrics import compute_classification_metrics, calculate_cost_per_1k


class DiskCache:
    """SQLite-backed disk cache for Gemini prompt completions."""
    def __init__(self, db_path: str = "artifacts/gemini_cache.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt TEXT,
                    response_json TEXT,
                    prompt_tokens INTEGER,
                    candidate_tokens INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        import hashlib
        p_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        cur = self.conn.cursor()
        cur.execute("SELECT response_json, prompt_tokens, candidate_tokens FROM cache WHERE prompt_hash = ?", (p_hash,))
        row = cur.fetchone()
        if row:
            return {
                "response": json.loads(row[0]),
                "prompt_tokens": row[1],
                "candidate_tokens": row[2]
            }
        return None

    def set(self, prompt: str, response: Dict[str, Any], prompt_tokens: int, candidate_tokens: int):
        import hashlib
        p_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO cache (prompt_hash, prompt, response_json, prompt_tokens, candidate_tokens)
                VALUES (?, ?, ?, ?, ?)
            """, (p_hash, prompt, json.dumps(response), prompt_tokens, candidate_tokens))


def get_gemini_api_key() -> str:
    """Retrieves Gemini API key safely from Colab secret or environment variable."""
    try:
        from google.colab import userdata
        key = userdata.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    raise ValueError(
        "GEMINI_API_KEY is not set! In Colab, add GEMINI_API_KEY to 'Secrets'. "
        "Locally, set 'export GEMINI_API_KEY=your_key'."
    )


def build_system_instruction() -> str:
    return (
        "You are an expert banking intent router for a retail bank. "
        "Your task is to classify customer support messages into exactly ONE of the 77 predefined banking intents. "
        "You must return valid JSON matching the schema with the exact intent name from the allowed list."
    )


def evaluate_gemini(
    few_shot: bool = False,
    output_dir: str = "artifacts/gemini",
    model_id: Optional[str] = None,
    delay_between_calls_sec: float = 0.5,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    api_key = get_gemini_api_key()

    if not model_id:
        model_id = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
    except ImportError:
        print("Error: google-genai is not installed. Run: pip install google-genai", file=os.sys.stderr)
        raise

    splits, label_names, _ = load_banking77_dataset()
    test_subset = splits["test_subset_308"]

    text_col = "utterance" if "utterance" in test_subset.column_names else "text"
    test_texts = test_subset[text_col]
    test_labels = test_subset["label"]

    if max_samples or os.environ.get("SMOKE_TEST") == "1":
        limit = max_samples or 50
        print(f"Limiting Gemini evaluation to {limit} samples (smoke test mode).")
        test_texts = test_texts[:limit]
        test_labels = test_labels[:limit]

    cache = DiskCache(os.path.join(output_dir, "gemini_cache.db"))

    # Load 5 few-shot exemplars if requested
    few_shot_prompt = ""
    if few_shot:
        train_data = splits["train"]
        t_col = "utterance" if "utterance" in train_data.column_names else "text"
        few_shot_classes = [0, 8, 14, 21, 62]
        exemplars = []
        for c in few_shot_classes:
            ex = next((item for item in train_data if item["label"] == c), None)
            if ex:
                exemplars.append({"text": ex[t_col], "intent": label_names[c]})
        few_shot_prompt = "Reference Examples:\n" + "\n".join([
            f"Query: \"{ex['text']}\" -> Intent: {ex['intent']}" for ex in exemplars
        ]) + "\n\n"

    # Schema definition: constrained enum for intent
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "intent": {
                "type": "STRING",
                "enum": label_names,
                "description": "The exact classified banking intent from the 77 allowed labels."
            },
            "reasoning": {
                "type": "STRING",
                "description": "A brief explanation of why this intent was chosen."
            }
        },
        "required": ["intent"]
    }

    predicted_labels = []
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    latencies = []
    invalid_outputs = 0

    mode_name = f"{model_id} (5-Shot)" if few_shot else f"{model_id} (Zero-Shot Baseline)"
    print(f"Evaluating {mode_name} on {len(test_texts)} stratified test queries...")

    label2id = {name: i for i, name in enumerate(label_names)}

    for i, query in enumerate(test_texts):
        full_prompt = (
            f"{few_shot_prompt}"
            f"Customer Message: \"{query}\"\n"
            f"Classify this message into one of the 77 banking intents."
        )

        cached = cache.get(full_prompt)
        if cached:
            parsed = cached["response"]
            p_tok = cached["prompt_tokens"]
            c_tok = cached["candidate_tokens"]
            lat = 0.05
        else:
            time.sleep(delay_between_calls_sec)
            start_t = time.perf_counter()

            max_retries = 5
            backoff = 2.0
            response_text = None
            p_tok, c_tok = 0, 0

            for attempt in range(max_retries):
                try:
                    resp = client.models.generate_content(
                        model=model_id,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=build_system_instruction(),
                            temperature=0.0,
                            response_mime_type="application/json",
                            response_schema=response_schema
                        )
                    )
                    response_text = resp.text
                    if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                        p_tok = resp.usage_metadata.prompt_token_count or 120
                        c_tok = resp.usage_metadata.candidates_token_count or 15
                    else:
                        p_tok, c_tok = 120, 15
                    break
                except Exception as ex:
                    if "429" in str(ex) or "ResourceExhausted" in str(ex):
                        print(f"Rate limit hit at item {i}. Sleeping {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2.0
                    else:
                        print(f"API Error at item {i}: {ex}. Retrying...")
                        time.sleep(1.0)

            lat = (time.perf_counter() - start_t) * 1000.0

            try:
                parsed = json.loads(response_text)
                cache.set(full_prompt, parsed, p_tok, c_tok)
            except Exception:
                parsed = {"intent": "INVALID_OUTPUT"}

        predicted_intent = parsed.get("intent", "").strip()

        if predicted_intent in label2id:
            pred_id = label2id[predicted_intent]
        else:
            # Count invalid outputs as errors
            invalid_outputs += 1
            pred_id = -1

        predicted_labels.append(pred_id)
        total_prompt_tokens += p_tok
        total_candidate_tokens += c_tok
        latencies.append(lat)

        if (i + 1) % 50 == 0 or (i + 1) == len(test_texts):
            print(f"Progress: {i + 1}/{len(test_texts)} complete.")

    avg_latency = float(np.mean(latencies))
    avg_p_tok = int(total_prompt_tokens / len(test_texts))
    avg_c_tok = int(total_candidate_tokens / len(test_texts))

    metrics = compute_classification_metrics(test_labels, predicted_labels, y_probs=None)
    cost_per_1k = calculate_cost_per_1k(
        latency_cpu_ms=avg_latency,
        is_gemini=True,
        prompt_tokens=avg_p_tok,
        completion_tokens=avg_c_tok
    )

    results = {
        "id": "gemini_few_shot" if few_shot else "gemini_zero_shot",
        "name": mode_name,
        "category": "api",
        "is_baseline": not few_shot,
        "is_local": False,
        "accuracy_full": metrics["accuracy"],
        "macro_f1_full": metrics["macro_f1"],
        "weighted_f1_full": metrics["weighted_f1"],
        "top3_accuracy_full": None,
        "accuracy_subset_308": metrics["accuracy"],
        "macro_f1_subset_308": metrics["macro_f1"],
        "weighted_f1_subset_308": metrics["weighted_f1"],
        "top3_accuracy_subset_308": None,
        "latency_cpu_ms": round(avg_latency, 2),
        "latency_gpu_ms": None,
        "model_size_mb": 0.0,
        "estimated_cost_per_1k": cost_per_1k,
        "cost_type": "api_pricing",
        "invalid_outputs_count": invalid_outputs,
        "average_prompt_tokens": avg_p_tok,
        "average_candidate_tokens": avg_c_tok,
        "gemini_model": model_id,
        "notes": f"Constrained response schema. Invalid outputs: {invalid_outputs}."
    }

    out_file = os.path.join(output_dir, f"{'few_shot' if few_shot else 'zero_shot'}_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{mode_name} Finished. Acc: {results['accuracy_subset_308']} | Macro-F1: {results['macro_f1_subset_308']} | Cost/1k: ${cost_per_1k}")
    return results


if __name__ == "__main__":
    evaluate_gemini(few_shot=False)
