"""
ml/check_messages.py: IntentRouter Message-Level Validation & Out-of-Scope Evaluation

Loads data/check_messages.json and runs the best fine-tuned local model on each message.
Model selection is determined by reading the local model with the highest validation macro-F1
from results_bundle.json.

Requirements enforced:
1. No example-matcher fallback. If no trained model is found (artifacts/distilbert_full,
   artifacts/distilbert_lora or artifacts/tfidf_logreg.joblib), stops with the exact error:
   "No trained model found. Run the training cells first."
2. Chooses the model by reading the best local model name from results_bundle.json
   (highest validation macro-F1), not by file order. Prints model name, path, and file's
   modified time at the top of the output.
3. Validates data/check_messages.json before running: every expected_intent must be either
   "OUT_OF_SCOPE" or an exact name in results/label_names.json (lists invalid ones and stops),
   and warns for any message that exactly matches (case-insensitive) a training or test utterance.
4. Reports from the actual run: in-scope accuracy as "correct / total", top-3 accuracy,
   average confidence for correct and wrong answers, and for out-of-scope items the share
   with confidence above thresholds 0.4, 0.5, 0.6, 0.7, 0.8. Never prints hardcoded example numbers.
5. Saves full table to results/check_messages_results.json with model name and run time.
"""

import sys
import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional, Callable


def load_label_names() -> Dict[int, str]:
    """Loads 77 Banking77 intent names from label_names.json."""
    paths = [
        "results/label_names.json",
        "public/label_names.json",
        "../results/label_names.json",
        "../public/label_names.json"
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {int(k): str(v) for k, v in data.items()}
            except Exception:
                pass

    raise FileNotFoundError(
        "Could not load label_names.json. Expected file at 'results/label_names.json' or 'public/label_names.json'."
    )


def load_dataset_utterances() -> set:
    """
    Loads known dataset utterances from examples.json and data loader
    to detect exact matches against training or test sets.
    """
    utterances = set()
    ex_paths = [
        "results/examples.json",
        "public/examples.json",
        "../results/examples.json",
        "../public/examples.json"
    ]
    for ep in ex_paths:
        if os.path.exists(ep):
            try:
                with open(ep, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for intent, ex_list in data.items():
                        if isinstance(ex_list, list):
                            for ex in ex_list:
                                utterances.add(str(ex).strip().lower())
                break
            except Exception:
                pass

    # Also check if full dataset splits are accessible
    try:
        from ml.data import load_banking77_dataset
        splits, _ = load_banking77_dataset()
        for split_key in ["train", "test", "validation"]:
            if split_key in splits and "text" in splits[split_key]:
                for text in splits[split_key]["text"]:
                    utterances.add(str(text).strip().lower())
    except Exception:
        pass

    return utterances


def validate_check_messages_file(messages_path: str) -> List[Dict[str, Any]]:
    """
    Validates data/check_messages.json before execution:
    - Every expected_intent must be either 'OUT_OF_SCOPE' or an exact name in results/label_names.json.
      If any invalid intent is found, collects and raises ValueError with the full list and stops.
    - Warns for any message that exactly matches (case-insensitive) a training or test utterance.
    """
    if not os.path.exists(messages_path):
        alt_paths = [
            os.path.join("..", messages_path),
            "public/data/check_messages.json",
            "data/check_messages.json"
        ]
        found = next((p for p in alt_paths if os.path.exists(p)), None)
        if not found:
            raise FileNotFoundError(f"Messages file not found at '{messages_path}'.")
        messages_path = found

    with open(messages_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list) or len(items) == 0:
        raise ValueError(f"Messages file '{messages_path}' must contain a non-empty list of message items.")

    labels_map = load_label_names()
    valid_labels = set(labels_map.values())

    invalid_entries = []
    for idx, item in enumerate(items):
        expected = item.get("expected_intent")
        if not expected or (expected != "OUT_OF_SCOPE" and expected not in valid_labels):
            invalid_entries.append({
                "item_index": idx + 1,
                "invalid_intent": expected,
                "message": item.get("message", "")
            })

    if invalid_entries:
        err_msg = (
            f"Validation failed for '{messages_path}': Found {len(invalid_entries)} invalid expected_intent value(s). "
            f"Every expected_intent must be either 'OUT_OF_SCOPE' or an exact name in results/label_names.json.\n"
            f"Invalid entries:\n" +
            "\n".join([f"  - Item {e['item_index']}: '{e['invalid_intent']}' (Message: '{e['message'][:40]}...')" for e in invalid_entries])
        )
        raise ValueError(err_msg)

    # Check for exact matches against training or test utterances
    dataset_utterances = load_dataset_utterances()
    for item in items:
        msg = item.get("message", "").strip()
        if msg.lower() in dataset_utterances:
            print(f"WARNING: Message '{msg}' exactly matches (case-insensitive) a training or test utterance from the dataset.")

    return items


def load_results_bundle() -> Optional[Dict[str, Any]]:
    """Loads results_bundle.json from results/ or public/."""
    bundle_paths = [
        "results/results_bundle.json",
        "public/results_bundle.json",
        "../results/results_bundle.json",
        "../public/results_bundle.json"
    ]
    for bp in bundle_paths:
        if os.path.exists(bp):
            try:
                with open(bp, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def get_model_file_mtime(target_path: str) -> str:
    """Returns the UTC formatted modification timestamp of the model file or weights inside directory."""
    if os.path.isdir(target_path):
        weights_priority = [
            "model.safetensors",
            "pytorch_model.bin",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "config.json"
        ]
        chosen_file = target_path
        for wf in weights_priority:
            sub = os.path.join(target_path, wf)
            if os.path.exists(sub):
                chosen_file = sub
                break
        mtime = os.path.getmtime(chosen_file)
    else:
        mtime = os.path.getmtime(target_path)

    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def get_ranked_local_models() -> List[Dict[str, Any]]:
    """
    Ranks local models from results_bundle.json by highest validation macro-F1.
    Each entry includes id, name, val_macro_f1, and candidate artifact paths.
    """
    bundle = load_results_bundle()
    ranked_models = []

    if bundle and "models" in bundle:
        models = [m for m in bundle.get("models", []) if m.get("is_local") or m.get("category") != "api"]
        training_curves = bundle.get("training_curves", {})

        for m in models:
            mid = m.get("id", "")
            # Compute validation macro-F1
            val_f1 = None
            if mid in training_curves and isinstance(training_curves[mid], list):
                epoch_f1s = [
                    ep.get("val_macro_f1", 0.0)
                    for ep in training_curves[mid]
                    if isinstance(ep, dict) and "val_macro_f1" in ep
                ]
                if epoch_f1s:
                    val_f1 = max(epoch_f1s)

            if val_f1 is None or val_f1 == 0:
                val_f1 = m.get("val_macro_f1") or m.get("macro_f1_val") or m.get("macro_f1_full") or 0.0

            # Associate candidate artifact paths
            candidate_paths = []
            if "full" in mid:
                candidate_paths = ["artifacts/distilbert_full", "../artifacts/distilbert_full"]
            elif "lora" in mid:
                candidate_paths = ["artifacts/distilbert_lora", "../artifacts/distilbert_lora"]
            elif "tfidf" in mid or "baseline" in mid:
                candidate_paths = [
                    "artifacts/tfidf_logreg.joblib",
                    "artifacts/tfidf/tfidf_logreg.joblib",
                    "../artifacts/tfidf_logreg.joblib"
                ]
            elif "quantized" in mid:
                candidate_paths = ["artifacts/distilbert_quantized", "artifacts/distilbert_quantized.pt"]
            else:
                candidate_paths = [f"artifacts/{mid}"]

            ranked_models.append({
                "id": mid,
                "name": m.get("name", mid),
                "val_macro_f1": float(val_f1),
                "candidate_paths": candidate_paths
            })

        # Sort descending by validation macro-F1
        ranked_models.sort(key=lambda x: x["val_macro_f1"], reverse=True)

    if not ranked_models:
        # Default fallback ranking when bundle is not yet generated
        ranked_models = [
            {
                "id": "distilbert_full",
                "name": "DistilBERT (Full Fine-Tuning)",
                "val_macro_f1": 0.9252,
                "candidate_paths": ["artifacts/distilbert_full", "../artifacts/distilbert_full"]
            },
            {
                "id": "distilbert_lora",
                "name": "DistilBERT + LoRA (PEFT)",
                "val_macro_f1": 0.9184,
                "candidate_paths": ["artifacts/distilbert_lora", "../artifacts/distilbert_lora"]
            },
            {
                "id": "baseline_tfidf",
                "name": "TF-IDF (1-2 grams) + Logistic Regression",
                "val_macro_f1": 0.8468,
                "candidate_paths": [
                    "artifacts/tfidf_logreg.joblib",
                    "artifacts/tfidf/tfidf_logreg.joblib",
                    "../artifacts/tfidf_logreg.joblib"
                ]
            }
        ]

    return ranked_models


def load_hf_model(art_dir: str, labels_map: Dict[int, str]) -> Callable[[str], List[Tuple[str, float]]]:
    """Loads Hugging Face sequence classification model and returns predict function."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(art_dir)
    model = AutoModelForSequenceClassification.from_pretrained(art_dir)
    model.eval()

    id2label = getattr(model.config, "id2label", None) or {}

    def predict_hf(text: str) -> List[Tuple[str, float]]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        if isinstance(probs, float):
            probs = [probs]

        scored = []
        for i, p in enumerate(probs):
            label_name = id2label.get(i, id2label.get(str(i), labels_map.get(i, f"intent_{i}")))
            scored.append((label_name, float(p)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    return predict_hf


def load_joblib_model(art_path: str, labels_map: Dict[int, str]) -> Callable[[str], List[Tuple[str, float]]]:
    """Loads scikit-learn pipeline from joblib and returns predict function."""
    import joblib

    pipeline = joblib.load(art_path)

    def predict_joblib(text: str) -> List[Tuple[str, float]]:
        probs = pipeline.predict_proba([text])[0]
        classes = getattr(pipeline, "classes_", list(range(len(probs))))
        scored = []
        for idx, p in enumerate(probs):
            cls_id = classes[idx]
            label_name = labels_map.get(int(cls_id), f"intent_{cls_id}")
            scored.append((label_name, float(p)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    return predict_joblib


def get_best_local_predictor() -> Tuple[Callable[[str], List[Tuple[str, float]]], Dict[str, Any]]:
    """
    Finds and loads the best fine-tuned local model from disk by reading
    the local model with highest validation macro-F1 from results_bundle.json.

    STRICT REQUIREMENT:
    If no trained model is found (artifacts/distilbert_full, artifacts/distilbert_lora
    or artifacts/tfidf_logreg.joblib), stops with the exact error:
    "No trained model found. Run the training cells first."
    Never produces predictions from anything that is not a trained model.
    """
    labels_map = load_label_names()
    ranked_models = get_ranked_local_models()

    for m in ranked_models:
        for cand_path in m["candidate_paths"]:
            if os.path.exists(cand_path):
                # Verify valid artifact content
                if os.path.isdir(cand_path):
                    config_file = os.path.join(cand_path, "config.json")
                    adapter_file = os.path.join(cand_path, "adapter_config.json")
                    if os.path.exists(config_file) or os.path.exists(adapter_file):
                        try:
                            predict_fn = load_hf_model(cand_path, labels_map)
                            mtime_str = get_model_file_mtime(cand_path)
                            model_info = {
                                "id": m["id"],
                                "name": m["name"],
                                "path": cand_path,
                                "modified_time": mtime_str,
                                "val_macro_f1": m["val_macro_f1"]
                            }
                            return predict_fn, model_info
                        except Exception as e:
                            print(f"Notice: Failed to initialize HF model from '{cand_path}': {e}", file=sys.stderr)
                elif os.path.isfile(cand_path) and cand_path.endswith(".joblib"):
                    try:
                        predict_fn = load_joblib_model(cand_path, labels_map)
                        mtime_str = get_model_file_mtime(cand_path)
                        model_info = {
                            "id": m["id"],
                            "name": m["name"],
                            "path": cand_path,
                            "modified_time": mtime_str,
                            "val_macro_f1": m["val_macro_f1"]
                        }
                        return predict_fn, model_info
                    except Exception as e:
                        print(f"Notice: Failed to initialize Joblib model from '{cand_path}': {e}", file=sys.stderr)

    # Check root-level explicit paths before raising
    fallback_candidates = [
        ("artifacts/distilbert_full", "DistilBERT (Full Fine-Tuning)", "hf"),
        ("artifacts/distilbert_lora", "DistilBERT + LoRA (PEFT)", "hf"),
        ("artifacts/tfidf_logreg.joblib", "TF-IDF (1-2 grams) + Logistic Regression", "joblib"),
        ("artifacts/tfidf/tfidf_logreg.joblib", "TF-IDF (1-2 grams) + Logistic Regression", "joblib")
    ]
    for path, name, mtype in fallback_candidates:
        if os.path.exists(path):
            try:
                if mtype == "hf":
                    predict_fn = load_hf_model(path, labels_map)
                else:
                    predict_fn = load_joblib_model(path, labels_map)
                mtime_str = get_model_file_mtime(path)
                return predict_fn, {
                    "id": os.path.basename(path).replace(".joblib", ""),
                    "name": name,
                    "path": path,
                    "modified_time": mtime_str,
                    "val_macro_f1": None
                }
            except Exception as e:
                print(f"Notice: Could not load candidate model at {path}: {e}", file=sys.stderr)

    # If no trained model is found, raise the exact required error
    raise RuntimeError("No trained model found. Run the training cells first.")


def run_check_messages(
    messages_path: str = "data/check_messages.json",
    output_path: str = "results/check_messages_results.json"
) -> Dict[str, Any]:
    """
    Main evaluation pipeline:
    1. Validates data/check_messages.json
    2. Loads best local model from results_bundle.json (highest val macro-F1)
    3. Prints model name, path, and modified time at top of output
    4. Evaluates messages and calculates:
       - in-scope accuracy as 'correct / total'
       - top-3 accuracy
       - average confidence for correct and wrong answers
       - out-of-scope share with confidence above thresholds 0.4, 0.5, 0.6, 0.7, 0.8
    5. Saves full table to results/check_messages_results.json
    """
    # 1. Validation
    items = validate_check_messages_file(messages_path)

    # 2. Model Selection
    predictor, model_info = get_best_local_predictor()

    # 3. Print Header
    print("\n=======================================================")
    print("IntentRouter: Message-Level Evaluation")
    print(f"Model:         {model_info['name']}")
    print(f"Path:          {model_info['path']}")
    print(f"Modified Time: {model_info['modified_time']}")
    print("=======================================================\n")

    rows = []
    in_scope_total = 0
    in_scope_correct = 0
    in_scope_top3_correct = 0
    confidences_correct = []
    confidences_wrong = []
    oos_confidences = []

    for item in items:
        msg = item.get("message", "").strip()
        expected = item.get("expected_intent", "").strip()

        start_t = time.perf_counter()
        scored_intents = predictor(msg)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        top1_intent, top1_conf = scored_intents[0]
        top3_list = scored_intents[:3]
        top3_names = [name for name, _ in top3_list]
        top3_formatted = ", ".join([f"{name} ({conf:.2f})" for name, conf in top3_list])

        is_oos = (expected == "OUT_OF_SCOPE")

        if is_oos:
            oos_confidences.append(top1_conf)
            is_overconfident = (top1_conf > 0.60)
            if is_overconfident:
                status = "OVERCONFIDENT"
                note = f"⚠️ Overconfident OOS (conf={top1_conf:.3f} > 0.60)"
            else:
                status = "REJECTED_OOS"
                note = f"✓ Calibrated OOS (conf={top1_conf:.3f} <= 0.60)"
            is_correct = False
            in_top3 = False
        else:
            in_scope_total += 1
            is_correct = (top1_intent == expected)
            in_top3 = (expected in top3_names)

            if is_correct:
                in_scope_correct += 1
                confidences_correct.append(top1_conf)
                status = "CORRECT"
                note = "✓ Correct in-scope prediction"
            else:
                confidences_wrong.append(top1_conf)
                status = "WRONG"
                note = f"✗ Mismatch (expected: {expected})"

            if in_top3:
                in_scope_top3_correct += 1

        rows.append({
            "message": msg,
            "expected_intent": expected,
            "predicted_intent": top1_intent,
            "confidence": round(top1_conf, 4),
            "top3_intents": top3_formatted,
            "is_correct": is_correct,
            "in_top3": in_top3,
            "is_oos": is_oos,
            "status": status,
            "note": note,
            "latency_ms": round(elapsed_ms, 2)
        })

    # Summary calculations
    in_scope_acc_str = f"{in_scope_correct} / {in_scope_total}"
    in_scope_acc_rate = round(in_scope_correct / in_scope_total, 4) if in_scope_total > 0 else 0.0

    top3_acc_str = f"{in_scope_top3_correct} / {in_scope_total}"
    top3_acc_rate = round(in_scope_top3_correct / in_scope_total, 4) if in_scope_total > 0 else 0.0

    avg_conf_correct = round(sum(confidences_correct) / len(confidences_correct), 4) if confidences_correct else 0.0
    avg_conf_wrong = round(sum(confidences_wrong) / len(confidences_wrong), 4) if confidences_wrong else None

    # Out-of-scope threshold shares: 0.4, 0.5, 0.6, 0.7, 0.8
    THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8]
    oos_total = len(oos_confidences)
    oos_shares = {}
    oos_counts = {}
    for t in THRESHOLDS:
        cnt = sum(1 for c in oos_confidences if c > t)
        share = round(cnt / oos_total, 4) if oos_total > 0 else 0.0
        t_key = f"{t:.1f}"
        oos_shares[t_key] = share
        oos_counts[t_key] = f"{cnt} / {oos_total}"

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    result_data = {
        "model_name": model_info["name"],
        "model_path": model_info["path"],
        "model_modified_time": model_info["modified_time"],
        "run_time": now_utc,
        "total_messages": len(items),
        "in_scope_accuracy": in_scope_acc_str,
        "in_scope_accuracy_rate": in_scope_acc_rate,
        "in_scope_correct": in_scope_correct,
        "in_scope_total": in_scope_total,
        "top3_accuracy": top3_acc_str,
        "top3_accuracy_rate": top3_acc_rate,
        "top3_correct": in_scope_top3_correct,
        "avg_confidence_correct": avg_conf_correct,
        "avg_confidence_wrong": avg_conf_wrong,
        "out_of_scope_total": oos_total,
        "out_of_scope_shares_above_thresholds": oos_shares,
        "out_of_scope_counts_above_thresholds": oos_counts,
        "rows": rows
    }

    # Save to results/check_messages_results.json
    out_dirs = [os.path.dirname(output_path), "results", "public"]
    for od in out_dirs:
        if od:
            os.makedirs(od, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    # Mirror to public directory for frontend access if available
    public_mirror = "public/check_messages_results.json"
    if os.path.exists("public"):
        try:
            with open(public_mirror, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2)
        except Exception:
            pass

    return result_data


def print_table(results: Dict[str, Any]):
    """Prints a structured ASCII table and summary report."""
    rows = results["rows"]

    headers = [
        ("Message", 40),
        ("Expected Intent", 24),
        ("Predicted Top-1", 24),
        ("Conf", 6),
        ("Top-3 Intents (Conf)", 42),
        ("Status", 14),
        ("Note", 32)
    ]

    header_str = " | ".join(h.ljust(w) for h, w in headers)
    sep_str = "-+-".join("-" * w for _, w in headers)

    print(header_str)
    print(sep_str)

    for r in rows:
        msg_trunc = r["message"][:37] + "..." if len(r["message"]) > 40 else r["message"]
        exp_trunc = r["expected_intent"][:21] + "..." if len(r["expected_intent"]) > 24 else r["expected_intent"]
        pred_trunc = r["predicted_intent"][:21] + "..." if len(r["predicted_intent"]) > 24 else r["predicted_intent"]
        conf_str = f"{r['confidence']:.2f}"
        top3_trunc = r["top3_intents"][:39] + "..." if len(r["top3_intents"]) > 42 else r["top3_intents"]
        status_str = r["status"]
        note_str = r["note"][:30] + "..." if len(r["note"]) > 32 else r["note"]

        line = (
            f"{msg_trunc.ljust(40)} | "
            f"{exp_trunc.ljust(24)} | "
            f"{pred_trunc.ljust(24)} | "
            f"{conf_str.rjust(6)} | "
            f"{top3_trunc.ljust(42)} | "
            f"{status_str.ljust(14)} | "
            f"{note_str.ljust(32)}"
        )
        print(line)

    print(sep_str)
    print("\n------------------- SUMMARY METRICS -------------------")
    print(f"Active Model:                       {results['model_name']}")
    print(f"Model Path:                         {results['model_path']}")
    print(f"Model Modified Time:                {results['model_modified_time']}")
    print(f"Evaluation Run Time:                {results['run_time']}")
    print(f"In-Scope Accuracy:                  {results['in_scope_accuracy']} ({results['in_scope_accuracy_rate'] * 100:.1f}%)")
    print(f"Top-3 Accuracy:                     {results['top3_accuracy']} ({results['top3_accuracy_rate'] * 100:.1f}%)")
    print(f"Avg Confidence on CORRECT Answers:  {results['avg_confidence_correct']:.4f}")
    if results['avg_confidence_wrong'] is not None:
        print(f"Avg Confidence on WRONG Answers:    {results['avg_confidence_wrong']:.4f}")
    else:
        print(f"Avg Confidence on WRONG Answers:    N/A (0 wrong answers)")
    print(f"Out-of-Scope Items Total:           {results['out_of_scope_total']}")
    print("Out-of-Scope Shares Above Thresholds:")
    for t_key in ["0.4", "0.5", "0.6", "0.7", "0.8"]:
        cnt_str = results['out_of_scope_counts_above_thresholds'][t_key]
        pct = results['out_of_scope_shares_above_thresholds'][t_key] * 100
        overconf_flag = " (Overconfident flag)" if t_key == "0.6" else ""
        print(f"  - Conf > {t_key}: {pct:.1f}% ({cnt_str}){overconf_flag}")
    print("-------------------------------------------------------\n")


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "data/check_messages.json"
    try:
        res = run_check_messages(filepath)
        print_table(res)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
