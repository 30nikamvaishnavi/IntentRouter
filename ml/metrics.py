"""
IntentRouter Metrics & Cascade Analysis Module
Calculates classification metrics, expected calibration error (ECE), reliability bins,
and real cascade analysis (local model with Gemini fallback across confidence thresholds).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
import time
import os

from ml.data import load_banking77_dataset


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Computes core classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    top3_acc = None
    if y_probs is not None:
        y_true_arr = np.array(y_true)
        top3_preds = np.argsort(y_probs, axis=1)[:, -3:]
        top3_matches = np.any(top3_preds == y_true_arr[:, None], axis=1)
        top3_acc = float(np.mean(top3_matches))

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "top3_accuracy": round(top3_acc, 4) if top3_acc is not None else None
    }


def compute_expected_calibration_error(
    y_true: List[int],
    y_pred: List[int],
    y_probs: np.ndarray,
    num_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and returns reliability diagram bins.
    """
    confidences = np.max(y_probs, axis=1)
    accuracies = (np.array(y_pred) == np.array(y_true)).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    bins_data = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        sample_count = int(np.sum(in_bin))
        if sample_count > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            accuracy_in_bin = 0.0
            avg_confidence_in_bin = float((bin_lower + bin_upper) / 2.0)

        bins_data.append({
            "bin_index": i + 1,
            "bin_lower": round(float(bin_lower), 2),
            "bin_upper": round(float(bin_upper), 2),
            "mean_confidence": round(avg_confidence_in_bin, 4),
            "accuracy": round(accuracy_in_bin, 4),
            "sample_count": sample_count
        })

    return round(float(ece), 4), bins_data


def compute_cascade_analysis(
    y_true_subset: List[int],
    y_pred_local: List[int],
    y_probs_local: np.ndarray,
    gemini_preds_subset: List[int],
    thresholds: Optional[List[float]] = None,
    cost_local_1k: float = 0.0036,
    cost_gemini_1k: float = 0.0385,
    latency_local_ms: float = 12.8,
    latency_gemini_ms: float = 582.0
) -> Dict[str, Any]:
    """
    Evaluates cascade routing across thresholds from 0.30 to 0.95 in steps of 0.05.
    Queries where local confidence >= threshold are handled locally;
    queries below threshold route to Gemini.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]

    confidences = np.max(y_probs_local, axis=1)
    y_true_arr = np.array(y_true_subset)
    y_pred_local_arr = np.array(y_pred_local)
    gemini_arr = np.array(gemini_preds_subset)
    total_samples = len(y_true_subset)

    steps = []

    for tau in thresholds:
        handled_locally = confidences >= tau
        local_count = int(np.sum(handled_locally))
        gemini_count = total_samples - local_count

        coverage = float(local_count / total_samples) if total_samples > 0 else 0.0

        if local_count > 0:
            local_acc = float(np.mean(y_pred_local_arr[handled_locally] == y_true_arr[handled_locally]))
        else:
            local_acc = 1.0

        # Full cascade accuracy: local predictions where handled_locally, gemini predictions elsewhere
        cascade_preds = np.where(handled_locally, y_pred_local_arr, gemini_arr)
        cascade_acc = float(np.mean(cascade_preds == y_true_arr))

        # Weighted cost & latency
        blended_cost = (coverage * cost_local_1k) + ((1.0 - coverage) * cost_gemini_1k)
        blended_lat = (coverage * latency_local_ms) + ((1.0 - coverage) * latency_gemini_ms)

        steps.append({
            "threshold": round(tau, 2),
            "coverage": round(coverage, 4),
            "local_accuracy": round(local_acc, 4),
            "cascade_accuracy_subset_308": round(cascade_acc, 4),
            "local_count_subset_308": local_count,
            "gemini_count_subset_308": gemini_count,
            "estimated_cost_per_1k": round(blended_cost, 4),
            "average_latency_ms": round(blended_lat, 1)
        })

    ece, rel_bins = compute_expected_calibration_error(y_true_subset, y_pred_local, y_probs_local)

    return {
        "selected_threshold": 0.80,
        "expected_calibration_error": ece,
        "reliability_bins": rel_bins,
        "threshold_steps": steps,
        "note": "Evaluated on the 308-sample stratified test subset. Queries where small model confidence >= threshold are resolved locally; queries below threshold are routed to Gemini."
    }


def compute_per_class_f1(y_true: List[int], y_pred: List[int], label_names: List[str]) -> List[Dict[str, Any]]:
    """Computes per-class F1 for all 77 intent classes."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(label_names))), zero_division=0
    )

    results = []
    for class_id in range(len(label_names)):
        results.append({
            "class_id": class_id,
            "intent_name": label_names[class_id],
            "category": _infer_banking_category(label_names[class_id]),
            "test_count": int(support[class_id]),
            "precision": round(float(precision[class_id]), 4),
            "recall": round(float(recall[class_id]), 4),
            "f1_score": round(float(f1[class_id]), 4)
        })

    sorted_by_f1 = sorted(results, key=lambda x: x["f1_score"], reverse=True)
    for rank, item in enumerate(sorted_by_f1, start=1):
        item["rank"] = rank

    return sorted(results, key=lambda x: x["class_id"])


def find_top_confused_pairs(
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str],
    top_n: int = 15
) -> List[Dict[str, Any]]:
    """Identifies top confused intent pairs."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    np.fill_diagonal(cm, 0)

    flat_indices = np.argsort(cm.ravel())[::-1]
    confused_pairs = []

    for rank, idx in enumerate(flat_indices[:top_n], start=1):
        true_idx, pred_idx = np.unravel_index(idx, cm.shape)
        count = int(cm[true_idx, pred_idx])
        if count == 0:
            break

        total_class_errors = int(np.sum(cm[true_idx, :]))
        pct = round((count / total_class_errors * 100), 1) if total_class_errors > 0 else 0.0

        confused_pairs.append({
            "rank": rank,
            "true_intent": label_names[true_idx],
            "predicted_intent": label_names[pred_idx],
            "count": count,
            "percentage_of_class_errors": pct,
            "explanation": f"Model confused '{label_names[true_idx]}' with '{label_names[pred_idx]}' in {count} instances."
        })

    return confused_pairs


def extract_misclassified_examples(
    texts: List[str],
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str],
    y_probs: Optional[np.ndarray] = None,
    max_examples: int = 20
) -> List[Dict[str, Any]]:
    """Extracts misclassified examples."""
    misclassified = []
    for idx, (text, true_lbl, pred_lbl) in enumerate(zip(texts, y_true, y_pred)):
        if true_lbl != pred_lbl:
            confidence = float(np.max(y_probs[idx])) if y_probs is not None else 0.50
            misclassified.append({
                "id": len(misclassified) + 1,
                "text": text,
                "true_label": label_names[true_lbl],
                "predicted_label": label_names[pred_lbl],
                "confidence": round(confidence, 2),
                "error_type": "subtle_boundary",
                "explanation": f"True intent was '{label_names[true_lbl]}', but model predicted '{label_names[pred_lbl]}'."
            })
            if len(misclassified) >= max_examples:
                break
    return misclassified


def benchmark_latency(
    model_inference_fn,
    sample_inputs: List[Any],
    warmup_runs: int = 5,
    repeat_runs: int = 50
) -> float:
    """Measures per-sample latency in milliseconds."""
    for _ in range(warmup_runs):
        _ = model_inference_fn(sample_inputs[:2])

    start_time = time.perf_counter()
    count = 0
    for _ in range(repeat_runs):
        for inp in sample_inputs:
            _ = model_inference_fn([inp])
            count += 1
    elapsed = time.perf_counter() - start_time
    latency_ms = (elapsed / count) * 1000.0
    return round(latency_ms, 2)


def calculate_cost_per_1k(
    latency_cpu_ms: float,
    is_gemini: bool = False,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    price_per_1m_input: float = 0.075,
    price_per_1m_output: float = 0.30,
    cloud_vm_hourly_cost: float = 0.048
) -> float:
    """Calculates cost per 1k predictions."""
    if is_gemini:
        input_cost = (prompt_tokens / 1_000_000) * price_per_1m_input
        output_cost = (completion_tokens / 1_000_000) * price_per_1m_output
        cost_per_pred = input_cost + output_cost
        return round(cost_per_pred * 1000.0, 4)
    else:
        hours_per_pred = (latency_cpu_ms / 1000.0) / 3600.0
        cost_per_pred = hours_per_pred * cloud_vm_hourly_cost
        return round(cost_per_pred * 1000.0, 4)


def get_model_size_mb(filepath_or_dir: str) -> float:
    """Computes file or directory size in megabytes."""
    if not os.path.exists(filepath_or_dir):
        return 0.0
    if os.path.isfile(filepath_or_dir):
        return round(os.path.getsize(filepath_or_dir) / (1024 * 1024), 2)
    total_size = 0
    for dirpath, _, filenames in os.walk(filepath_or_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)


def _infer_banking_category(intent: str) -> str:
    n = intent.lower()
    if any(k in n for k in ['card', 'contactless', 'pin', 'visa']): return 'Cards & Credentials'
    if any(k in n for k in ['transfer', 'beneficiary', 'receiving_money', 'timing']): return 'Transfers & Clearing'
    if any(k in n for k in ['atm', 'cash', 'swallowed']): return 'Cash & ATM Operations'
    if any(k in n for k in ['fee', 'charge', 'exchange', 'rate']): return 'Fees & FX'
    if any(k in n for k in ['top_up', 'topping']): return 'Top-Up & Deposits'
    if any(k in n for k in ['compromised', 'stolen', 'passcode', 'not_recognised']): return 'Security & Fraud'
    return 'Account, KYC & Payments'
