"""
IntentRouter - Baseline 1: TF-IDF (1-2 ngrams) + Logistic Regression
Evaluates linear baseline performance, per-sample CPU latency, and saves metrics.
"""

import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.data import load_banking77_dataset
from ml.metrics import (
    compute_classification_metrics,
    benchmark_latency,
    calculate_cost_per_1k,
    get_model_size_mb
)


def train_tfidf_baseline(output_dir: str = "artifacts/tfidf"):
    os.makedirs(output_dir, exist_ok=True)
    splits, _ = load_banking77_dataset()

    train_texts = splits["train"]["text"]
    train_labels = splits["train"]["label"]
    test_texts = splits["test"]["text"]
    test_labels = splits["test"]["label"]

    subset_texts = splits["test_subset_308"]["text"]
    subset_labels = splits["test_subset_308"]["label"]

    print("Building TF-IDF (1-2 ngrams) + Logistic Regression pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=4.0, random_state=42, solver="lbfgs"))
    ])

    print("Fitting baseline model...")
    pipeline.fit(train_texts, train_labels)

    model_path = os.path.join(output_dir, "tfidf_logreg.joblib")
    joblib.dump(pipeline, model_path)
    model_size_mb = get_model_size_mb(model_path)

    # Full test set evaluation
    test_preds = pipeline.predict(test_texts)
    test_probs = pipeline.predict_proba(test_texts)
    metrics_full = compute_classification_metrics(test_labels, test_preds.tolist(), test_probs)

    # Subset 308 evaluation (for fair comparison with Gemini)
    sub_preds = pipeline.predict(subset_texts)
    sub_probs = pipeline.predict_proba(subset_texts)
    metrics_subset = compute_classification_metrics(subset_labels, sub_preds.tolist(), sub_probs)

    # Benchmark CPU latency
    sample_queries = test_texts[:10]
    latency_cpu_ms = benchmark_latency(lambda q: pipeline.predict(q), sample_queries)
    estimated_cost_1k = calculate_cost_per_1k(latency_cpu_ms)

    results = {
        "id": "baseline_tfidf",
        "name": "TF-IDF (1-2 grams) + Logistic Regression",
        "category": "baseline",
        "is_baseline": True,
        "is_local": True,
        "accuracy_full": metrics_full["accuracy"],
        "macro_f1_full": metrics_full["macro_f1"],
        "weighted_f1_full": metrics_full["weighted_f1"],
        "top3_accuracy_full": metrics_full["top3_accuracy"],
        "accuracy_subset_308": metrics_subset["accuracy"],
        "macro_f1_subset_308": metrics_subset["macro_f1"],
        "weighted_f1_subset_308": metrics_subset["weighted_f1"],
        "top3_accuracy_subset_308": metrics_subset["top3_accuracy"],
        "latency_cpu_ms": latency_cpu_ms,
        "latency_gpu_ms": None,
        "model_size_mb": model_size_mb,
        "estimated_cost_per_1k": estimated_cost_1k,
        "cost_type": "compute_estimate",
        "notes": "Classic linear n-gram baseline. Fast CPU inference, solid baseline for high-frequency terms."
    }

    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"TF-IDF Baseline Results: Acc={results['accuracy_full']} | F1={results['macro_f1_full']} | Latency={latency_cpu_ms}ms")
    return results


if __name__ == "__main__":
    train_tfidf_baseline()
