"""
IntentRouter - Results Export and Bundle Merger
Merges all individual benchmark result files into /results/*.json:
- comparison.json
- per_class.json
- confusion.json
- misclassified.json
- robustness.json
- latency.json
- cascade.json
- metadata.json
and generates the unified results_bundle.json.

CRITICAL REQUIREMENT:
If any required field or metric value is missing, raise an error immediately.
Do NOT substitute default values.
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, Any, List


REQUIRED_MODEL_FIELDS = [
    "id", "name", "category", "is_baseline", "is_local",
    "accuracy_full", "macro_f1_full", "weighted_f1_full",
    "accuracy_subset_308", "macro_f1_subset_308", "weighted_f1_subset_308",
    "latency_cpu_ms", "model_size_mb", "estimated_cost_per_1k",
    "cost_type", "notes"
]


def validate_strict(data: Dict[str, Any], path_prefix: str = ""):
    """Recursively validates that no field has None or missing values where required."""
    for key, value in data.items():
        current_path = f"{path_prefix}.{key}" if path_prefix else key
        if value is None and key not in ["top3_accuracy_full", "top3_accuracy_subset_308", "latency_gpu_ms"]:
            raise ValueError(f"Strict validation failed: '{current_path}' is None or missing! Defaults are forbidden.")
        if isinstance(value, dict):
            validate_strict(value, current_path)


def export_bundle(
    models: List[Dict[str, Any]],
    training_curves: Dict[str, Any],
    confused_pairs: List[Dict[str, Any]],
    misclassified_examples: List[Dict[str, Any]],
    per_class_f1: List[Dict[str, Any]],
    robustness_tests: List[Dict[str, Any]],
    quantization_summary: Dict[str, Any],
    cascade_summary: Dict[str, Any],
    hardware_info: Dict[str, Any],
    library_versions: Dict[str, Any],
    gemini_model: str = "gemini-2.5-flash",
    data_source_used: str = "PolyAI/banking77",
    output_dir: str = "results",
    is_sample_data: bool = False
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("public", exist_ok=True)

    # 1. Validate models list
    if not models or len(models) < 4:
        raise ValueError(f"Expected at least 4 models, got {len(models) if models else 0}")

    for idx, m in enumerate(models):
        for field in REQUIRED_MODEL_FIELDS:
            if field not in m:
                raise ValueError(f"Missing required field '{field}' in model[{idx}] ({m.get('name', 'unknown')})")
            if m[field] is None:
                raise ValueError(f"Field '{field}' in model[{idx}] cannot be None.")

    # 2. Validate error analysis
    if len(confused_pairs) < 15:
        raise ValueError(f"Expected at least 15 confused intent pairs, got {len(confused_pairs)}")
    if len(misclassified_examples) < 20:
        raise ValueError(f"Expected at least 20 misclassified examples, got {len(misclassified_examples)}")
    if len(per_class_f1) != 77:
        raise ValueError(f"Expected exactly 77 intent classes in per_class_f1, got {len(per_class_f1)}")

    # 3. Validate robustness tests
    if len(robustness_tests) < 3:
        raise ValueError(f"Expected at least 3 robustness perturbation tests, got {len(robustness_tests)}")

    # 4. Validate cascade analysis
    if not cascade_summary.get("threshold_steps") or len(cascade_summary["threshold_steps"]) < 10:
        raise ValueError(f"Expected at least 10 cascade threshold steps, got {len(cascade_summary.get('threshold_steps', []))}")

    # Construct metadata
    metadata = {
        "project_name": "IntentRouter - Banking Intent Classification & Benchmarking",
        "run_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "seed": 42,
        "gemini_model": gemini_model,
        "hardware": hardware_info,
        "libraries": library_versions,
        "dataset": {
            "dataset_name": "PolyAI/banking77",
            "source_url": "https://huggingface.co/datasets/PolyAI/banking77",
            "num_classes": 77,
            "train_samples": 10003,
            "val_samples": 1000,
            "test_full_samples": 3080,
            "test_subset_samples": 308,
            "subset_samples_per_class": 4,
            "data_source_used": data_source_used
        },
        "author": "IntentRouter Benchmark Suite"
    }

    validate_strict(metadata, "metadata")
    validate_strict(quantization_summary, "quantization")
    validate_strict(cascade_summary, "cascade")

    # Save individual target files in /results/
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(models, f, indent=2)

    with open(os.path.join(output_dir, "per_class.json"), "w") as f:
        json.dump(per_class_f1, f, indent=2)

    with open(os.path.join(output_dir, "confusion.json"), "w") as f:
        json.dump(confused_pairs, f, indent=2)

    with open(os.path.join(output_dir, "misclassified.json"), "w") as f:
        json.dump(misclassified_examples, f, indent=2)

    with open(os.path.join(output_dir, "robustness.json"), "w") as f:
        json.dump(robustness_tests, f, indent=2)

    with open(os.path.join(output_dir, "cascade.json"), "w") as f:
        json.dump(cascade_summary, f, indent=2)

    latency_summary = [
        {"model": m["name"], "cpu_latency_ms": m["latency_cpu_ms"], "gpu_latency_ms": m.get("latency_gpu_ms")}
        for m in models
    ]
    with open(os.path.join(output_dir, "latency.json"), "w") as f:
        json.dump(latency_summary, f, indent=2)

    # Master merged bundle
    bundle = {
        "is_sample_data": is_sample_data,
        "version": "1.0.0",
        "metadata": metadata,
        "models": models,
        "training_curves": training_curves,
        "error_analysis": {
            "best_model_id": "distilbert_full",
            "confused_pairs": confused_pairs,
            "misclassified_examples": misclassified_examples,
            "per_class_f1": per_class_f1
        },
        "robustness": {
            "note": "Evaluated on small synthetic test set (308 samples, 4 per class) perturbed with typos, lowercasing, and conversational fillers.",
            "tests": robustness_tests
        },
        "quantization": quantization_summary,
        "cascade": cascade_summary
    }

    bundle_path = os.path.join(output_dir, "results_bundle.json")
    with open(bundle_path, "w") as f:
        json.dump(bundle, f, indent=2)

    # Also save to public/results_bundle.json for dashboard
    with open("public/results_bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"Export successful: saved individual /results/*.json files and master {bundle_path}.")
    return bundle


if __name__ == "__main__":
    print("ml/export_results.py ready.")
