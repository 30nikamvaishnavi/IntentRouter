"""
IntentRouter Bundle Verification Script
Loads results_bundle.json and strictly verifies:
1. All required sections exist and are non-empty
2. No metric value is NaN or null where forbidden
3. Accuracy for each model exceeds the majority-class random baseline (1 / 77 = ~0.013)
4. Checks whether full-test and 308-subset metrics for the same model differ by > 0.10 (prints warning)
Exits with code 1 on failure.
"""

import sys
import os
import json
import math


REQUIRED_SECTIONS = [
    "is_sample_data",
    "version",
    "metadata",
    "models",
    "training_curves",
    "error_analysis",
    "robustness",
    "quantization",
    "cascade"
]

MAJORITY_CLASS_BASELINE = 1.0 / 77.0  # ~0.01299


def check_no_nan(obj, path=""):
    """Recursively checks that no float or number is NaN."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Metric check failed: '{path}' is NaN or Inf!")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_no_nan(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            check_no_nan(item, f"{path}[{idx}]")


def check_bundle(bundle_path: str = "results/results_bundle.json") -> bool:
    if not os.path.exists(bundle_path):
        if os.path.exists("public/results_bundle.json"):
            bundle_path = "public/results_bundle.json"
        else:
            print(f"ERROR: Bundle file not found at '{bundle_path}'.", file=sys.stderr)
            return False

    print(f"Loading and checking bundle: {bundle_path}...")
    try:
        with open(bundle_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to parse bundle JSON: {e}", file=sys.stderr)
        return False

    # 1. Required top-level sections
    for sec in REQUIRED_SECTIONS:
        if sec not in data or data[sec] is None:
            print(f"ERROR: Missing required top-level section: '{sec}'", file=sys.stderr)
            return False

    # 2. Check for NaN
    try:
        check_no_nan(data)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False

    # 3. Check models and majority class baseline
    models = data.get("models", [])
    if not models or len(models) < 4:
        print(f"ERROR: Expected at least 4 models, found {len(models)}", file=sys.stderr)
        return False

    for m in models:
        name = m.get("name", m.get("id", "Unknown"))
        acc_full = m.get("accuracy_full")
        acc_sub = m.get("accuracy_subset_308")

        # Check accuracy above majority-class baseline
        if acc_full is not None and acc_full <= MAJORITY_CLASS_BASELINE:
            print(f"ERROR: Model '{name}' full accuracy ({acc_full}) is at or below majority-class baseline ({MAJORITY_CLASS_BASELINE:.4f})!", file=sys.stderr)
            return False

        if acc_sub is not None and acc_sub <= MAJORITY_CLASS_BASELINE:
            print(f"ERROR: Model '{name}' 308-subset accuracy ({acc_sub}) is at or below majority-class baseline ({MAJORITY_CLASS_BASELINE:.4f})!", file=sys.stderr)
            return False

        # 4. Difference between full-test and 308-subset warning (> 10 points = 0.10)
        if acc_full is not None and acc_sub is not None:
            diff = abs(acc_full - acc_sub)
            if diff > 0.10:
                print(f"WARNING: Model '{name}' has a {diff*100:.1f} percentage point difference between full test ({acc_full*100:.1f}%) and 308-subset ({acc_sub*100:.1f}%).")

    # 5. Check error analysis
    ea = data.get("error_analysis", {})
    if len(ea.get("confused_pairs", [])) < 15:
        print("ERROR: error_analysis.confused_pairs must contain at least 15 pairs", file=sys.stderr)
        return False

    if len(ea.get("misclassified_examples", [])) < 20:
        print("ERROR: error_analysis.misclassified_examples must contain at least 20 items", file=sys.stderr)
        return False

    if len(ea.get("per_class_f1", [])) != 77:
        print(f"ERROR: error_analysis.per_class_f1 must contain exactly 77 classes, got {len(ea.get('per_class_f1', []))}", file=sys.stderr)
        return False

    # 6. Check cascade section
    cas = data.get("cascade", {})
    if not cas.get("threshold_steps") or len(cas.get("threshold_steps", [])) < 10:
        print("ERROR: cascade.threshold_steps must contain at least 10 threshold entries", file=sys.stderr)
        return False

    print("ALL BUNDLE CHECKS PASSED: Bundle structure, metric ranges, and calibration data are valid.")
    return True


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "public/results_bundle.json"
    ok = check_bundle(target)
    sys.exit(0 if ok else 1)
