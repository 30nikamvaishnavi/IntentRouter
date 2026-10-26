"""
ml/make_readme_results.py: Generates empirical benchmark results into README.md

Reads a verified real results_bundle.json (never sample or public/ files)
and writes the results tables:
1. Model Comparison Table
2. Confidence Cascade Table
3. Confusion Pairs Table
4. Robustness Table
5. Check Messages Table
into README.md strictly between <!-- RESULTS_START --> and <!-- RESULTS_END -->.

Strict rules enforced:
- Must stop with an error if path contains 'sample' or is in 'public/'.
- Must stop with an error if the bundle's metadata is missing:
  * A run date
  * A GPU name
  * Library versions
- Must stop with an error if any sample flag is set (e.g. is_sample_data: True).
- Must NEVER fill a value that is missing with defaults or synthetic numbers.
"""

import sys
import os
import json
from typing import Dict, Any, List, Optional


RESULTS_START_MARKER = "<!-- RESULTS_START -->"
RESULTS_END_MARKER = "<!-- RESULTS_END -->"


def validate_file_path(path: str) -> str:
    """
    Validates that the bundle path is not a sample file and not in public/.
    """
    normalized = path.replace("\\", "/")
    filename = os.path.basename(normalized).lower()

    if "sample" in filename or "sample" in normalized.lower():
        raise ValueError(
            f"Invalid bundle path '{path}': files containing 'sample' in name or path are forbidden. "
            "Only real run bundles are permitted."
        )

    parts = [p.lower() for p in normalized.split("/") if p]
    if "public" in parts or normalized.startswith("public/"):
        raise ValueError(
            f"Invalid bundle path '{path}': reading from 'public/' is forbidden. "
            "Please provide a real results bundle from the training run (e.g., 'results/results_bundle.json')."
        )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results bundle not found at '{path}'. Run the training cells in IntentRouter.ipynb first."
        )

    return path


def validate_bundle_metadata(bundle: Dict[str, Any], path: str):
    """
    Strictly verifies bundle metadata:
    - Sample flag must NOT be true.
    - Run date must be present and non-empty.
    - GPU name must be present and non-empty.
    - Library versions must be present and non-empty.
    """
    # 1. Sample flag check
    is_sample = False
    if bundle.get("is_sample_data") is True or bundle.get("is_sample") is True or bundle.get("sample") is True:
        is_sample = True

    metadata = bundle.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Bundle '{path}' is missing the required 'metadata' section.")

    if metadata.get("is_sample_data") is True or metadata.get("is_sample") is True or metadata.get("sample") is True:
        is_sample = True

    if is_sample:
        raise ValueError(
            f"Sample flag is set in '{path}' (is_sample_data=True). "
            "This bundle contains sample/mock data. Run the training notebook on a GPU to produce real results."
        )

    # 2. Run date check
    run_date = metadata.get("run_date") or metadata.get("timestamp")
    if not run_date or not isinstance(run_date, str) or not run_date.strip():
        raise ValueError(f"Bundle '{path}' metadata is missing a valid 'run_date'.")

    # 3. GPU name check
    hardware = metadata.get("hardware")
    gpu_name = None
    if isinstance(hardware, dict):
        gpu_name = hardware.get("gpu_name")
    if not gpu_name:
        gpu_name = metadata.get("gpu_name")

    if not gpu_name or not isinstance(gpu_name, str) or not gpu_name.strip():
        raise ValueError(f"Bundle '{path}' metadata is missing a valid 'gpu_name' under hardware.")

    # 4. Library versions check
    libraries = metadata.get("libraries")
    if not libraries or not isinstance(libraries, dict) or len(libraries) == 0:
        raise ValueError(f"Bundle '{path}' metadata is missing required 'libraries' versions.")

    for lib_key in ["torch", "transformers"]:
        val = libraries.get(lib_key)
        if not val or not str(val).strip():
            raise ValueError(f"Bundle '{path}' metadata is missing library version for '{lib_key}'.")


def format_model_comparison_table(models: List[Dict[str, Any]]) -> str:
    """Formats the model comparison table without ever synthesizing missing numbers."""
    if not models or not isinstance(models, list):
        raise ValueError("Results bundle is missing 'models' list.")

    lines = [
        "### 1. Model Comparison",
        "| Model | Accuracy | Macro-F1 | CPU latency | Cost per 1k |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]

    for m in models:
        name = m.get("name")
        if not name:
            raise ValueError("Model entry is missing 'name'.")

        # Determine accuracy
        acc = m.get("accuracy_full")
        if acc is None:
            acc = m.get("accuracy_subset_308")
            acc_note = "*"
        else:
            acc_note = ""

        if acc is None:
            raise ValueError(f"Model '{name}' is missing accuracy metrics (neither accuracy_full nor accuracy_subset_308 found).")

        # Determine macro-F1
        f1 = m.get("macro_f1_full")
        if f1 is None:
            f1 = m.get("macro_f1_subset_308")
        if f1 is None:
            raise ValueError(f"Model '{name}' is missing macro_f1 metrics.")

        # Latency
        latency = m.get("latency_cpu_ms")
        if latency is None:
            raise ValueError(f"Model '{name}' is missing 'latency_cpu_ms'.")

        # Cost
        cost = m.get("estimated_cost_per_1k")
        if cost is None:
            raise ValueError(f"Model '{name}' is missing 'estimated_cost_per_1k'.")

        acc_str = f"{acc * 100:.2f}%{acc_note}"
        f1_str = f"{f1 * 100:.2f}%"
        lat_str = f"{latency:.2f} ms"
        cost_str = f"${cost:.4f}"

        lines.append(f"| {name} | {acc_str} | {f1_str} | {lat_str} | {cost_str} |")

    lines.append("")
    lines.append(
        "*Note: Values marked with `*` indicate evaluation on the 308-sample stratified test subset "
        "(4 per class) due to API rate limits. Local models are evaluated across the full 3,080 test set.*"
    )
    lines.append("")
    return "\n".join(lines)


def format_cascade_table(cascade_data: Dict[str, Any]) -> str:
    """Formats the confidence cascade threshold table without default fillers."""
    if not cascade_data or not isinstance(cascade_data, dict):
        raise ValueError("Results bundle is missing 'cascade' data.")

    ece = cascade_data.get("expected_calibration_error")
    if ece is None:
        raise ValueError("Cascade data is missing 'expected_calibration_error'.")

    steps = cascade_data.get("threshold_steps")
    if not steps or not isinstance(steps, list):
        raise ValueError("Cascade data is missing 'threshold_steps'.")

    lines = [
        "### 2. Confidence Cascade (Threshold Routing)",
        f"**Expected Calibration Error (ECE):** `{ece:.4f}`\n",
        "| Confidence Threshold ($\tau$) | Local Coverage | Local Model Accuracy | Cascade Accuracy | Escalated Queries | Est. Cost / 1k | Avg Latency |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for step in steps:
        t = step.get("threshold")
        cov = step.get("coverage")
        local_acc = step.get("local_accuracy")
        cascade_acc = step.get("cascade_accuracy_subset_308")
        if cascade_acc is None:
            cascade_acc = step.get("cascade_accuracy")
        escalated = step.get("gemini_count_subset_308")
        if escalated is None:
            escalated = step.get("escalated_count")
        cost = step.get("estimated_cost_per_1k")
        lat = step.get("average_latency_ms")

        # Strict validation: never fill missing values
        if any(v is None for v in [t, cov, local_acc, cascade_acc, cost, lat]):
            raise ValueError(f"Cascade threshold step for τ={t} contains missing metric values.")

        esc_str = str(escalated) if escalated is not None else "---"
        rec_mark = " *(Recommended)*" if abs(t - 0.80) < 1e-4 else ""

        lines.append(
            f"| **{t:.2f}**{rec_mark} | {cov * 100:.2f}% | {local_acc * 100:.2f}% | "
            f"{cascade_acc * 100:.2f}% | {esc_str} | ${cost:.4f} | {lat:.1f} ms |"
        )

    lines.append("")
    return "\n".join(lines)


def format_confusion_pairs_table(error_analysis: Dict[str, Any]) -> str:
    """Formats top confused intent pairs."""
    if not error_analysis or not isinstance(error_analysis, dict):
        raise ValueError("Results bundle is missing 'error_analysis' data.")

    pairs = error_analysis.get("confused_pairs")
    if not pairs or not isinstance(pairs, list):
        raise ValueError("Error analysis is missing 'confused_pairs'.")

    lines = [
        "### 3. Confused Intent Pairs",
        "| Rank | True Intent | Predicted Intent | Error Count | Share of Class Errors | Root Cause / Explanation |",
        "| :---: | :--- | :--- | :---: | :---: | :--- |"
    ]

    for p in pairs[:10]:
        rank = p.get("rank")
        t_int = p.get("true_intent")
        p_int = p.get("predicted_intent")
        cnt = p.get("count")
        pct = p.get("percentage_of_class_errors")
        exp = p.get("explanation", "")

        if any(v is None for v in [rank, t_int, p_int, cnt, pct]):
            raise ValueError(f"Confused pair entry {p} has missing required metrics.")

        lines.append(f"| {rank} | `{t_int}` | `{p_int}` | {cnt} | {pct:.1f}% | {exp} |")

    lines.append("")
    return "\n".join(lines)


def format_robustness_table(robustness_data: Dict[str, Any]) -> str:
    """Formats robustness perturbations table."""
    if not robustness_data or not isinstance(robustness_data, dict):
        raise ValueError("Results bundle is missing 'robustness' data.")

    tests = robustness_data.get("tests")
    if not tests or not isinstance(tests, list):
        raise ValueError("Robustness data is missing 'tests'.")

    lines = [
        "### 4. Robustness to Conversational Perturbations",
        "| Perturbation Type | DistilBERT Clean | DistilBERT Perturbed | DistilBERT Drop ($\Delta$) | Gemini Clean | Gemini Perturbed | Gemini Drop ($\Delta$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for t in tests:
        p_type = t.get("perturbation_type")
        c_d = t.get("clean_accuracy_distilbert")
        p_d = t.get("perturbed_accuracy_distilbert")
        d_d = t.get("delta_accuracy_distilbert")
        c_g = t.get("clean_accuracy_gemini")
        p_g = t.get("perturbed_accuracy_gemini")
        d_g = t.get("delta_accuracy_gemini")

        if any(v is None for v in [p_type, c_d, p_d, d_d, c_g, p_g, d_g]):
            raise ValueError(f"Robustness test entry '{p_type}' has missing required values.")

        lines.append(
            f"| **{p_type}** | {c_d * 100:.2f}% | {p_d * 100:.2f}% | **{d_d * 100:+.2f}%** | "
            f"{c_g * 100:.2f}% | {p_g * 100:.2f}% | **{d_g * 100:+.2f}%** |"
        )

    lines.append("")
    return "\n".join(lines)


def load_check_messages_data(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads check messages data from bundle if embedded, or from results/check_messages_results.json.
    Never invents or falls back to synthetic numbers.
    """
    if "check_messages" in bundle and isinstance(bundle["check_messages"], dict):
        return bundle["check_messages"]

    check_path = "results/check_messages_results.json"
    if os.path.exists(check_path):
        with open(check_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(
        "Check messages results are missing from both results_bundle.json and results/check_messages_results.json. "
        "Run 'python ml/check_messages.py data/check_messages.json' with a trained model first."
    )


def format_check_messages_section(check_data: Dict[str, Any]) -> str:
    """Formats the check messages evaluation table and metrics."""
    in_scope_acc = check_data.get("in_scope_accuracy")
    top3_acc = check_data.get("top3_accuracy")
    avg_conf_correct = check_data.get("avg_confidence_correct")
    avg_conf_wrong = check_data.get("avg_confidence_wrong")
    oos_shares = check_data.get("out_of_scope_shares_above_thresholds", {})
    rows = check_data.get("rows", [])
    model_name = check_data.get("model_name", "Local Model")
    run_time = check_data.get("run_time", "N/A")

    if in_scope_acc is None or avg_conf_correct is None or not rows:
        raise ValueError("Check messages results contain missing summary metrics.")

    lines = [
        "### 5. Check Messages & Out-of-Scope Calibration",
        f"- **Model Evaluated:** `{model_name}` (Run Time: `{run_time}`)",
        f"- **In-Scope Accuracy:** `{in_scope_acc}`",
        f"- **Top-3 Accuracy:** `{top3_acc}`",
        f"- **Average Confidence (Correct Answers):** `{avg_conf_correct:.4f}`",
        f"- **Average Confidence (Wrong Answers):** `{avg_conf_wrong if avg_conf_wrong is not None else 'N/A'}`",
        "- **Out-of-Scope Confidence Threshold Distribution:**"
    ]

    for thresh in ["0.4", "0.5", "0.6", "0.7", "0.8"]:
        share = oos_shares.get(thresh)
        if share is not None:
            lines.append(f"  - Above {thresh}: `{share * 100:.1f}%`")
        else:
            raise ValueError(f"Missing out-of-scope share for threshold {thresh}.")

    lines.append("")
    lines.append("| Message | Expected Intent | Predicted Top-1 | Conf | Top-3 Intents | Status |")
    lines.append("| :--- | :--- | :--- | :---: | :--- | :---: |")

    for r in rows:
        msg = r.get("message", "").replace("|", "\\|")
        exp = r.get("expected_intent")
        pred = r.get("predicted_intent")
        conf = r.get("confidence")
        top3 = r.get("top3_intents", "").replace("|", "\\|")
        status = r.get("status")

        if any(v is None for v in [exp, pred, conf, status]):
            raise ValueError(f"Check message row has missing fields: {r}")

        lines.append(f"| {msg} | `{exp}` | `{pred}` | {conf:.4f} | {top3} | **{status}** |")

    lines.append("")
    return "\n".join(lines)


def generate_results_markdown(bundle: Dict[str, Any]) -> str:
    """Builds the complete markdown results section from empirical bundle data."""
    metadata = bundle["metadata"]
    run_date = metadata.get("run_date")
    gpu_name = metadata.get("hardware", {}).get("gpu_name")
    libs = metadata.get("libraries", {})
    lib_str = ", ".join([f"{k} {v}" for k, v in libs.items() if k in ["torch", "transformers", "peft"]])

    header = [
        f"*Empirical Benchmark Run: `{run_date}` | Hardware: `{gpu_name}` | Libraries: `{lib_str}`*",
        ""
    ]

    parts = [
        "\n".join(header),
        format_model_comparison_table(bundle["models"]),
        format_cascade_table(bundle["cascade"]),
        format_confusion_pairs_table(bundle["error_analysis"]),
        format_robustness_table(bundle["robustness"]),
        format_check_messages_section(load_check_messages_data(bundle))
    ]

    return "\n".join(parts)


def update_readme_with_results(readme_path: str, results_md: str):
    """Inserts the generated results markdown between RESULTS markers in README.md."""
    if not os.path.exists(readme_path):
        raise FileNotFoundError(f"README.md not found at '{readme_path}'.")

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    start_idx = content.find(RESULTS_START_MARKER)
    end_idx = content.find(RESULTS_END_MARKER)

    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        raise ValueError(
            f"README.md must contain '{RESULTS_START_MARKER}' followed by '{RESULTS_END_MARKER}' markers."
        )

    before = content[: start_idx + len(RESULTS_START_MARKER)]
    after = content[end_idx:]

    new_content = f"{before}\n\n{results_md.strip()}\n\n{after}"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def main():
    bundle_path = sys.argv[1] if len(sys.argv) > 1 else "results/results_bundle.json"
    readme_path = sys.argv[2] if len(sys.argv) > 2 else "README.md"

    print(f"Validating bundle file path: '{bundle_path}'...")
    valid_path = validate_file_path(bundle_path)

    print(f"Reading bundle from '{valid_path}'...")
    with open(valid_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    print("Strictly checking bundle metadata and sample flags...")
    validate_bundle_metadata(bundle, valid_path)

    print("Generating results tables (model comparison, cascade, confusion pairs, robustness, check messages)...")
    results_md = generate_results_markdown(bundle)

    print(f"Writing generated results into '{readme_path}' between markers...")
    update_readme_with_results(readme_path, results_md)

    print("✓ Successfully updated README.md with empirical results from verified bundle!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
