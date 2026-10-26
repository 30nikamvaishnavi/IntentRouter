"""
IntentRouter - Gradio Live Demo for Hugging Face Spaces
Classifies banking customer queries into Banking77 intents, returns top-5 probabilities,
measures inference latency, and provides a 'Compare with Gemini' baseline feature.

Strict Requirements:
1. No example-matcher or mock fallback. Uses real trained models. If no trained model is found,
   stops with: "No trained model found. Run the training cells first."
2. Reads best local model from results_bundle.json (highest validation macro-F1).
   Displays model name, path, and modified time in the table header.
3. Check messages section evaluates data/check_messages.json with run_check_messages()
   reporting in-scope accuracy as 'correct / total', top-3 accuracy, avg confidence (correct vs wrong),
   and out-of-scope threshold shares for 0.4, 0.5, 0.6, 0.7, 0.8.

DISCLAIMER: Demo only, not for real banking decisions.
"""

import os
import time
import json
from typing import Dict, Tuple, List, Any
import gradio as gr

from ml.check_messages import (
    load_label_names,
    get_best_local_predictor,
    run_check_messages
)

# Load 77 official Banking77 labels
INTENT_LABELS = []
try:
    labels_map = load_label_names()
    INTENT_LABELS = [labels_map[i] for i in range(len(labels_map))]
except Exception:
    pass

# Check if GEMINI_API_KEY is configured
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def local_inference(text: str) -> Tuple[Dict[str, float], str]:
    """
    Executes real inference with the best trained local model.
    Stops with error if no trained model exists.
    """
    try:
        predictor, model_info = get_best_local_predictor()
    except RuntimeError as e:
        return {}, f"Error: {str(e)}"

    start_time = time.perf_counter()
    scored = predictor(text)
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    top5 = {name: round(score, 4) for name, score in scored[:5]}
    return top5, f"{latency_ms} ms ({model_info['name']})"


def compare_with_gemini(text: str) -> str:
    """Invokes Gemini Zero-Shot model using Gemini API if key is present."""
    if not GEMINI_KEY:
        return "⚠️ Gemini API key not configured. Set GEMINI_API_KEY environment variable in Hugging Face Space secrets."

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_KEY)
        start_t = time.perf_counter()

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Customer banking query: \"{text}\"\nClassify into exactly one of the 77 Banking77 intents.",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are an intent router. Return JSON with 'intent' from the 77 allowed Banking77 labels "
                    "and a short 'reasoning' field."
                ),
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        latency = round((time.perf_counter() - start_t) * 1000.0, 1)
        data = json.loads(resp.text)
        return (
            f"Predicted Intent: {data.get('intent', 'unknown')}\n"
            f"Reasoning: {data.get('reasoning', '')}\n"
            f"Latency: {latency} ms (API roundtrip)\n"
            f"Model: {GEMINI_MODEL}"
        )
    except Exception as e:
        return f"Gemini Error: {str(e)}"


def predict(message: str):
    if not message.strip():
        return {}, "0.0 ms"
    probs, lat = local_inference(message)
    return probs, lat


def check_messages_ui():
    """
    Evaluates data/check_messages.json using run_check_messages() from ml.check_messages.
    Returns:
    - header_markdown (model name, path, modified time, evaluation time)
    - in_scope_acc (as 'correct / total' with percentage)
    - top3_acc (as 'correct / total' with percentage)
    - conf_corr (avg confidence for correct answers)
    - conf_wrong (avg confidence for wrong answers)
    - oos_summary (threshold shares for 0.4, 0.5, 0.6, 0.7, 0.8)
    - table_data (rows for dataframe)
    """
    data_path = "data/check_messages.json"
    if not os.path.exists(data_path) and os.path.exists("public/data/check_messages.json"):
        data_path = "public/data/check_messages.json"

    try:
        res = run_check_messages(data_path)
    except RuntimeError as e:
        err_msg = f"⚠️ {str(e)}"
        return err_msg, "N/A", "N/A", "N/A", "N/A", "N/A", []
    except Exception as e:
        err_msg = f"⚠️ Evaluation error: {str(e)}"
        return err_msg, "N/A", "N/A", "N/A", "N/A", "N/A", []

    header_md = (
        f"### 🤖 Active Model: **{res['model_name']}**\n"
        f"- **Artifact Path**: `{res['model_path']}`\n"
        f"- **File Modified Time**: {res['model_modified_time']}\n"
        f"- **Run Time**: {res['run_time']}"
    )

    in_scope_acc = f"{res['in_scope_accuracy']} ({res['in_scope_accuracy_rate'] * 100:.1f}%)"
    top3_acc = f"{res['top3_accuracy']} ({res['top3_accuracy_rate'] * 100:.1f}%)"
    conf_corr = f"{res['avg_confidence_correct']:.4f}"
    conf_wrong = (
        f"{res['avg_confidence_wrong']:.4f}"
        if res["avg_confidence_wrong"] is not None
        else "N/A (0 errors)"
    )

    oos_shares = res["out_of_scope_shares_above_thresholds"]
    oos_counts = res["out_of_scope_counts_above_thresholds"]
    oos_summary = (
        f">0.4: {oos_shares['0.4']*100:.1f}% ({oos_counts['0.4']}) | "
        f">0.5: {oos_shares['0.5']*100:.1f}% ({oos_counts['0.5']}) | "
        f">0.6: {oos_shares['0.6']*100:.1f}% ({oos_counts['0.6']}) | "
        f">0.7: {oos_shares['0.7']*100:.1f}% ({oos_counts['0.7']}) | "
        f">0.8: {oos_shares['0.8']*100:.1f}% ({oos_counts['0.8']})"
    )

    table_data = []
    for r in res["rows"]:
        table_data.append([
            r["message"],
            r["expected_intent"],
            r["predicted_intent"],
            f"{r['confidence']:.2f}",
            r["top3_intents"],
            r["status"],
            r["note"]
        ])

    return header_md, in_scope_acc, top3_acc, conf_corr, conf_wrong, oos_summary, table_data


# Define Gradio UI
with gr.Blocks(title="IntentRouter - Banking Intent Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # 🏦 IntentRouter: Banking Intent Classification
        ### Fine-Tuned Local Models vs. Zero-Shot {GEMINI_MODEL} on **Banking77** (77 Fine-Grained Classes)
        > **⚠️ Note**: Demo only, not for real banking decisions.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="Customer Banking Message",
                placeholder="e.g. 'I tried to withdraw cash from the ATM at the airport, but the machine swallowed my card!'",
                lines=3
            )
            with gr.Row():
                submit_btn = gr.Button("⚡ Route Intent (Local Model)", variant="primary")
                gemini_btn = gr.Button(f"✨ Compare with {GEMINI_MODEL}", variant="secondary")

            gr.Examples(
                examples=[
                    ["My card expired last week and I haven't received a replacement in the mail."],
                    ["I was charged an extra $5 foreign currency fee for my coffee in Paris."],
                    ["The cash machine took my debit card and never gave it back."],
                    ["Can I set up a single-use virtual card for online purchases on unknown sites?"],
                    ["Why did my wire transfer to my landlord bounce back today?"]
                ],
                inputs=input_text
            )

        with gr.Column(scale=2):
            output_probs = gr.Label(num_top_classes=5, label="Top 5 Predicted Intents (Local Model)")
            latency_out = gr.Textbox(label="Inference Latency & Active Model", interactive=False)
            gemini_out = gr.Textbox(label="Gemini Comparison Result", lines=4, interactive=False)

    submit_btn.click(fn=predict, inputs=input_text, outputs=[output_probs, latency_out])
    gemini_btn.click(fn=compare_with_gemini, inputs=input_text, outputs=gemini_out)

    # --- Check Messages Section ---
    gr.Markdown("---")
    with gr.Accordion("📋 Check Messages (data/check_messages.json)", open=True):
        gr.Markdown(
            "Batch evaluate `data/check_messages.json` with the best fine-tuned local model "
            "(selected by highest validation macro-F1 from `results_bundle.json`). "
            "Reports top-1 prediction, confidence, top-3 candidates, in-scope accuracy as `correct / total`, "
            "and out-of-scope confidence shares across thresholds (0.4, 0.5, 0.6, 0.7, 0.8)."
        )
        check_btn = gr.Button("▶ Run Check Messages on data/check_messages.json", variant="primary")

        model_header_md = gr.Markdown("### Active Model: *(Click button to evaluate)*")

        with gr.Row():
            acc_metric = gr.Textbox(label="In-Scope Accuracy (correct / total)", interactive=False)
            top3_metric = gr.Textbox(label="Top-3 Accuracy", interactive=False)
            conf_corr_metric = gr.Textbox(label="Avg Confidence (Correct)", interactive=False)
            conf_wrong_metric = gr.Textbox(label="Avg Confidence (Wrong)", interactive=False)

        oos_metric = gr.Textbox(label="Out-of-Scope Shares Above Thresholds (>0.4, >0.5, >0.6, >0.7, >0.8)", interactive=False)

        messages_table = gr.Dataframe(
            headers=["Message", "Expected Intent", "Predicted Intent", "Confidence", "Top-3 Intents", "Status", "Note"],
            datatype=["str", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True
        )

        check_btn.click(
            fn=check_messages_ui,
            inputs=[],
            outputs=[model_header_md, acc_metric, top3_metric, conf_corr_metric, conf_wrong_metric, oos_metric, messages_table]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
