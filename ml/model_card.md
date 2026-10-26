---
language:
- en
license: mit
tags:
- text-classification
- intent-classification
- banking
- customer-support
- distilbert
- lora
- peft
datasets:
- PolyAI/banking77
metrics:
- accuracy
- f1
model-index:
- name: IntentRouter-DistilBERT-Banking77
  results:
  - task:
      type: text-classification
      name: Intent Classification
    dataset:
      name: PolyAI/banking77
      type: PolyAI/banking77
      split: test
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9282
    - name: Macro F1
      type: f1
      value: 0.9244
    - name: Top-3 Accuracy
      type: top_3_accuracy
      value: 0.9685
---

# IntentRouter: Fine-Tuned DistilBERT for Banking Intent Routing

IntentRouter is a lightweight, low-latency intent classifier fine-tuned on the **Banking77** dataset (77 fine-grained customer query classes). Designed for financial customer operations, it serves as a high-throughput, low-cost router that categorizes incoming queries in under **13 ms on CPU** (and **2.3 ms on GPU**) before routing requests to human specialists, automated workflows, or an LLM fallback.

## Model Summary

- **Base Architecture**: `distilbert-base-uncased` (6 transformer layers, 768 hidden dimensions, 12 attention heads)
- **Parameters**: 66,955,085 (Full Fine-Tuning) / 590,669 (LoRA PEFT adapter, 0.87% trainable parameters)
- **Quantization**: INT8 PyTorch Dynamic Quantization reduces disk footprint to **67.2 MB** with 2.28x CPU speedup.
- **Task**: 77-class Single-Label Text Classification
- **Domain**: Retail Banking, Fintech, Payments, Cards, ATM, Transfers, AML & KYC

---

## Intended Use

### Primary Applications
1. **First-Response Intent Routing**: Automated triage of customer support chats, emails, and ticket queues into specialized department queues (e.g., Disputes, Lost Cards, Wire Operations, Card Logistics).
2. **Deterministic Bot Dispatch**: Triggering authenticated transactional workflows without expensive LLM API invocations.
3. **Escalation & Safety Guardrails**: Low-latency sentiment and security triage (e.g., immediate freeze recommendations on `compromised_card`).

### Out-of-Scope & Prohibited Use
- **Automated Financial Approvals**: This model must **NOT** make automated credit, loan, or account closure decisions without human banker supervision.
- **Authentication Credentials**: Never train or deploy on raw customer passwords, full 16-digit PANs, or CVV codes.

---

## Dataset & Training Procedure

- **Source Dataset**: `PolyAI/banking77` (Casanueva et al., 2020)
- **Splits**:
  - Training: 9,003 examples (stratified 90% split)
  - Validation: 1,000 examples (stratified 10% split, fixed seed 42)
  - Test: 3,080 examples (official test split, completely held out during tuning)
  - Gemini Comparison Subset: 308 stratified examples (4 examples per class)
- **Hyperparameters**:
  - Learning Rate: `5e-5` (Linear decay with warmup)
  - Batch Size: `32`
  - Max Sequence Length: `64` tokens
  - Optimization: AdamW, weight decay `0.01`, mixed precision FP16
  - Epochs: 3 (with early stopping patience 2 on validation accuracy)

---

## Evaluation & Benchmarks

All models were evaluated on both the **full test set (3,080 samples)** and the **stratified fair-comparison subset (308 samples)** alongside a zero-shot/few-shot **Gemini 2.5 Flash** baseline:

| Model Architecture | Accuracy (Full) | Macro-F1 (Full) | Top-3 Acc | Latency (CPU) | Size | Cost / 1k queries |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Reg (Baseline)** | 85.23% | 84.68% | 94.12% | **1.4 ms** | 18.4 MB | $0.0004 |
| **DistilBERT (Full Fine-Tuning)** | **92.82%** | **92.44%** | **96.85%** | 12.8 ms | 268 MB | $0.0036 |
| **DistilBERT + LoRA (PEFT r=16)** | 92.14% | 91.78% | 96.43% | 12.8 ms | **2.4 MB** | $0.0036 |
| **DistilBERT (PyTorch INT8 CPU)** | 92.40% | 91.96% | 96.62% | **5.6 ms** | 67.2 MB | $0.0016 |
| **Gemini 2.5 Flash (Zero-Shot)** | 87.34%* | 86.82%* | N/A | 582 ms | API | $0.0385 |
| **Gemini 2.5 Flash (5-Shot)** | 90.58%* | 90.14%* | N/A | 845 ms | API | $0.1850 |

*\* Evaluated on the 308-sample stratified subset due to rate-limiting constraints.*

---

## Limitations & Biases

1. **Single Dataset**: The model is exclusively trained on PolyAI/banking77. Its performance on enterprise-specific jargon, mortgage products, commercial treasury, or merchant acquirer queries may require domain adaptation.
2. **English Only**: Training queries are in modern conversational British and American English. It does not generalize to multilingual or code-switched queries.
3. **Short Context**: Optimized for customer utterance lengths (<64 tokens); performance degrades on long multi-paragraph email threads.
4. **Subtle Intent Overlap**: Top confused pairs include semantically adjacent operations such as `card_arrival` vs `card_delivery_estimate`, and `transfer_not_received_by_recipient` vs `pending_transfer`. In production, these should trigger joint fallback resolution.

---

## Citation & Acknowledgments

```bibtex
@inproceedings{casanueva2020efficient,
  title={Efficient Intent Detection with Dual Sentence Encoders},
  author={Casanueva, I{\~n}igo and Tem{\v{c}}inas, Tadas and Daniela, Gerz and Matthew, Henderson and Ivan, Vuli{\'c}},
  booktitle={Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI},
  pages={38--45},
  year={2020}
}
```
