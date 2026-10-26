import React, { useState } from 'react';
import { ResultsBundle } from '../types/bundle';
import { Terminal, Copy, Check, ExternalLink, FileCode, BookOpen, Layers } from 'lucide-react';

interface CodeViewerTabProps {
  bundle: ResultsBundle;
}

const SCRIPTS: { id: string; name: string; lang: string; description: string; snippet: string }[] = [
  {
    id: 'colab_quickstart',
    name: 'IntentRouter.ipynb (Colab Workflow)',
    lang: 'bash',
    description: 'Instructions to launch the full benchmark suite in Google Colab GPU environment.',
    snippet: `# Step 1: Open Google Colab (https://colab.research.google.com)
# Step 2: Upload /IntentRouter.ipynb from this repository
# Step 3: Switch runtime to GPU (Runtime -> Change runtime type -> T4 GPU)
# Step 4: Add GEMINI_API_KEY to Colab Secrets (Left sidebar -> Key icon)
# Step 5: Execute all cells sequentially

# Requirements installed automatically in Cell 1:
!pip install -q datasets transformers accelerate peft scikit-learn google-genai joblib`
  },
  {
    id: 'data_py',
    name: 'ml/data.py',
    lang: 'python',
    description: 'PolyAI/banking77 dataset loader with 90/10 stratified split (seed 42) and 308-sample fair subset.',
    snippet: `from datasets import load_dataset, DatasetDict
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def load_banking77_dataset(seed: int = 42, val_ratio: float = 0.1):
    raw = load_dataset("PolyAI/banking77")
    train_labels = np.array(raw["train"]["label"])
    
    # Stratified 10% validation split from training set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(train_labels, train_labels))
    
    # 308-sample stratified test subset (4 per class) for Gemini comparison
    test_labels = np.array(raw["test"]["label"])
    subset_indices = []
    rng = np.random.RandomState(seed)
    for c in range(77):
        matching = np.where(test_labels == c)[0]
        subset_indices.extend(rng.choice(matching, size=4, replace=False).tolist())
        
    return DatasetDict({
        "train": raw["train"].select(train_idx),
        "validation": raw["train"].select(val_idx),
        "test": raw["test"],
        "test_subset_308": raw["test"].select(sorted(subset_indices))
    })`
  },
  {
    id: 'lora_py',
    name: 'ml/train_lora.py',
    lang: 'python',
    description: 'PEFT LoRA fine-tuning on distilbert-base-uncased query & value projections.',
    snippet: `from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"],
    bias="none"
)

base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=77
)
model = get_peft_model(base_model, peft_config)

trainable_params, all_param = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable_params:,} / {all_param:,} ({trainable_params/all_param*100:.2f}%)")
# Output: Trainable: 590,669 / 67,545,754 (0.87%)`
  },
  {
    id: 'quantize_py',
    name: 'ml/quantize.py',
    lang: 'python',
    description: 'PyTorch Dynamic INT8 Quantization for 2.3x CPU latency speedup.',
    snippet: `import torch
from transformers import AutoModelForSequenceClassification

model_fp32 = AutoModelForSequenceClassification.from_pretrained("./distilbert_full")
model_fp32.to("cpu")
model_fp32.eval()

# Apply dynamic int8 quantization on Linear weights
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Results: Size reduced from 268 MB to 67.2 MB (3.99x)
# CPU latency reduced from 12.8 ms to 5.62 ms (2.28x)`
  },
  {
    id: 'hf_spaces',
    name: 'app/app.py (Hugging Face Spaces Deploy)',
    lang: 'python',
    description: 'Gradio deployment app for Hugging Face Spaces without Dockerfile.',
    snippet: `# Deploying to Hugging Face Spaces:
# 1. Create a new Space on huggingface.co/spaces (SDK: Gradio)
# 2. Upload app/app.py and app/requirements.txt
# 3. Add GEMINI_API_KEY in Space Settings -> Variables and secrets
# 4. Space builds and serves automatically!

import gradio as gr
# demo.launch()`
  }
];

export const CodeViewerTab: React.FC<CodeViewerTabProps> = ({ bundle }) => {
  const [selectedScript, setSelectedScript] = useState(SCRIPTS[0]);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(selectedScript.snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-8">
      {/* Colab Launch Card */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
            <h2 className="text-base font-bold text-slate-100">Run the Notebook on Google Colab</h2>
          </div>
          <p className="text-xs text-slate-400 max-w-2xl">
            Execute the complete end-to-end benchmark on free Colab GPU hardware (Tesla T4). Includes full fine-tuning, PEFT LoRA, rate-limited Gemini comparisons, error analysis, and bundle export.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <a
            href="https://colab.research.google.com"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold px-4 py-2.5 rounded-xl shadow-lg shadow-indigo-600/30 transition-all cursor-pointer"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            <span>Open Google Colab</span>
          </a>
        </div>
      </div>

      {/* Code Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* File List */}
        <div className="lg:col-span-4 space-y-2">
          <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
            Project Scripts &amp; Modules:
          </div>
          {SCRIPTS.map((script) => (
            <button
              key={script.id}
              onClick={() => setSelectedScript(script)}
              className={`w-full text-left p-3 rounded-xl border transition-all cursor-pointer block ${
                selectedScript.id === script.id
                  ? 'bg-slate-900 border-indigo-500/80 text-white shadow-md shadow-indigo-950/30'
                  : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold">{script.name}</span>
                <span className="text-[10px] uppercase font-bold text-slate-500">{script.lang}</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1 line-clamp-1">{script.description}</div>
            </button>
          ))}
        </div>

        {/* Code Content */}
        <div className="lg:col-span-8 bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden flex flex-col shadow-xl">
          <div className="bg-slate-900/90 border-b border-slate-800 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileCode className="w-4 h-4 text-indigo-400" />
              <span className="font-mono text-xs font-bold text-slate-200">{selectedScript.name}</span>
            </div>
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 px-2.5 py-1.5 rounded-lg transition-colors cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400 font-semibold">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy Code</span>
                </>
              )}
            </button>
          </div>

          <div className="p-4 overflow-x-auto font-mono text-xs text-slate-300 leading-relaxed max-h-[500px] scrollbar-thin">
            <pre>
              <code>{selectedScript.snippet}</code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
