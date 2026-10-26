import React, { useState } from 'react';
import { ResultsBundle } from '../types/bundle';
import { validateResultsBundle } from '../utils/validateBundle';
import { X, Upload, CheckCircle2, AlertTriangle, FileJson, RefreshCw } from 'lucide-react';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onBundleUpdated: (newBundle: ResultsBundle) => void;
  onResetToSample: () => void;
  currentIsSample: boolean;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onBundleUpdated,
  onResetToSample,
  currentIsSample,
}) => {
  const [jsonText, setJsonText] = useState('');
  const [validationError, setValidationError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setJsonText(content);
      validateAndApply(content);
    };
    reader.readAsText(file);
  };

  const validateAndApply = (content: string) => {
    setValidationError(null);
    setSuccessMsg(null);

    let parsed: unknown;
    try {
      parsed = JSON.parse(content);
    } catch (err: unknown) {
      setValidationError("Syntax Error: Input is not valid JSON.");
      return;
    }

    const result = validateResultsBundle(parsed);
    if (!result.valid || !result.bundle) {
      setValidationError(result.error || "Schema validation failed.");
      return;
    }

    // Success! Update active bundle
    onBundleUpdated(result.bundle);
    setSuccessMsg("Success: Results bundle strictly validated against schema!");
    setTimeout(() => {
      onClose();
    }, 1200);
  };

  const handleManualApply = () => {
    if (!jsonText.trim()) {
      setValidationError("Please paste results_bundle.json content first.");
      return;
    }
    validateAndApply(jsonText);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="bg-slate-950 px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-100">Upload or Inspect results_bundle.json</h3>
              <p className="text-xs text-slate-400">Strict schema validation enforced. Never falls back silently.</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-4 overflow-y-auto">
          {validationError && (
            <div className="bg-rose-950/50 border border-rose-800/80 rounded-xl p-3.5 text-xs text-rose-300 flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <strong className="block font-semibold">Strict Schema Error:</strong>
                <span className="font-mono text-[11px] break-all">{validationError}</span>
              </div>
            </div>
          )}

          {successMsg && (
            <div className="bg-emerald-950/50 border border-emerald-800/80 rounded-xl p-3 text-xs text-emerald-300 flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>{successMsg}</span>
            </div>
          )}

          {/* File Picker */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-2">
              Select results_bundle.json from your machine:
            </label>
            <input
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              className="block w-full text-xs text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 file:cursor-pointer cursor-pointer bg-slate-950 border border-slate-800 rounded-xl p-1"
            />
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="flex-1 border-b border-slate-800"></span>
            <span>OR PASTE JSON MANUALLY</span>
            <span className="flex-1 border-b border-slate-800"></span>
          </div>

          {/* Text Area */}
          <div>
            <textarea
              rows={8}
              value={jsonText}
              onChange={(e) => setJsonText(e.target.value)}
              placeholder="Paste full results_bundle.json content here..."
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 font-mono text-xs text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500 resize-none"
            />
          </div>
        </div>

        {/* Modal Footer */}
        <div className="bg-slate-950 px-6 py-4 border-t border-slate-800 flex items-center justify-between">
          <button
            onClick={() => {
              onResetToSample();
              onClose();
            }}
            className="text-xs text-slate-400 hover:text-amber-300 transition-colors flex items-center gap-1.5 cursor-pointer font-medium"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reset to Sample Bundle</span>
          </button>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="text-xs text-slate-300 hover:text-white px-4 py-2 rounded-xl hover:bg-slate-800 transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={handleManualApply}
              className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold px-4 py-2 rounded-xl shadow-lg shadow-indigo-600/30 transition-all cursor-pointer"
            >
              Validate &amp; Load Bundle
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
