import React from 'react';
import { ResultsBundle } from '../types/bundle';
import { ShieldCheck, Cpu, Database, Calendar, AlertTriangle, Upload, Sparkles } from 'lucide-react';

interface HeaderProps {
  bundle: ResultsBundle;
  onOpenUpload: () => void;
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ bundle, onOpenUpload, activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'overview', label: 'Overview & Taxonomy' },
    { id: 'comparison', label: 'Model Comparison & Cascade' },
    { id: 'training', label: 'Training & PEFT' },
    { id: 'errors', label: 'Error Analysis' },
    { id: 'robustness', label: 'Robustness & Quant' },
    { id: 'tryit', label: 'Try It (Live Router)' },
    { id: 'code', label: 'Notebook & Scripts' },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-40">
      {/* Sample Data Banner - Shown ONLY in sample mode */}
      {bundle.is_sample_data && (
        <div className="bg-amber-500/15 border-b border-amber-500/30 px-4 py-2 text-xs font-medium text-amber-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              <strong>Sample (illustrative).</strong> Load a real run bundle to inspect actual training artifacts and benchmark logs.
            </span>
          </div>
          <button
            onClick={onOpenUpload}
            className="underline hover:text-amber-100 flex items-center gap-1 cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5" />
            Upload Real Bundle
          </button>
        </div>
      )}

      {/* Main Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-white font-bold text-lg">
              IR
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold text-slate-100 tracking-tight">IntentRouter</h1>
                <span className="text-[10px] uppercase tracking-wider font-semibold bg-indigo-950 text-indigo-400 border border-indigo-800/60 px-2 py-0.5 rounded-full">
                  Banking77 SLM Suite
                </span>
                {bundle.metadata.gemini_model && (
                  <span className="text-[10px] font-mono font-medium text-amber-300 bg-amber-950/80 border border-amber-800/60 px-2 py-0.5 rounded-full flex items-center gap-1">
                    <Sparkles className="w-2.5 h-2.5" />
                    Baseline: {bundle.metadata.gemini_model}
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400">
                Compare the trade-offs between accuracy, latency, cost and privacy using the measured results below.
              </p>
            </div>
          </div>

          {/* Metadata badges strictly following rules:
              In sample mode the header shows only "Sample (illustrative)", no run date and no GPU.
              In real mode the run date, GPU, library versions and model names come only from bundle.metadata. */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {bundle.is_sample_data ? (
              <div className="flex items-center gap-1.5 bg-amber-950/40 border border-amber-800/60 px-3 py-1.5 rounded-lg text-amber-300 font-semibold">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Sample (illustrative)</span>
              </div>
            ) : (
              <>
                <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 px-2.5 py-1.5 rounded-lg text-slate-300">
                  <Calendar className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Run: <strong className="text-slate-100">{bundle.metadata.run_date}</strong></span>
                </div>

                <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 px-2.5 py-1.5 rounded-lg text-slate-300">
                  <Cpu className="w-3.5 h-3.5 text-indigo-400" />
                  <span>{bundle.metadata.hardware.gpu_name}</span>
                </div>
              </>
            )}

            <div className="flex items-center gap-1.5 bg-slate-900 border border-slate-800 px-2.5 py-1.5 rounded-lg text-slate-300">
              <Database className="w-3.5 h-3.5 text-emerald-400" />
              <span>{bundle.metadata.dataset.dataset_name} ({bundle.metadata.dataset.num_classes} intents)</span>
            </div>

            <button
              onClick={onOpenUpload}
              className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-3 py-1.5 rounded-lg transition-colors cursor-pointer font-medium"
            >
              <Upload className="w-3.5 h-3.5 text-indigo-300" />
              <span>Bundle JSON</span>
            </button>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex space-x-1 mt-4 overflow-x-auto scrollbar-none border-t border-slate-800/80 pt-2">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-all cursor-pointer whitespace-nowrap ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                }`}
              >
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
