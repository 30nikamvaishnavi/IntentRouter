import React, { useState, useEffect } from 'react';
import { ResultsBundle } from './types/bundle';
import { validateResultsBundle } from './utils/validateBundle';
import { Header } from './components/Header';
import { OverviewTab } from './components/OverviewTab';
import { ComparisonTab } from './components/ComparisonTab';
import { TrainingTab } from './components/TrainingTab';
import { ErrorAnalysisTab } from './components/ErrorAnalysisTab';
import { RobustnessTab } from './components/RobustnessTab';
import { TryItTab } from './components/TryItTab';
import { CodeViewerTab } from './components/CodeViewerTab';
import { UploadModal } from './components/UploadModal';
import { AlertCircle, RefreshCw, Terminal, ExternalLink, ShieldCheck } from 'lucide-react';

export default function App() {
  const [bundle, setBundle] = useState<ResultsBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [isUploadOpen, setIsUploadOpen] = useState<boolean>(false);

  // Load bundle from API / static file
  const loadBundle = async (url: string = '/results_bundle.json') => {
    setLoading(true);
    setError(null);
    try {
      // Try /api/bundle first, fallback to static /results_bundle.json
      let res = await fetch('/api/bundle');
      if (!res.ok) {
        res = await fetch(url);
      }
      if (!res.ok) {
        throw new Error(`Failed to load results bundle (${res.status} ${res.statusText})`);
      }

      const json = await res.json();
      const validation = validateResultsBundle(json);

      if (!validation.valid || !validation.bundle) {
        throw new Error(validation.error || 'Failed strict schema validation.');
      }

      setBundle(validation.bundle);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadBundle();
  }, []);

  const handleResetToSample = async () => {
    await loadBundle('/sample_results_bundle.json');
  };

  const handleBundleUpdated = (newBundle: ResultsBundle) => {
    setBundle(newBundle);
    setError(null);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center text-slate-400 gap-3">
        <RefreshCw className="w-8 h-8 animate-spin text-indigo-500" />
        <span className="text-xs font-mono">Loading &amp; strictly validating IntentRouter bundle...</span>
      </div>
    );
  }

  if (error || !bundle) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
        <div className="max-w-xl bg-slate-900 border border-rose-800/80 rounded-2xl p-6 text-slate-200 space-y-4 shadow-2xl">
          <div className="flex items-center gap-3 text-rose-400">
            <AlertCircle className="w-7 h-7 shrink-0" />
            <h2 className="text-base font-bold">Bundle Schema Validation Failed</h2>
          </div>
          <p className="text-xs text-slate-300">
            Per strict project guidelines, the results bundle failed schema validation and defaults will not be substituted.
          </p>
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs text-rose-300 break-all leading-relaxed">
            {error}
          </div>
          <div className="flex items-center gap-3 pt-2">
            <button
              onClick={() => handleResetToSample()}
              className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold px-4 py-2 rounded-xl transition-all cursor-pointer"
            >
              Load Valid Sample Bundle
            </button>
            <button
              onClick={() => setIsUploadOpen(true)}
              className="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold px-4 py-2 rounded-xl transition-all cursor-pointer"
            >
              Upload Another File
            </button>
          </div>
        </div>
        <UploadModal
          isOpen={isUploadOpen}
          onClose={() => setIsUploadOpen(false)}
          onBundleUpdated={handleBundleUpdated}
          onResetToSample={handleResetToSample}
          currentIsSample={false}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Header with Nav, Badges, and Sample Banner */}
      <Header
        bundle={bundle}
        onOpenUpload={() => setIsUploadOpen(true)}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8">
        {activeTab === 'overview' && (
          <OverviewTab
            bundle={bundle}
            onNavigateToTryIt={() => setActiveTab('tryit')}
          />
        )}
        {activeTab === 'comparison' && <ComparisonTab bundle={bundle} />}
        {activeTab === 'training' && <TrainingTab bundle={bundle} />}
        {activeTab === 'errors' && <ErrorAnalysisTab bundle={bundle} />}
        {activeTab === 'robustness' && <RobustnessTab bundle={bundle} />}
        {activeTab === 'tryit' && <TryItTab bundle={bundle} />}
        {activeTab === 'code' && <CodeViewerTab bundle={bundle} />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-200">IntentRouter Benchmarking Suite</span>
            <span className="text-slate-600">•</span>
            <span>PolyAI/banking77 Dataset (77 Classes)</span>
            <span className="text-slate-600">•</span>
            <span className="text-emerald-400 font-mono">MIT License</span>
          </div>

          <div className="flex items-center gap-4 text-[11px]">
            <a
              href="https://huggingface.co/datasets/PolyAI/banking77"
              target="_blank"
              rel="noreferrer"
              className="hover:text-indigo-400 transition-colors flex items-center gap-1"
            >
              <span>Banking77 on HF</span>
              <ExternalLink className="w-3 h-3" />
            </a>
            <button
              onClick={() => setActiveTab('code')}
              className="hover:text-indigo-400 transition-colors cursor-pointer"
            >
              Colab Quickstart
            </button>
            <button
              onClick={() => setIsUploadOpen(true)}
              className="hover:text-indigo-400 transition-colors cursor-pointer"
            >
              Upload Results Bundle
            </button>
          </div>
        </div>
      </footer>

      {/* Upload & Inspect Modal */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onBundleUpdated={handleBundleUpdated}
        onResetToSample={handleResetToSample}
        currentIsSample={bundle.is_sample_data}
      />
    </div>
  );
}
