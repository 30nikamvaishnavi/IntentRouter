import React, { useState, useEffect } from 'react';
import { ResultsBundle } from '../types/bundle';
import { 
  Compass, Zap, DollarSign, Lock, Server, ArrowRight, 
  Search, CreditCard, RefreshCw, AlertOctagon, HelpCircle, 
  Banknote, ShieldAlert, UserCheck, Layers, AlertCircle
} from 'lucide-react';

interface OverviewTabProps {
  bundle: ResultsBundle;
  onNavigateToTryIt: () => void;
}

interface IntentItem {
  id: number;
  name: string;
  category: string;
  examples: string[];
}

export const OverviewTab: React.FC<OverviewTabProps> = ({ bundle, onNavigateToTryIt }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [intentsList, setIntentsList] = useState<IntentItem[]>([]);
  const [isLoadingTaxonomy, setIsLoadingTaxonomy] = useState(true);

  // Load all 77 intents and 2 real training utterances from label_names.json & examples.json
  useEffect(() => {
    async function loadRealData() {
      try {
        const [labelsRes, examplesRes] = await Promise.all([
          fetch('/label_names.json'),
          fetch('/examples.json')
        ]);

        if (labelsRes.ok && examplesRes.ok) {
          const labelsData: Record<string, string> = await labelsRes.json();
          const examplesData: Record<string, string[]> = await examplesRes.json();

          const items: IntentItem[] = Object.entries(labelsData).map(([idStr, name]) => {
            const id = parseInt(idStr, 10);
            return {
              id,
              name,
              category: inferCategory(name),
              examples: examplesData[name] || []
            };
          });

          // Sort by ID 0 to 76
          items.sort((a, b) => a.id - b.id);
          setIntentsList(items);
        }
      } catch (err) {
        console.warn('Failed to load external taxonomy files:', err);
      } finally {
        setIsLoadingTaxonomy(false);
      }
    }
    loadRealData();
  }, []);

  function inferCategory(name: string): string {
    const n = name.toLowerCase();
    if (n.includes('card') || n.includes('contactless') || n.includes('pin') || n.includes('visa')) return 'Cards & Credentials';
    if (n.includes('transfer') || n.includes('beneficiary') || n.includes('receiving_money') || n.includes('timing')) return 'Transfers & Clearing';
    if (n.includes('atm') || n.includes('cash') || n.includes('swallowed')) return 'Cash & ATM Operations';
    if (n.includes('fee') || n.includes('charge') || n.includes('exchange') || n.includes('rate')) return 'Fees & FX';
    if (n.includes('top_up') || n.includes('topping')) return 'Top-Up & Deposits';
    if (n.includes('compromised') || n.includes('stolen') || n.includes('passcode') || n.includes('not_recognised')) return 'Security & Fraud';
    return 'Account, KYC & Payments';
  }

  // Filter intents
  const filteredIntents = intentsList.filter(item => {
    const matchesSearch = item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.examples.some(ex => ex.toLowerCase().includes(searchQuery.toLowerCase()));
    const matchesCategory = selectedCategory === 'all' || item.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const categories = Array.from(new Set(intentsList.map(i => i.category)));

  // Identify best models dynamically from bundle without hardcoding
  const bestAccModel = [...bundle.models].sort((a, b) => b.accuracy_full - a.accuracy_full)[0];
  const fastestModel = [...bundle.models].sort((a, b) => a.latency_cpu_ms - b.latency_cpu_ms)[0];
  const lowestCostModel = [...bundle.models].sort((a, b) => a.estimated_cost_per_1k - b.estimated_cost_per_1k)[0];
  const quantizedModel = bundle.models.find(m => m.category === 'quantized');

  return (
    <div className="space-y-8">
      {/* Executive Problem Statement */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-900/90 to-indigo-950/40 border border-slate-800 rounded-2xl p-6 md:p-8 relative overflow-hidden shadow-xl">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 text-xs font-semibold">
            <Zap className="w-3.5 h-3.5" />
            <span>Banking Intent Classification Suite</span>
          </div>
          <h2 className="text-2xl md:text-3xl font-extrabold text-white tracking-tight leading-tight">
            IntentRouter: Banking Intent Classification &amp; Evaluation
          </h2>
          <p className="text-sm md:text-base text-slate-300 leading-relaxed">
            Compare the trade-offs between accuracy, latency, cost and privacy using the measured results below.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <button
              onClick={onNavigateToTryIt}
              className="inline-flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-indigo-600/30 transition-all cursor-pointer"
            >
              <span>Test Live Router in Playground</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <a
              href="#architecture"
              className="inline-flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold px-4 py-2.5 rounded-lg border border-slate-700 transition-colors"
            >
              <span>View System Architecture</span>
            </a>
          </div>
        </div>

        {/* 4 Pillars Card Grid - strictly follows:
            "the Overview must not state any latency, cost, accuracy or comparison until a real bundle is loaded; before that show 'Load a real run to see results'." */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-8 pt-6 border-t border-slate-800">
          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/15 text-emerald-400 flex items-center justify-center mb-2.5">
              <Zap className="w-4 h-4" />
            </div>
            <div className="text-sm font-bold text-slate-100">
              {bundle.is_sample_data ? (
                <span className="text-amber-400 text-xs italic">Load a real run to see results</span>
              ) : (
                `${fastestModel?.latency_cpu_ms} ms (${fastestModel?.name})`
              )}
            </div>
            <div className="text-xs text-slate-400 mt-1">CPU Latency</div>
            <p className="text-[11px] text-slate-500 mt-2">
              Evaluated on per-sample query routing without external network calls.
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/15 text-cyan-400 flex items-center justify-center mb-2.5">
              <DollarSign className="w-4 h-4" />
            </div>
            <div className="text-sm font-bold text-slate-100">
              {bundle.is_sample_data ? (
                <span className="text-amber-400 text-xs italic">Load a real run to see results</span>
              ) : (
                `$${lowestCostModel?.estimated_cost_per_1k.toFixed(4)} / 1k queries`
              )}
            </div>
            <div className="text-xs text-slate-400 mt-1">Inference Cost</div>
            <p className="text-[11px] text-slate-500 mt-2">
              Measured compute-time estimate vs. token pricing for Gemini API.
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/15 text-indigo-400 flex items-center justify-center mb-2.5">
              <Lock className="w-4 h-4" />
            </div>
            <div className="text-sm font-bold text-slate-100">
              On-Premises Privacy
            </div>
            <div className="text-xs text-slate-400 mt-1">Data Boundary</div>
            <p className="text-[11px] text-slate-500 mt-2">
              Local SLM models process messages inside customer infrastructure without external API data transmission.
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
            <div className="w-8 h-8 rounded-lg bg-amber-500/15 text-amber-400 flex items-center justify-center mb-2.5">
              <Server className="w-4 h-4" />
            </div>
            <div className="text-sm font-bold text-slate-100">
              {bundle.is_sample_data ? (
                <span className="text-amber-400 text-xs italic">Load a real run to see results</span>
              ) : (
                `${(bestAccModel?.accuracy_full * 100).toFixed(2)}% (${bestAccModel?.name})`
              )}
            </div>
            <div className="text-xs text-slate-400 mt-1">Test Set Accuracy</div>
            <p className="text-[11px] text-slate-500 mt-2">
              Evaluated on official held-out test splits across all 77 banking classes.
            </p>
          </div>
        </div>
      </div>

      {/* Architecture Section */}
      <div id="architecture" className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
          <div>
            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <Layers className="w-5 h-5 text-indigo-400" />
              <span>Routing Pipeline &amp; Cascade Architecture</span>
            </h3>
            <p className="text-xs text-slate-400">
              Cascade triage architecture combining edge SLM inference with cloud API fallback
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400 bg-slate-950 border border-slate-800 px-3 py-1 rounded-md">
            Baseline: {bundle.metadata.gemini_model || 'Gemini 2.5 Flash'}
          </span>
        </div>

        {/* Visual Diagram */}
        <div className="bg-slate-950 border border-slate-800/90 rounded-xl p-6 overflow-x-auto">
          <div className="min-w-[700px] flex items-center justify-between gap-4">
            <div className="w-48 bg-slate-900 border border-slate-700/80 rounded-lg p-3 text-center shrink-0">
              <div className="text-[10px] font-semibold text-cyan-400 uppercase tracking-wider mb-1">Customer Input</div>
              <div className="text-xs font-medium text-slate-200">Customer Support Message</div>
              <div className="text-[10px] text-slate-500 mt-1">Incoming Query Stream</div>
            </div>

            <ArrowRight className="w-5 h-5 text-slate-600 shrink-0" />

            <div className="w-56 bg-indigo-950/60 border border-indigo-700/60 rounded-lg p-3.5 text-center shrink-0 shadow-lg shadow-indigo-950/40">
              <div className="text-[10px] font-semibold text-indigo-300 uppercase tracking-wider mb-1 flex items-center justify-center gap-1">
                <Zap className="w-3 h-3 text-amber-400" />
                <span>Small Language Model</span>
              </div>
              <div className="text-xs font-bold text-white">Fine-Tuned DistilBERT / LoRA</div>
              <div className="text-[10px] text-indigo-300/80 mt-1">Local CPU / GPU Inference</div>
            </div>

            <ArrowRight className="w-5 h-5 text-slate-600 shrink-0" />

            <div className="w-48 bg-slate-900 border border-slate-700/80 rounded-lg p-3 text-center shrink-0">
              <div className="text-[10px] font-semibold text-purple-400 uppercase tracking-wider mb-1">Confidence Filter</div>
              <div className="text-xs font-bold text-slate-200">Softmax Confidence</div>
              <div className="text-[10px] text-slate-400 mt-1">P(intent) &ge; Threshold &tau;</div>
            </div>

            <ArrowRight className="w-5 h-5 text-slate-600 shrink-0" />

            <div className="space-y-2 shrink-0 w-60">
              <div className="bg-emerald-950/40 border border-emerald-700/50 rounded-lg p-2.5 text-left">
                <div className="text-[10px] font-bold text-emerald-400 uppercase">High Confidence (&ge; &tau;)</div>
                <div className="text-xs font-medium text-slate-200">Local Routing Decision</div>
                <div className="text-[10px] text-emerald-300/70">Immediate Dispatch / Action</div>
              </div>
              <div className="bg-amber-950/40 border border-amber-700/50 rounded-lg p-2.5 text-left">
                <div className="text-[10px] font-bold text-amber-400 uppercase">Low Confidence (&lt; &tau;)</div>
                <div className="text-xs font-medium text-slate-200">{bundle.metadata.gemini_model || 'Gemini'} / Human Escalation</div>
                <div className="text-[10px] text-amber-300/70">Ambiguity Disambiguation</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Real Taxonomy Explorer from label_names.json and examples.json */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <Compass className="w-5 h-5 text-cyan-400" />
              <span>Banking77 Intent Taxonomy ({intentsList.length} Unique Classes)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Exact intent labels loaded from <code className="text-cyan-300">results/label_names.json</code> with real training utterances from <code className="text-cyan-300">results/examples.json</code>.
            </p>
            <div className="text-[11px] text-amber-400/90 italic mt-1 flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              <span>Illustrative grouping by the author, not part of the dataset.</span>
            </div>
          </div>

          {/* Search and Category Filter */}
          <div className="flex flex-wrap items-center gap-2.5">
            <div className="relative">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search all 77 intents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-56 sm:w-64"
              />
            </div>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              <option value="all">All Groups ({categories.length})</option>
              {categories.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        {/* 77 Classes Grid */}
        {isLoadingTaxonomy ? (
          <div className="py-12 text-center text-xs text-slate-500">Loading Banking77 taxonomy...</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {filteredIntents.map((item) => (
              <div
                key={item.id}
                className="bg-slate-950/70 border border-slate-800/80 hover:border-slate-700 rounded-xl p-3.5 transition-all space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-indigo-300 break-all">
                    {item.name}
                  </span>
                  <span className="text-[10px] font-mono text-slate-500 shrink-0 ml-2">
                    ID #{item.id}
                  </span>
                </div>

                <div className="text-[10px] font-semibold text-slate-400">
                  {item.category}
                </div>

                {/* 2 Real Utterances */}
                <div className="space-y-1 pt-1 border-t border-slate-800/60">
                  <span className="text-[9px] uppercase font-bold text-slate-500 block">Real Training Examples:</span>
                  {item.examples.map((ex, exIdx) => (
                    <div key={exIdx} className="text-[11px] text-slate-300 bg-slate-900/60 p-1.5 rounded border border-slate-800/40 italic">
                      "{ex}"
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
