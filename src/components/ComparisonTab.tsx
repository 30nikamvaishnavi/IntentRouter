import React, { useState } from 'react';
import { ResultsBundle, ModelMetrics, CascadeStep } from '../types/bundle';
import { 
  CheckCircle2, AlertCircle, TrendingUp, Cpu, DollarSign, 
  Clock, HardDrive, BarChart3, HelpCircle, Sliders, ShieldCheck, Sparkles, Layers
} from 'lucide-react';

interface ComparisonTabProps {
  bundle: ResultsBundle;
}

export const ComparisonTab: React.FC<ComparisonTabProps> = ({ bundle }) => {
  const [useSubset, setUseSubset] = useState<boolean>(true);
  const models = bundle.models;
  const cascade = bundle.cascade;

  // Threshold slider state (default to bundle's selected_threshold or 0.80)
  const [sliderIndex, setSliderIndex] = useState<number>(() => {
    if (!cascade || !cascade.threshold_steps) return 10;
    const matchIdx = cascade.threshold_steps.findIndex(
      s => Math.abs(s.threshold - (cascade.selected_threshold || 0.80)) < 0.01
    );
    return matchIdx >= 0 ? matchIdx : 10;
  });

  const currentStep: CascadeStep | undefined = cascade?.threshold_steps?.[sliderIndex];

  return (
    <div className="space-y-8">
      {/* Fair Comparison Toggle Bar */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-slate-100">Comparative Model Benchmarks</h2>
            <span className="text-[10px] uppercase font-semibold bg-indigo-950 text-indigo-300 border border-indigo-800 px-2 py-0.5 rounded-full">
              Baseline: {bundle.metadata.gemini_model || 'Gemini 2.5 Flash'}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Toggle between the Gemini rate-limited fair subset (308 samples, 4 per class) and the full test set (3,080 samples).
          </p>
        </div>

        {/* Toggle switch */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 self-start md:self-auto">
          <button
            onClick={() => setUseSubset(true)}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              useSubset
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Fair Subset (308 Samples • All Models)
          </button>
          <button
            onClick={() => setUseSubset(false)}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
              !useSubset
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Full Test Set (3,080 Samples)
          </button>
        </div>
      </div>

      {/* Main Benchmarks Table */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-950/80 border-b border-slate-800 text-slate-400 uppercase font-semibold text-[11px] tracking-wider">
                <th className="py-3.5 px-4">Model Architecture</th>
                <th className="py-3.5 px-4">Type</th>
                <th className="py-3.5 px-4 text-right">
                  Accuracy ({useSubset ? '308 Sub' : 'Full'})
                </th>
                <th className="py-3.5 px-4 text-right">Macro-F1</th>
                <th className="py-3.5 px-4 text-right">Weighted-F1</th>
                <th className="py-3.5 px-4 text-right">Top-3 Acc</th>
                <th className="py-3.5 px-4 text-right">CPU Latency</th>
                <th className="py-3.5 px-4 text-right">Disk Size</th>
                <th className="py-3.5 px-4 text-right">Est. Cost / 1k</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {models.map((m) => {
                const acc = useSubset ? m.accuracy_subset_308 : m.accuracy_full;
                const f1 = useSubset ? m.macro_f1_subset_308 : m.macro_f1_full;
                const wf1 = useSubset ? m.weighted_f1_subset_308 : m.weighted_f1_full;
                const top3 = useSubset ? m.top3_accuracy_subset_308 : m.top3_accuracy_full;

                return (
                  <tr
                    key={m.id}
                    className={`transition-colors ${
                      m.is_baseline
                        ? 'bg-indigo-950/25 border-l-4 border-l-indigo-500 hover:bg-indigo-950/40'
                        : 'hover:bg-slate-900/70'
                    }`}
                  >
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-slate-100">{m.name}</span>
                        {m.is_baseline && (
                          <span className="bg-indigo-900/80 text-indigo-300 text-[10px] font-bold px-2 py-0.5 rounded border border-indigo-700">
                            Baseline
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5 max-w-sm line-clamp-1">{m.notes}</div>
                    </td>

                    <td className="py-3.5 px-4">
                      <span className="inline-block px-2 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide border border-slate-800 bg-slate-950 text-slate-300">
                        {m.category}
                      </span>
                    </td>

                    {/* Accuracy */}
                    <td className="py-3.5 px-4 text-right">
                      <div className="font-mono font-bold text-slate-100">
                        {acc !== null ? `${(acc * 100).toFixed(2)}%` : '---'}
                      </div>
                      <div className="w-16 h-1.5 bg-slate-800 rounded-full ml-auto mt-1 overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 rounded-full"
                          style={{ width: `${(acc || 0) * 100}%` }}
                        />
                      </div>
                    </td>

                    {/* Macro-F1 */}
                    <td className="py-3.5 px-4 text-right font-mono text-slate-200">
                      {f1 !== null ? `${(f1 * 100).toFixed(2)}%` : '---'}
                    </td>

                    {/* Weighted-F1 */}
                    <td className="py-3.5 px-4 text-right font-mono text-slate-300">
                      {wf1 !== null ? `${(wf1 * 100).toFixed(2)}%` : '---'}
                    </td>

                    {/* Top-3 */}
                    <td className="py-3.5 px-4 text-right font-mono text-slate-300">
                      {top3 !== null ? (
                        <span className="text-emerald-400 font-semibold">{(top3 * 100).toFixed(2)}%</span>
                      ) : (
                        <span className="text-slate-600 text-[10px] italic">N/A</span>
                      )}
                    </td>

                    {/* Latency */}
                    <td className="py-3.5 px-4 text-right font-mono">
                      <span className={m.latency_cpu_ms < 20 ? 'text-emerald-400 font-semibold' : 'text-amber-400'}>
                        {m.latency_cpu_ms} ms
                      </span>
                      {m.latency_gpu_ms && (
                        <span className="block text-[10px] text-slate-500">GPU: {m.latency_gpu_ms}ms</span>
                      )}
                    </td>

                    {/* Size */}
                    <td className="py-3.5 px-4 text-right font-mono text-slate-300">
                      {m.model_size_mb > 0 ? `${m.model_size_mb} MB` : <span className="text-slate-500">API</span>}
                    </td>

                    {/* Cost / 1k */}
                    <td className="py-3.5 px-4 text-right font-mono">
                      <span className="text-slate-100 font-bold">
                        ${m.estimated_cost_per_1k.toFixed(4)}
                      </span>
                      <span className="block text-[10px] text-slate-500">
                        {m.cost_type === 'compute_estimate' ? 'Compute est.' : 'Tokens pricing'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* CASCADE SECTION - Requirement 4 */}
      {cascade && cascade.threshold_steps && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-6 shadow-xl">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center gap-2">
                <Sliders className="w-5 h-5 text-indigo-400" />
                <h3 className="text-base font-bold text-slate-100">
                  Cascade Routing Analysis (Small Model &amp; Gemini Fallback)
                </h3>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {cascade.note}
              </p>
            </div>
            <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-xl font-mono text-xs">
              <span className="text-slate-400">Model ECE (Expected Calibration Error):</span>
              <span className="text-emerald-400 font-bold">
                {(cascade.expected_calibration_error * 100).toFixed(2)}%
              </span>
            </div>
          </div>

          {/* Interactive Threshold Slider */}
          <div className="bg-slate-950/80 border border-slate-800/90 rounded-xl p-5 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <span className="text-xs font-bold text-slate-200 uppercase tracking-wide">
                  Confidence Threshold (&tau;)
                </span>
                <span className="text-xs text-slate-500 ml-2">
                  (Queries with confidence &ge; &tau; are handled locally; queries &lt; &tau; route to Gemini)
                </span>
              </div>
              <span className="font-mono text-base font-bold text-indigo-400 bg-indigo-950/60 border border-indigo-800 px-3 py-0.5 rounded-lg">
                &tau; = {currentStep?.threshold.toFixed(2)}
              </span>
            </div>

            <input
              type="range"
              min={0}
              max={cascade.threshold_steps.length - 1}
              step={1}
              value={sliderIndex}
              onChange={(e) => setSliderIndex(parseInt(e.target.value, 10))}
              className="w-full accent-indigo-500 cursor-pointer"
            />

            <div className="flex justify-between text-[10px] font-mono text-slate-500">
              <span>0.30 (Aggressive Local)</span>
              <span>0.65</span>
              <span>0.80 (Recommended)</span>
              <span>0.95 (Conservative Escalation)</span>
            </div>
          </div>

          {/* Cascade Step Metrics Cards */}
          {currentStep && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Local SLM Coverage</span>
                <div className="text-2xl font-black text-indigo-400 font-mono mt-1">
                  {(currentStep.coverage * 100).toFixed(1)}%
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  {currentStep.local_count_subset_308} of 308 queries handled locally
                </div>
              </div>

              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Local Accuracy on Handled</span>
                <div className="text-2xl font-black text-emerald-400 font-mono mt-1">
                  {(currentStep.local_accuracy * 100).toFixed(2)}%
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Accuracy on queries with P &ge; {currentStep.threshold.toFixed(2)}
                </div>
              </div>

              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Full Cascade Accuracy (308 Subset)</span>
                <div className="text-2xl font-black text-cyan-400 font-mono mt-1">
                  {(currentStep.cascade_accuracy_subset_308 * 100).toFixed(2)}%
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Includes Gemini fallback on remaining {currentStep.gemini_count_subset_308} queries
                </div>
              </div>

              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4">
                <span className="text-[10px] text-slate-500 uppercase font-bold block">Cascade Cost &amp; Latency</span>
                <div className="text-2xl font-black text-amber-400 font-mono mt-1">
                  ${currentStep.estimated_cost_per_1k.toFixed(4)}
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Avg Latency: <strong className="text-slate-200">{currentStep.average_latency_ms.toFixed(1)} ms</strong>
                </div>
              </div>
            </div>
          )}

          {/* Reliability Diagram & Calibration */}
          {cascade.reliability_bins && (
            <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                  <BarChart3 className="w-4 h-4 text-emerald-400" />
                  <span>Model Reliability Diagram (Calibration Curve)</span>
                </h4>
                <span className="text-[11px] font-mono text-slate-400">
                  Confidence Bins vs. Empirical Accuracy
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-5 lg:grid-cols-10 gap-2 pt-2">
                {cascade.reliability_bins.map((bin) => {
                  const diff = Math.abs(bin.accuracy - bin.mean_confidence);
                  const isWellCalibrated = diff < 0.05;
                  return (
                    <div key={bin.bin_index} className="bg-slate-900 border border-slate-800/80 rounded-lg p-2.5 space-y-1.5 text-center">
                      <span className="text-[10px] font-mono text-slate-500 block">
                        [{bin.bin_lower.toFixed(1)}-{bin.bin_upper.toFixed(1)}]
                      </span>
                      <div className="font-mono text-xs font-bold text-slate-100">
                        {(bin.accuracy * 100).toFixed(1)}%
                      </div>
                      <div className="text-[9px] text-slate-400 font-mono">
                        Conf: {(bin.mean_confidence * 100).toFixed(1)}%
                      </div>
                      <div className="h-1 bg-slate-950 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${isWellCalibrated ? 'bg-emerald-500' : 'bg-amber-500'}`}
                          style={{ width: `${bin.accuracy * 100}%` }}
                        />
                      </div>
                      <span className="text-[9px] text-slate-600 block">N={bin.sample_count}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
