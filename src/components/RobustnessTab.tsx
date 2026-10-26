import React from 'react';
import { ResultsBundle } from '../types/bundle';
import { ShieldCheck, Cpu, HardDrive, Zap, AlertCircle, Sparkles, Layers, Sliders } from 'lucide-react';

interface RobustnessTabProps {
  bundle: ResultsBundle;
}

export const RobustnessTab: React.FC<RobustnessTabProps> = ({ bundle }) => {
  const { robustness, quantization } = bundle;
  const geminiModel = bundle.metadata.gemini_model || 'Gemini 2.5 Flash';

  return (
    <div className="space-y-8">
      {/* Robustness Section */}
      <div className="space-y-4">
        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-indigo-400" />
                <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                  <span>Robustness to Real-World Conversational Noise</span>
                </h2>
                <span className="text-[10px] uppercase font-bold bg-amber-950 text-amber-300 border border-amber-800 px-2 py-0.5 rounded-full">
                  Small Synthetic Test ({robustness.tests[0]?.samples_count || 308} samples)
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                {robustness.note}
              </p>
            </div>
          </div>
        </div>

        {/* 3 Perturbation Experiment Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {robustness.tests.map((test) => (
            <div
              key={test.perturbation_type}
              className="bg-slate-900/60 border border-slate-800 hover:border-slate-700 rounded-2xl p-5 space-y-4 transition-all"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200">
                  {test.perturbation_type}
                </span>
                <span className="text-[10px] font-mono text-slate-500">
                  N={test.samples_count}
                </span>
              </div>

              <p className="text-[11px] text-slate-400 leading-relaxed min-h-[36px]">
                {test.description}
              </p>

              {/* Accuracy Drop Head-to-Head */}
              <div className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-3.5 space-y-2.5">
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">DistilBERT Local</span>
                    <span className="font-mono text-xs font-bold text-slate-200">
                      {(test.perturbed_accuracy_distilbert * 100).toFixed(1)}%{' '}
                      <span className={test.delta_accuracy_distilbert < 0 ? 'text-rose-400' : 'text-emerald-400'}>
                        ({test.delta_accuracy_distilbert > 0 ? '+' : ''}{(test.delta_accuracy_distilbert * 100).toFixed(1)}%)
                      </span>
                    </span>
                  </div>
                  <div className="h-1.5 bg-slate-900 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 rounded-full"
                      style={{ width: `${test.perturbed_accuracy_distilbert * 100}%` }}
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">{geminiModel}</span>
                    <span className="font-mono text-xs font-bold text-slate-200">
                      {(test.perturbed_accuracy_gemini * 100).toFixed(1)}%{' '}
                      <span className={test.delta_accuracy_gemini < 0 ? 'text-rose-400' : 'text-emerald-400'}>
                        ({test.delta_accuracy_gemini > 0 ? '+' : ''}{(test.delta_accuracy_gemini * 100).toFixed(1)}%)
                      </span>
                    </span>
                  </div>
                  <div className="h-1.5 bg-slate-900 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-500 rounded-full"
                      style={{ width: `${test.perturbed_accuracy_gemini * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Sample Perturbation */}
              <div className="space-y-1.5 pt-1 text-[11px]">
                <div className="text-slate-500 text-[10px] uppercase font-bold">Query Transformation:</div>
                <div className="bg-slate-950 p-2 rounded border border-slate-800 text-slate-400 italic">
                  "{test.example_original}"
                </div>
                <div className="bg-slate-950 p-2 rounded border border-indigo-950 text-indigo-300 font-mono text-[10px]">
                  &rarr; "{test.example_perturbed}"
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* PyTorch Dynamic Quantization Deep Dive */}
      <div className="bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950/30 border border-slate-800 rounded-2xl p-6 md:p-8 space-y-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <Zap className="w-5 h-5 text-cyan-400" />
              <span>PyTorch Dynamic INT8 Quantization (CPU Inference Speedup)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Method: <code className="text-cyan-300 font-mono">{quantization.quantization_method}</code>
            </p>
          </div>
          <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-950/60 border border-emerald-800 px-3 py-1 rounded-lg self-start">
            {quantization.speedup_factor}x Latency Speedup
          </span>
        </div>

        {/* 4 Quantization Tiles */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4">
            <span className="text-[10px] text-slate-500 uppercase font-bold block">CPU Latency</span>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-2xl font-black text-emerald-400 font-mono">{quantization.latency_cpu_int8_ms} ms</span>
              <span className="text-xs line-through text-slate-500 font-mono">{quantization.latency_cpu_fp32_ms} ms</span>
            </div>
            <div className="text-xs text-slate-400 mt-1">{quantization.speedup_factor}x faster per query</div>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4">
            <span className="text-[10px] text-slate-500 uppercase font-bold block">Model File Size</span>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-2xl font-black text-cyan-400 font-mono">{quantization.size_int8_mb} MB</span>
              <span className="text-xs line-through text-slate-500 font-mono">{quantization.size_fp32_mb} MB</span>
            </div>
            <div className="text-xs text-slate-400 mt-1">{quantization.size_reduction_ratio}x disk reduction</div>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4">
            <span className="text-[10px] text-slate-500 uppercase font-bold block">Test Set Accuracy</span>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-2xl font-black text-slate-100 font-mono">{(quantization.accuracy_int8 * 100).toFixed(2)}%</span>
              <span className="text-xs text-slate-500 font-mono">FP32: {(quantization.accuracy_fp32 * 100).toFixed(2)}%</span>
            </div>
            <div className="text-xs text-rose-400 mt-1">Delta: {(quantization.accuracy_drop * 100).toFixed(2)}%</div>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4">
            <span className="text-[10px] text-slate-500 uppercase font-bold block">Quantized Linear Layers</span>
            <div className="flex items-baseline gap-2 mt-1">
              <span className="text-2xl font-black text-indigo-400 font-mono">qint8</span>
            </div>
            <div className="text-xs text-slate-400 mt-1">Dynamic weight scaling on CPU</div>
          </div>
        </div>

        {/* Quantization Note */}
        <div className="bg-slate-950/90 border border-slate-800/90 rounded-xl p-4 text-xs text-slate-300 leading-relaxed">
          <strong className="text-slate-100">Engineering Evaluation:</strong> In retail banking environments with high request frequency, running <strong>{quantization.quantized_model}</strong> achieves a measured latency of <strong>{quantization.latency_cpu_int8_ms} ms</strong> on CPU (a {quantization.speedup_factor}x speedup compared to FP32) while maintaining comparable intent accuracy.
        </div>
      </div>
    </div>
  );
};
