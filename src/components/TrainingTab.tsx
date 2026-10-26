import React from 'react';
import { ResultsBundle } from '../types/bundle';
import { Activity, Cpu, HardDrive, Clock, CheckCircle2, Zap, Layers, Sparkles } from 'lucide-react';

interface TrainingTabProps {
  bundle: ResultsBundle;
}

export const TrainingTab: React.FC<TrainingTabProps> = ({ bundle }) => {
  const curves = bundle.training_curves;
  const loraModel = bundle.models.find(m => m.id === 'distilbert_lora');
  const fullModel = bundle.models.find(m => m.id === 'distilbert_full');

  // Compute parameter reduction dynamically
  const paramReduction = loraModel && fullModel && fullModel.total_params && loraModel.trainable_params
    ? ((1 - (loraModel.trainable_params / fullModel.total_params)) * 100).toFixed(2)
    : null;

  const vramSavings = loraModel && fullModel && fullModel.gpu_vram_peak_mb && loraModel.gpu_vram_peak_mb
    ? ((1 - (loraModel.gpu_vram_peak_mb / fullModel.gpu_vram_peak_mb)) * 100).toFixed(1)
    : null;

  return (
    <div className="space-y-8">
      {/* Overview Card */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <Activity className="w-5 h-5 text-indigo-400" />
              <span>Training Dynamics &amp; Parameter-Efficient Fine-Tuning (PEFT)</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Comparing convergence trajectories, loss curves, and computational resource consumption between Full Fine-Tuning and LoRA.
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1.5 text-cyan-400 font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
              {fullModel?.name || 'Full Fine-Tuning'}
            </span>
            <span className="flex items-center gap-1.5 text-purple-400 font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-purple-400"></span>
              {loraModel?.name || 'LoRA PEFT'}
            </span>
          </div>
        </div>
      </div>

      {/* PEFT Efficiency Head-to-Head Cards (All numbers read dynamically from bundle) */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <div className="text-[11px] font-bold text-slate-400 uppercase flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>Trainable Parameters</span>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black text-slate-100 font-mono">
              {loraModel?.trainable_params ? `${(loraModel.trainable_params / 1000).toFixed(1)} K` : '---'}
            </div>
            {paramReduction && (
              <div className="text-xs text-emerald-400 font-semibold mt-0.5">
                {paramReduction}% Parameter Reduction
              </div>
            )}
          </div>
          <div className="text-[11px] text-slate-500 mt-2">
            {fullModel?.total_params ? `${(fullModel.total_params / 1_000_000).toFixed(1)}M (Full)` : ''} vs. {loraModel?.trainable_params ? `${(loraModel.trainable_params / 1000).toFixed(1)}K (LoRA)` : ''}
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <div className="text-[11px] font-bold text-slate-400 uppercase flex items-center gap-1.5">
            <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
            <span>Disk / Artifact Size</span>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black text-slate-100 font-mono">
              {loraModel ? `${loraModel.model_size_mb} MB` : '---'}
            </div>
            {loraModel && fullModel && (
              <div className="text-xs text-emerald-400 font-semibold mt-0.5">
                {(fullModel.model_size_mb / loraModel.model_size_mb).toFixed(0)}x Smaller Checkpoint
              </div>
            )}
          </div>
          <div className="text-[11px] text-slate-500 mt-2">
            {fullModel?.model_size_mb} MB (Full) vs. {loraModel?.model_size_mb} MB (LoRA)
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <div className="text-[11px] font-bold text-slate-400 uppercase flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span>Peak GPU VRAM</span>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black text-slate-100 font-mono">
              {loraModel?.gpu_vram_peak_mb ? `${(loraModel.gpu_vram_peak_mb / 1024).toFixed(2)} GB` : '---'}
            </div>
            {vramSavings && (
              <div className="text-xs text-emerald-400 font-semibold mt-0.5">
                {vramSavings}% VRAM Savings
              </div>
            )}
          </div>
          <div className="text-[11px] text-slate-500 mt-2">
            {fullModel?.gpu_vram_peak_mb ? `${(fullModel.gpu_vram_peak_mb / 1024).toFixed(2)} GB (Full)` : ''} vs. {loraModel?.gpu_vram_peak_mb ? `${(loraModel.gpu_vram_peak_mb / 1024).toFixed(2)} GB (LoRA)` : ''}
          </div>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
          <div className="text-[11px] font-bold text-slate-400 uppercase flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-purple-400" />
            <span>Wall-Clock Training Time</span>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black text-slate-100 font-mono">
              {loraModel?.training_time_sec ? `${loraModel.training_time_sec} s` : '---'}
            </div>
            {loraModel?.training_time_sec && fullModel?.training_time_sec && (
              <div className="text-xs text-emerald-400 font-semibold mt-0.5">
                {((1 - (loraModel.training_time_sec / fullModel.training_time_sec)) * 100).toFixed(1)}% Faster Training
              </div>
            )}
          </div>
          <div className="text-[11px] text-slate-500 mt-2">
            Hardware: {bundle.metadata.hardware.gpu_name}
          </div>
        </div>
      </div>

      {/* Epoch Metrics Table & Progression */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Full Fine-Tuning Epoch Progression */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-cyan-300 flex items-center gap-2">
                <span>{fullModel?.name || 'DistilBERT: Full Fine-Tuning'}</span>
              </h3>
              <p className="text-[11px] text-slate-400">All layers trainable</p>
            </div>
            <span className="text-xs font-mono font-bold text-cyan-400">
              Val Acc: {(curves.distilbert_full[curves.distilbert_full.length - 1]?.val_accuracy * 100).toFixed(2)}%
            </span>
          </div>

          <div className="space-y-3">
            {curves.distilbert_full.map((ep) => (
              <div key={ep.epoch} className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3.5 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-slate-200">Epoch {ep.epoch}</span>
                  <div className="flex items-center gap-3 font-mono text-[11px]">
                    <span className="text-slate-400">Train Loss: <strong className="text-slate-200">{ep.train_loss.toFixed(4)}</strong></span>
                    <span className="text-slate-400">Val Loss: <strong className="text-slate-200">{ep.val_loss.toFixed(4)}</strong></span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 pt-1 text-xs font-mono">
                  <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 block uppercase">Validation Accuracy</span>
                    <span className="text-emerald-400 font-bold text-sm">{(ep.val_accuracy * 100).toFixed(2)}%</span>
                  </div>
                  <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 block uppercase">Validation Macro-F1</span>
                    <span className="text-cyan-400 font-bold text-sm">{(ep.val_macro_f1 * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* LoRA PEFT Progression */}
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-purple-300 flex items-center gap-2">
                <span>{loraModel?.name || 'DistilBERT: LoRA PEFT'}</span>
              </h3>
              <p className="text-[11px] text-slate-400">Freeze encoder, train attention projection adapters</p>
            </div>
            <span className="text-xs font-mono font-bold text-purple-400">
              Val Acc: {(curves.distilbert_lora[curves.distilbert_lora.length - 1]?.val_accuracy * 100).toFixed(2)}%
            </span>
          </div>

          <div className="space-y-3">
            {curves.distilbert_lora.map((ep) => (
              <div key={ep.epoch} className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3.5 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-slate-200">Epoch {ep.epoch}</span>
                  <div className="flex items-center gap-3 font-mono text-[11px]">
                    <span className="text-slate-400">Train Loss: <strong className="text-slate-200">{ep.train_loss.toFixed(4)}</strong></span>
                    <span className="text-slate-400">Val Loss: <strong className="text-slate-200">{ep.val_loss.toFixed(4)}</strong></span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 pt-1 text-xs font-mono">
                  <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 block uppercase">Validation Accuracy</span>
                    <span className="text-emerald-400 font-bold text-sm">{(ep.val_accuracy * 100).toFixed(2)}%</span>
                  </div>
                  <div className="bg-slate-900/80 p-2 rounded border border-slate-800">
                    <span className="text-[10px] text-slate-500 block uppercase">Validation Macro-F1</span>
                    <span className="text-purple-400 font-bold text-sm">{(ep.val_macro_f1 * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
