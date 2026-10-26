import React, { useState } from 'react';
import { ResultsBundle, ConfusedPair, MisclassifiedExample, PerClassF1 } from '../types/bundle';
import { AlertTriangle, ArrowRight, Search, Filter, ShieldAlert, BarChart2, CheckCircle2, XCircle } from 'lucide-react';

interface ErrorAnalysisTabProps {
  bundle: ResultsBundle;
}

export const ErrorAnalysisTab: React.FC<ErrorAnalysisTabProps> = ({ bundle }) => {
  const [queryFilter, setQueryFilter] = useState('');
  const [f1Search, setF1Search] = useState('');
  const [f1Sort, setF1Sort] = useState<'asc' | 'desc'>('asc'); // default 'asc' so lowest F1 classes appear first!
  const [activeSubTab, setActiveSubTab] = useState<'pairs' | 'examples' | 'perclass'>('pairs');

  const { confused_pairs, misclassified_examples, per_class_f1 } = bundle.error_analysis;

  // Filtered misclassified examples
  const filteredExamples = misclassified_examples.filter(ex => 
    ex.text.toLowerCase().includes(queryFilter.toLowerCase()) ||
    ex.true_label.toLowerCase().includes(queryFilter.toLowerCase()) ||
    ex.predicted_label.toLowerCase().includes(queryFilter.toLowerCase())
  );

  // Filtered and sorted 77-class F1
  const sortedPerClass = [...per_class_f1]
    .filter(cls => cls.intent_name.toLowerCase().includes(f1Search.toLowerCase()) || cls.category.toLowerCase().includes(f1Search.toLowerCase()))
    .sort((a, b) => f1Sort === 'asc' ? a.f1_score - b.f1_score : b.f1_score - a.f1_score);

  return (
    <div className="space-y-8">
      {/* Header and Sub-Navigation */}
      <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
              <span>Comprehensive Error Analysis</span>
            </h2>
            <span className="text-[10px] bg-slate-800 text-slate-300 px-2.5 py-0.5 rounded-full border border-slate-700 font-mono">
              Best Model: {bundle.error_analysis.best_model_id}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Investigating semantic overlap, boundary ambiguity, and long-tail class weaknesses across the 77 intent taxonomy.
          </p>
        </div>

        {/* Sub-tab pills */}
        <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800 self-start md:self-auto text-xs">
          <button
            onClick={() => setActiveSubTab('pairs')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeSubTab === 'pairs'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            15 Most Confused Pairs
          </button>
          <button
            onClick={() => setActiveSubTab('examples')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeSubTab === 'examples'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            20 Misclassified Queries
          </button>
          <button
            onClick={() => setActiveSubTab('perclass')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition-all cursor-pointer ${
              activeSubTab === 'perclass'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All 77 Classes F1 Explorer
          </button>
        </div>
      </div>

      {/* Sub-tab 1: 15 Confused Pairs */}
      {activeSubTab === 'pairs' && (
        <div className="space-y-4">
          <div className="text-xs text-slate-400 flex items-center justify-between">
            <span>Ranked by frequency of confusion (True vs. Predicted Intent)</span>
            <span className="text-[11px] font-mono text-slate-500">Top 15 Pairs Shown</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {confused_pairs.slice(0, 15).map((cp) => (
              <div
                key={cp.rank}
                className="bg-slate-900/60 border border-slate-800 hover:border-slate-700 rounded-xl p-4 transition-all space-y-3"
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold uppercase tracking-wider bg-slate-950 text-slate-400 border border-slate-800 px-2 py-0.5 rounded">
                    Rank #{cp.rank}
                  </span>
                  <div className="text-xs font-mono font-bold text-amber-400">
                    {cp.count} errors ({cp.percentage_of_class_errors}% of class)
                  </div>
                </div>

                {/* Intent pair visually */}
                <div className="space-y-1.5 text-xs">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0"></span>
                    <span className="text-slate-400 text-[10px] uppercase font-bold">True:</span>
                    <code className="text-emerald-300 font-mono text-[11px] truncate">{cp.true_intent}</code>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-rose-400 shrink-0"></span>
                    <span className="text-slate-400 text-[10px] uppercase font-bold">Pred:</span>
                    <code className="text-rose-300 font-mono text-[11px] truncate">{cp.predicted_intent}</code>
                  </div>
                </div>

                {/* Banking root cause explanation */}
                <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-2.5 text-[11px] text-slate-300 leading-relaxed">
                  {cp.explanation}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sub-tab 2: 20 Misclassified Examples */}
      {activeSubTab === 'examples' && (
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="text-xs text-slate-400">
              Concrete customer support messages misclassified during test evaluation ({filteredExamples.length} matches)
            </div>
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Filter by keyword or label..."
                value={queryFilter}
                onChange={(e) => setQueryFilter(e.target.value)}
                className="bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-64"
              />
            </div>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="bg-slate-950/80 border-b border-slate-800 text-slate-400 text-[11px] uppercase tracking-wider font-semibold">
                    <th className="py-3 px-4 w-12 text-center">#</th>
                    <th className="py-3 px-4 max-w-md">Customer Query</th>
                    <th className="py-3 px-4">Ground Truth</th>
                    <th className="py-3 px-4">Predicted Intent</th>
                    <th className="py-3 px-4 text-center">Confidence</th>
                    <th className="py-3 px-4">Error Pattern</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {filteredExamples.slice(0, 20).map((ex) => (
                    <tr key={ex.id} className="hover:bg-slate-900/50 transition-colors">
                      <td className="py-3 px-4 text-center text-slate-500 font-mono text-[11px]">{ex.id}</td>
                      <td className="py-3 px-4 text-slate-200 max-w-md italic">
                        "{ex.text}"
                        <span className="block text-[11px] text-slate-400 font-normal not-italic mt-1">
                          {ex.explanation}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono">
                        <span className="bg-emerald-950 text-emerald-300 border border-emerald-800 px-2 py-0.5 rounded text-[11px]">
                          {ex.true_label}
                        </span>
                      </td>
                      <td className="py-3 px-4 font-mono">
                        <span className="bg-rose-950 text-rose-300 border border-rose-800 px-2 py-0.5 rounded text-[11px]">
                          {ex.predicted_label}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center font-mono text-slate-300">
                        {(ex.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-[10px] uppercase font-semibold text-slate-400 bg-slate-950 border border-slate-800 px-2 py-0.5 rounded">
                          {ex.error_type.replace('_', ' ')}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Sub-tab 3: All 77 Classes F1 Explorer */}
      {activeSubTab === 'perclass' && (
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="text-xs text-slate-400">
              Inspect test metrics across all 77 Banking77 classes. Showing {sortedPerClass.length} classes.
            </div>

            <div className="flex items-center gap-2">
              <div className="relative">
                <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search class or domain..."
                  value={f1Search}
                  onChange={(e) => setF1Search(e.target.value)}
                  className="bg-slate-950 border border-slate-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-52"
                />
              </div>

              <button
                onClick={() => setF1Sort(s => s === 'asc' ? 'desc' : 'asc')}
                className="bg-slate-950 border border-slate-800 hover:bg-slate-800 px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-300 flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <Filter className="w-3.5 h-3.5 text-indigo-400" />
                <span>Sort F1: {f1Sort === 'asc' ? 'Lowest First (Challenging)' : 'Highest First'}</span>
              </button>
            </div>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl overflow-hidden">
            <div className="overflow-x-auto max-h-[600px] scrollbar-thin">
              <table className="w-full text-left text-xs border-collapse">
                <thead className="sticky top-0 bg-slate-950 border-b border-slate-800 z-10 text-slate-400 text-[11px] uppercase tracking-wider font-semibold">
                  <tr>
                    <th className="py-3 px-4 w-14 text-center">Rank</th>
                    <th className="py-3 px-4">Intent Label</th>
                    <th className="py-3 px-4">Category</th>
                    <th className="py-3 px-4 text-center">Test Count</th>
                    <th className="py-3 px-4 text-right">Precision</th>
                    <th className="py-3 px-4 text-right">Recall</th>
                    <th className="py-3 px-4 text-right">F1 Score</th>
                    <th className="py-3 px-4 w-32">F1 Bar</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {sortedPerClass.map((cls, idx) => {
                    const isLow = cls.f1_score < 0.85;
                    const isHigh = cls.f1_score >= 0.95;
                    return (
                      <tr key={cls.class_id} className={`hover:bg-slate-900/60 transition-colors ${isLow ? 'bg-rose-950/10' : ''}`}>
                        <td className="py-2.5 px-4 text-center text-slate-500 text-[11px]">#{cls.rank ?? (idx + 1)}</td>
                        <td className="py-2.5 px-4 font-semibold text-slate-200">
                          {cls.intent_name}
                        </td>
                        <td className="py-2.5 px-4 font-sans text-slate-400 text-[11px]">{cls.category}</td>
                        <td className="py-2.5 px-4 text-center text-slate-400">{cls.test_count}</td>
                        <td className="py-2.5 px-4 text-right text-slate-300">{(cls.precision * 100).toFixed(1)}%</td>
                        <td className="py-2.5 px-4 text-right text-slate-300">{(cls.recall * 100).toFixed(1)}%</td>
                        <td className="py-2.5 px-4 text-right font-bold">
                          <span className={isLow ? 'text-rose-400' : isHigh ? 'text-emerald-400' : 'text-slate-100'}>
                            {(cls.f1_score * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-2.5 px-4">
                          <div className="h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                            <div
                              className={`h-full rounded-full ${isLow ? 'bg-rose-500' : isHigh ? 'bg-emerald-500' : 'bg-indigo-500'}`}
                              style={{ width: `${cls.f1_score * 100}%` }}
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
