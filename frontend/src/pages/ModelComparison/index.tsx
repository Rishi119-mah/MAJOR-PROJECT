import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { getModelMetrics } from '../../services/api';
import { ModelMetricRecord } from '../../types';
import {
  BarChart3,
  Filter,
  Layers,
  Award,
  Search,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis,
} from 'recharts';

export const ModelComparisonPage: React.FC = () => {
  const { t, mode } = useApp();
  const [metrics, setMetrics] = useState<ModelMetricRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [featureFilter, setFeatureFilter] = useState<string>('All');
  const [protocolFilter, setProtocolFilter] = useState<string>('All');
  const [search, setSearch] = useState<string>('');
  const [selectedMetricForMatrix, setSelectedMetricForMatrix] = useState<ModelMetricRecord | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      setLoading(true);
      try {
        const data = await getModelMetrics(undefined, undefined, mode);
        setMetrics(data);
        if (data.length > 0) {
          setSelectedMetricForMatrix(data[0]);
        }
      } catch (e) {
        console.warn('Failed to load model metrics:', e);
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, [mode]);

  // Filtered metrics
  const filtered = metrics.filter((m) => {
    const matchFeat = featureFilter === 'All' || m.feature_group?.toLowerCase().includes(featureFilter.toLowerCase());
    const matchProto =
      protocolFilter === 'All' ||
      (m.evaluation_type && m.evaluation_type.toLowerCase().includes(protocolFilter.toLowerCase())) ||
      (m.evaluation_protocol && m.evaluation_protocol.toLowerCase().includes(protocolFilter.toLowerCase()));
    const matchSearch =
      !search ||
      m.model?.toLowerCase().includes(search.toLowerCase()) ||
      m.feature_group?.toLowerCase().includes(search.toLowerCase()) ||
      m.experiment_id?.toLowerCase().includes(search.toLowerCase());
    return matchFeat && matchProto && matchSearch;
  });

  // Top 10 for bar chart
  const barChartData = filtered
    .filter((m) => m.fake_f1 !== undefined && m.fake_f1 !== null && m.fake_f1 > 0)
    .sort((a, b) => (b.fake_f1 || 0) - (a.fake_f1 || 0))
    .slice(0, 10)
    .map((m) => ({
      name: `${m.model} (${m.feature_group})`,
      fake_f1: m.fake_f1 !== null && m.fake_f1 !== undefined ? Number(m.fake_f1.toFixed(4)) : null,
      roc_auc: m.roc_auc !== null && m.roc_auc !== undefined ? Number(m.roc_auc.toFixed(4)) : null,
      precision: m.fake_precision !== null && m.fake_precision !== undefined ? Number(m.fake_precision.toFixed(4)) : null,
      recall: m.fake_recall !== null && m.fake_recall !== undefined ? Number(m.fake_recall.toFixed(4)) : null,
      protocol: m.evaluation_type || m.evaluation_protocol || 'Not recorded',
    }));

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Machine Learning & Graph Model Comparison</h2>
          <p className="text-xs text-slate-500 mt-1">
            Empirical benchmark leaderboard across 75+ evaluated pipelines, comparing feature groups, cross-validation runs, and holdout test splits.
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Feature Group Filter */}
          <select
            value={featureFilter}
            onChange={(e) => setFeatureFilter(e.target.value)}
            className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-slate-700 font-medium focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          >
            <option value="All">All Feature Groups</option>
            <option value="review_only">Review Behaviors (12)</option>
            <option value="network_only">Network Centrality (4)</option>
            <option value="review_network">Review + Network (16)</option>
            <option value="review_network_image">Multimodal All (28)</option>
            <option value="tuned">Tuned Models</option>
          </select>

          {/* Protocol Filter */}
          <select
            value={protocolFilter}
            onChange={(e) => setProtocolFilter(e.target.value)}
            className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-slate-700 font-medium focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          >
            <option value="All">All Protocols</option>
            <option value="Final independent test">Final Holdout Test (20%)</option>
            <option value="Development cross-validation">Dev 5x2 CV</option>
            <option value="Historical">Historical Paper Baselines</option>
          </select>

          {/* Search Box */}
          <div className="relative">
            <Search size={14} className="absolute left-2.5 top-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search model..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="text-xs pl-8 pr-3 py-1.5 bg-slate-50 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none w-36 sm:w-48"
            />
          </div>
        </div>
      </div>

      {/* Charts Grid: Bar Chart & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Bar Chart (8 cols) */}
        <div className="lg:col-span-8 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Fake-Class F1 & ROC-AUC Benchmarks</h3>
              <p className="text-xs text-slate-500">Top 10 models ranked by positive class fake-buyer detection F1 score</p>
            </div>
            <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-indigo-50 text-indigo-700">
              Positive Class = Fake
            </span>
          </div>

          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barChartData} margin={{ top: 10, right: 10, left: -10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis
                  dataKey="name"
                  angle={-20}
                  textAnchor="end"
                  interval={0}
                  tick={{ fontSize: 10, fill: '#64748b' }}
                />
                <YAxis domain={[0.6, 1.0]} tick={{ fontSize: 11, fill: '#64748b' }} />
                <RechartsTooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff' }}
                />
                <Legend verticalAlign="top" height={30} iconType="circle" />
                <Bar dataKey="fake_f1" name="Fake F1 Score" fill="#4f46e5" radius={[4, 4, 0, 0]} />
                <Bar dataKey="roc_auc" name="ROC-AUC" fill="#06b6d4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Confusion Matrix Visualizer (4 cols) */}
        <div className="lg:col-span-4 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900">Confusion Matrix Inspector</h3>
            <span className="text-[10px] font-mono bg-slate-100 text-slate-600 px-2 py-0.5 rounded font-semibold max-w-[200px] truncate">
              {selectedMetricForMatrix?.evaluation_protocol || selectedMetricForMatrix?.evaluation_type || 'Evaluation Split'}
            </span>
          </div>

          {selectedMetricForMatrix ? (
            <div className="space-y-3">
              <div className="text-xs">
                <span className="text-slate-500">Selected Model:</span>
                <span className="font-bold text-slate-900 font-mono block truncate">
                  {selectedMetricForMatrix.model} ({selectedMetricForMatrix.feature_group})
                </span>
              </div>

              {/* 2x2 Matrix Grid */}
              <div className="grid grid-cols-2 gap-2 text-center text-xs">
                {/* True Negative */}
                <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
                  <span className="text-[10px] uppercase font-bold text-emerald-700 block">True Neg (TN)</span>
                  <span className="text-xl font-bold font-mono text-emerald-900">
                    {selectedMetricForMatrix.tn !== undefined && selectedMetricForMatrix.tn !== null
                      ? selectedMetricForMatrix.tn
                      : '—'}
                  </span>
                  <span className="text-[10px] text-emerald-600 block mt-0.5">Correct Genuine</span>
                </div>

                {/* False Positive */}
                <div className="bg-rose-50 border border-rose-200 rounded-lg p-3">
                  <span className="text-[10px] uppercase font-bold text-rose-700 block">False Pos (FP)</span>
                  <span className="text-xl font-bold font-mono text-rose-900">
                    {selectedMetricForMatrix.fp !== undefined && selectedMetricForMatrix.fp !== null
                      ? selectedMetricForMatrix.fp
                      : '—'}
                  </span>
                  <span className="text-[10px] text-rose-600 block mt-0.5">Type I Error</span>
                </div>

                {/* False Negative */}
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                  <span className="text-[10px] uppercase font-bold text-amber-700 block">False Neg (FN)</span>
                  <span className="text-xl font-bold font-mono text-amber-900">
                    {selectedMetricForMatrix.fn !== undefined && selectedMetricForMatrix.fn !== null
                      ? selectedMetricForMatrix.fn
                      : '—'}
                  </span>
                  <span className="text-[10px] text-amber-600 block mt-0.5">Type II Error</span>
                </div>

                {/* True Positive */}
                <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3">
                  <span className="text-[10px] uppercase font-bold text-indigo-700 block">True Pos (TP)</span>
                  <span className="text-xl font-bold font-mono text-indigo-900">
                    {selectedMetricForMatrix.tp !== undefined && selectedMetricForMatrix.tp !== null
                      ? selectedMetricForMatrix.tp
                      : '—'}
                  </span>
                  <span className="text-[10px] text-indigo-600 block mt-0.5">Caught Fake Buyers</span>
                </div>
              </div>

              {/* Derived rates */}
              <div className="p-3 bg-slate-50 rounded-lg space-y-1.5 text-[11px] text-slate-600 border border-slate-100">
                <div className="flex justify-between">
                  <span>Sensitivity (Recall):</span>
                  <span className="font-mono font-bold text-slate-800">
                    {selectedMetricForMatrix.fake_recall !== undefined && selectedMetricForMatrix.fake_recall !== null
                      ? `${(selectedMetricForMatrix.fake_recall * 100).toFixed(1)}%`
                      : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Specificity:</span>
                  <span className="font-mono font-bold text-slate-800">
                    {selectedMetricForMatrix.specificity !== undefined && selectedMetricForMatrix.specificity !== null
                      ? `${(selectedMetricForMatrix.specificity * 100).toFixed(1)}%`
                      : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>False Positive Rate:</span>
                  <span className="font-mono font-bold text-rose-600">
                    {selectedMetricForMatrix.false_positive_rate !== undefined && selectedMetricForMatrix.false_positive_rate !== null
                      ? `${(selectedMetricForMatrix.false_positive_rate * 100).toFixed(1)}%`
                      : '—'}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-6 text-center text-xs text-slate-400">Click a model in the table to inspect matrix.</div>
          )}
        </div>
      </div>

      {/* Comprehensive Benchmark Leaderboard Table */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900">Comprehensive Research Leaderboard ({filtered.length} entries)</h3>
            <p className="text-xs text-slate-500">Click any row to inspect confusion matrix and detailed error rates</p>
          </div>
          <span className="text-xs text-slate-500">
            Showing {filtered.length} of {metrics.length} recorded experiments
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Algorithm</th>
                <th className="py-2.5 px-3">Modality / Feature Group</th>
                <th className="py-2.5 px-3">Evaluation Protocol</th>
                <th className="py-2.5 px-3 text-right">Accuracy</th>
                <th className="py-2.5 px-3 text-right">Fake Precision</th>
                <th className="py-2.5 px-3 text-right">Fake Recall</th>
                <th className="py-2.5 px-3 text-right text-indigo-600 font-bold">Fake F1</th>
                <th className="py-2.5 px-3 text-right">ROC-AUC</th>
                <th className="py-2.5 px-3 text-right">PR-AUC</th>
                <th className="py-2.5 px-3 text-right">Brier Loss</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {filtered.map((m, idx) => {
                const isSelected = selectedMetricForMatrix?.experiment_id === m.experiment_id && selectedMetricForMatrix?.model === m.model;
                return (
                  <tr
                    key={`${m.experiment_id}_${m.model}_${idx}`}
                    onClick={() => setSelectedMetricForMatrix(m)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-indigo-50/70 font-semibold' : 'hover:bg-slate-50/70'
                    }`}
                  >
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-900">{m.model}</td>
                    <td className="py-2.5 px-3 font-sans">
                      <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 text-[11px]">
                        {m.feature_group}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-sans text-slate-500 text-[11px]">
                      {m.evaluation_type || m.evaluation_protocol || 'Not recorded'}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-700">{m.accuracy !== null && m.accuracy !== undefined ? m.accuracy.toFixed(4) : '—'}</td>
                    <td className="py-2.5 px-3 text-right text-slate-700">{m.fake_precision !== null && m.fake_precision !== undefined ? m.fake_precision.toFixed(4) : '—'}</td>
                    <td className="py-2.5 px-3 text-right text-slate-700">{m.fake_recall !== null && m.fake_recall !== undefined ? m.fake_recall.toFixed(4) : '—'}</td>
                    <td className="py-2.5 px-3 text-right font-bold text-indigo-600">{m.fake_f1 !== null && m.fake_f1 !== undefined ? m.fake_f1.toFixed(4) : '—'}</td>
                    <td className="py-2.5 px-3 text-right text-slate-700">{m.roc_auc !== null && m.roc_auc !== undefined ? m.roc_auc.toFixed(4) : '—'}</td>
                    <td className="py-2.5 px-3 text-right text-slate-700">{m.pr_auc !== null && m.pr_auc !== undefined ? m.pr_auc.toFixed(4) : '—'}</td>
                    <td className="py-2.5 px-3 text-right text-slate-500">{m.brier_score !== null && m.brier_score !== undefined ? m.brier_score.toFixed(4) : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

