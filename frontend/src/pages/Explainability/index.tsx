import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { getGlobalShap } from '../../services/api';
import { GlobalShapResponse, ShapFeatureItem } from '../../types';
import {
  Sparkles,
  Network,
  Image as ImageIcon,
  MessageSquare,
  Info,
  CheckCircle2,
  TrendingUp,
  Sliders,
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
  PieChart,
  Pie,
  Cell,
} from 'recharts';

export const ExplainabilityPage: React.FC = () => {
  const { t, mode } = useApp();
  const [shapData, setShapData] = useState<GlobalShapResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedModality, setSelectedModality] = useState<string>('all');

  useEffect(() => {
    const fetchShap = async () => {
      setLoading(true);
      try {
        const data = await getGlobalShap(mode);
        setShapData(data);
      } catch (e) {
        console.warn('Failed to load global SHAP:', e);
      } finally {
        setLoading(false);
      }
    };
    fetchShap();
  }, [mode]);

  const features = shapData?.features || [];
  const filteredFeatures = selectedModality === 'all'
    ? features
    : features.filter((f) => f.modality === selectedModality);

  const top12Features = filteredFeatures.slice(0, 12);

  // Dynamic Modality Weights directly from backend TreeSHAP analysis
  const netWeight = shapData?.modality_breakdown?.network ?? null;
  const revWeight = shapData?.modality_breakdown?.review ?? null;
  const imgWeight = shapData?.modality_breakdown?.image ?? null;

  const modalityPie = (netWeight !== null && revWeight !== null && imgWeight !== null) ? [
    { name: 'Review Behaviors', value: revWeight, color: '#0ea5e9' },
    { name: 'Network Centrality', value: netWeight, color: '#4f46e5' },
    { name: 'Image Similarity', value: imgWeight, color: '#10b981' },
  ] : [];

  const peakFeatureName = shapData?.top_feature || (top12Features[0]?.feature || '—');
  const peakFeaturePct = top12Features[0]?.percent_importance !== undefined
    ? top12Features[0].percent_importance.toFixed(2)
    : '—';

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">TreeSHAP Explainability & Global Attribution</h2>
          <p className="text-xs text-slate-500 mt-1">
            Interprets global and local feature importance via Shapley additive explanations on the final validated multimodal XGBoost pipeline.
          </p>
        </div>

        {/* Modality Filter */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-semibold text-slate-600 uppercase">Modality:</label>
          <select
            value={selectedModality}
            onChange={(e) => setSelectedModality(e.target.value)}
            className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-slate-800 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="all">All Modalities (28 Features)</option>
            <option value="network">Network Centrality Features</option>
            <option value="review">Review Behavioral Features</option>
            <option value="image">Image Similarity Features</option>
          </select>
        </div>
      </div>

      {/* Causal Distinction Disclaimer */}
      <div className="bg-indigo-50/70 border border-indigo-200 p-4 rounded-xl flex items-start gap-3 text-xs text-indigo-900">
        <Info size={18} className="text-indigo-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-bold">Correlation vs. Causation Interpretation Principle:</p>
          <p className="leading-relaxed">
            Shapley attribution measures the marginal contribution of each feature to the model’s fraud-buyer risk prediction on the evaluation dataset.
            These values demonstrate empirical feature importance and predictive signal—they represent correlational behavioral signatures rather than direct causal claims.
          </p>
        </div>
      </div>

      {/* Charts Grid: Bar Chart & Modality Donut */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Global SHAP Ranking Bar Chart (8 cols) */}
        <div className="lg:col-span-8 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Top Feature Importance (Mean |SHAP Value|)</h3>
              <p className="text-xs text-slate-500">Ranked by global impact on fake-review buyer classification</p>
            </div>
            <span className="text-[11px] font-mono font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
              Peak: {peakFeatureName} {peakFeaturePct !== '—' ? `(${peakFeaturePct}%)` : ''}
            </span>
          </div>

          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                layout="vertical"
                data={top12Features}
                margin={{ top: 10, right: 30, left: 60, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} unit="%" />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ fontSize: 11, fill: '#334155', fontFamily: 'monospace' }}
                  width={140}
                />
                <RechartsTooltip
                  formatter={(val: any) => [`${val}% importance`, 'Global Contribution']}
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff' }}
                />
                <Bar dataKey="percent_importance" fill="#4f46e5" radius={[0, 4, 4, 0]}>
                  {top12Features.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={
                        entry.modality === 'network'
                          ? '#4f46e5'
                          : entry.modality === 'image'
                          ? '#10b981'
                          : '#0ea5e9'
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center gap-4 text-xs pt-2 border-t border-slate-100">
            <span className="flex items-center gap-1.5 text-slate-600">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-600" /> Network Centrality
            </span>
            <span className="flex items-center gap-1.5 text-slate-600">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-500" /> Review Behaviors
            </span>
            <span className="flex items-center gap-1.5 text-slate-600">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Image Similarity
            </span>
          </div>
        </div>

        {/* Modality Contribution Pie (4 cols) */}
        <div className="lg:col-span-4 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-slate-900">Modality Contribution Share</h3>
              <span className="text-[10px] font-mono text-slate-500 font-semibold">100% Total</span>
            </div>
            <p className="text-xs text-slate-500">Aggregate feature weight grouped by analytical modality</p>

            <div className="h-56 flex items-center justify-center mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={modalityPie}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={80}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {modalityPie.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <RechartsTooltip
                    formatter={(val: any) => [`${val}%`, 'Modality Weight']}
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff' }}
                  />
                  <Legend verticalAlign="bottom" height={36} iconType="circle" />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-3 bg-slate-50 rounded-lg text-xs space-y-2 border border-slate-100">
            <div className="flex justify-between items-center">
              <span className="text-slate-600 flex items-center gap-1.5">
                <Network size={14} className="text-indigo-600" /> Network Centralities:
              </span>
              <span className="font-mono font-bold text-slate-900">{netWeight !== null ? `${netWeight.toFixed(2)}%` : '—'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-600 flex items-center gap-1.5">
                <MessageSquare size={14} className="text-sky-600" /> Review Behaviors:
              </span>
              <span className="font-mono font-bold text-slate-900">{revWeight !== null ? `${revWeight.toFixed(2)}%` : '—'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-600 flex items-center gap-1.5">
                <ImageIcon size={14} className="text-emerald-600" /> Image Similarity:
              </span>
              <span className="font-mono font-bold text-slate-900">{imgWeight !== null ? `${imgWeight.toFixed(2)}%` : '—'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Catalog Table */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
        <h3 className="text-sm font-bold text-slate-900">Complete Feature Importance Catalog (All 28 Dimensions)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Rank</th>
                <th className="py-2.5 px-3">Feature Name</th>
                <th className="py-2.5 px-3">Analytical Modality</th>
                <th className="py-2.5 px-3 text-right">Mean |SHAP Value|</th>
                <th className="py-2.5 px-3 text-right font-bold text-indigo-600">Relative Weight</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {features.map((f, idx) => (
                <tr key={f.feature} className="hover:bg-slate-50/70 transition-colors">
                  <td className="py-2 px-3 text-slate-400">#{idx + 1}</td>
                  <td className="py-2 px-3 font-sans font-medium text-slate-900">{f.feature}</td>
                  <td className="py-2 px-3 font-sans capitalize">
                    <span
                      className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                        f.modality === 'network'
                          ? 'bg-indigo-50 text-indigo-700'
                          : f.modality === 'image'
                          ? 'bg-emerald-50 text-emerald-700'
                          : 'bg-sky-50 text-sky-700'
                      }`}
                    >
                      {f.modality}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-right text-slate-700">{f.mean_abs_shap.toFixed(6)}</td>
                  <td className="py-2 px-3 text-right font-bold text-indigo-600">{f.percent_importance.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

