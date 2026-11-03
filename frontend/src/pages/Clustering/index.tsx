import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { getClusteringDetails } from '../../services/api';
import { ClusteringDetailsResponse, ClusterProfile, ClusteringSweepMetric } from '../../types';
import {
  Boxes,
  TrendingUp,
  Award,
  Layers,
  Info,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Sliders,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from 'recharts';

export const ClusteringPage: React.FC = () => {
  const { t, mode } = useApp();
  const [details, setDetails] = useState<ClusteringDetailsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeK, setActiveK] = useState<'k2' | 'k20'>('k2');
  const [selectedCluster, setSelectedCluster] = useState<ClusterProfile | null>(null);

  useEffect(() => {
    const fetchClustering = async () => {
      setLoading(true);
      try {
        const data = await getClusteringDetails(mode);
        setDetails(data);
        if (data.k2_profiles.length > 0) {
          setSelectedCluster(data.k2_profiles[1]); // Focus on high-risk Cluster 1 by default
        }
      } catch (e) {
        console.warn('Failed to load clustering details:', e);
      } finally {
        setLoading(false);
      }
    };
    fetchClustering();
  }, [mode]);

  const profiles = activeK === 'k2' ? details?.k2_profiles || [] : details?.k20_profiles || [];

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Unsupervised K-Means Clustering Analysis</h2>
          <p className="text-xs text-slate-500 mt-1">
            Evaluates mathematical cluster separation across $k=2 \dots 20$ and compares cluster behavioral profiles against ground-truth fake-buyer prevalence.
          </p>
        </div>

        {/* Toggle K=2 vs K=20 */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-semibold text-slate-600 uppercase">View Solution:</label>
          <div className="flex items-center bg-slate-100 p-0.5 rounded-lg border border-slate-200 text-xs">
            <button
              onClick={() => {
                setActiveK('k2');
                if (details?.k2_profiles) setSelectedCluster(details.k2_profiles[1]);
              }}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeK === 'k2'
                  ? 'bg-indigo-600 text-white shadow-sm font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Optimal k = 2 (Peak Separation)
            </button>
            <button
              onClick={() => {
                setActiveK('k20');
                if (details?.k20_profiles) setSelectedCluster(details.k20_profiles[0]);
              }}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeK === 'k20'
                  ? 'bg-indigo-600 text-white shadow-sm font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Paper Baseline k = 20
            </button>
          </div>
        </div>
      </div>

      {/* Methodological Truth Note */}
      <div className="bg-indigo-50/70 border border-indigo-200 p-4 rounded-xl flex items-start gap-3 text-xs text-indigo-900">
        <Info size={18} className="text-indigo-600 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-bold">Rigorous Separation of Ground-Truth vs Model Predictions:</p>
          <p className="leading-relaxed">
            In compliance with our empirical audit, this dashboard strictly separates <strong>observed ground-truth fake prevalence</strong> from{' '}
            <strong>out-of-fold predicted fake probabilities ($p \ge 0.5$)</strong>.
            Cluster separation mathematically peaks at $k={details?.optimal_k ?? 2} (Silhouette: <strong>{details?.optimal_silhouette !== undefined && details?.optimal_silhouette !== null ? details.optimal_silhouette.toFixed(4) : '—'}</strong>), where Cluster 1 isolates {details?.k2_profiles?.[1]?.n_products?.toLocaleString() || '—'} products exhibiting an observed{' '}
            <strong>{details?.k2_profiles?.[1]?.ground_truth_fake_prevalence !== undefined ? `${(details.k2_profiles[1].ground_truth_fake_prevalence * 100).toFixed(1)}%` : '—'} ground-truth fake-review buyer prevalence</strong> (vs {details?.k2_profiles?.[0]?.ground_truth_fake_prevalence !== undefined ? `${(details.k2_profiles[0].ground_truth_fake_prevalence * 100).toFixed(1)}%` : '—'} in Cluster 0).
          </p>
        </div>
      </div>

      {/* Sweep Curves Grid (Silhouette & Davies-Bouldin) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Silhouette Score Line Chart */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Silhouette Score Sweep (k = 2 to 20)</h3>
              <p className="text-xs text-slate-500">Higher indicates tighter intra-cluster separation</p>
            </div>
            <span className="text-[11px] font-mono font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
              Peak: {details?.optimal_silhouette !== undefined && details?.optimal_silhouette !== null ? `${details.optimal_silhouette.toFixed(4)} at k=${details.optimal_k || 2}` : 'Unavailable'}
            </span>
          </div>

          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={details?.sweep_metrics || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="k" tick={{ fontSize: 11, fill: '#64748b' }} />
                <YAxis domain={[0.1, 0.3]} tick={{ fontSize: 11, fill: '#64748b' }} />
                <RechartsTooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff' }}
                />
                <Line
                  type="monotone"
                  dataKey="silhouette"
                  name="Silhouette Score"
                  stroke="#4f46e5"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#4f46e5' }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Davies-Bouldin Index Line Chart */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Davies-Bouldin Index (k = 2 to 20)</h3>
              <p className="text-xs text-slate-500">Lower indicates superior cluster compactness</p>
            </div>
            <span className="text-[11px] font-mono font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded">
              Lower is Better
            </span>
          </div>

          <div className="h-60">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={details?.sweep_metrics || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="k" tick={{ fontSize: 11, fill: '#64748b' }} />
                <YAxis domain={[1.2, 1.8]} tick={{ fontSize: 11, fill: '#64748b' }} />
                <RechartsTooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#fff' }}
                />
                <Line
                  type="monotone"
                  dataKey="davies_bouldin"
                  name="Davies-Bouldin Index"
                  stroke="#059669"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: '#059669' }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Cluster Profiles Comparison (Cards & Deep Dive) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Cluster Population & Prevalence Cards (7 cols) */}
        <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Cluster Structure ({activeK === 'k2' ? '2 Partitions' : '20 Partitions'})
              </h3>
              <p className="text-xs text-slate-500">Comparing cluster population vs ground-truth fake buyer concentration</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {profiles.slice(0, 6).map((c) => {
              const isSelected = selectedCluster?.cluster_ID === c.cluster_ID;
              const isHighRisk = c.ground_truth_fake_prevalence > 0.6;

              return (
                <div
                  key={c.cluster_ID}
                  onClick={() => setSelectedCluster(c)}
                  className={`p-4 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'border-indigo-600 bg-indigo-50/50 shadow-sm ring-1 ring-indigo-600'
                      : 'border-slate-200 bg-white hover:bg-slate-50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-sm text-slate-900 font-mono">
                      Cluster {c.cluster_ID}
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase ${
                        isHighRisk ? 'bg-rose-100 text-rose-800' : 'bg-emerald-100 text-emerald-800'
                      }`}
                    >
                      {isHighRisk ? 'Fraud Concentrated' : 'Normal Catalog'}
                    </span>
                  </div>

                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-500">Products in Cluster:</span>
                      <span className="font-mono font-bold text-slate-800">
                        {c.n_products} ({(c.share_of_total * 100).toFixed(1)}%)
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-slate-500">Ground-Truth Fake Prevalence:</span>
                      <span className={`font-mono font-bold ${isHighRisk ? 'text-rose-600' : 'text-slate-700'}`}>
                        {(c.ground_truth_fake_prevalence * 100).toFixed(1)}% ({c.ground_truth_fake_count} prods)
                      </span>
                    </div>

                    <div className="flex justify-between">
                      <span className="text-slate-500">Mean Out-of-Fold Model p(Fake):</span>
                      <span className="font-mono font-bold text-indigo-600">
                        {(c.mean_p_fake_predicted * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Selected Cluster Behavioral Deep Dive (5 cols) */}
        <div className="lg:col-span-5 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Cluster Behavioral Fingerprint</h3>
              <p className="text-xs text-slate-500">Distinct topological and review dynamics</p>
            </div>
            <span className="font-mono text-xs font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
              Cluster #{selectedCluster?.cluster_ID ?? 0}
            </span>
          </div>

          {selectedCluster ? (
            <div className="space-y-3 text-xs">
              <div className="p-3 bg-slate-50 rounded-lg space-y-2 border border-slate-100">
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">Mean Weighted Degree:</span>
                  <span className="font-mono font-bold text-slate-900">
                    {selectedCluster.mean_w_degree !== undefined && selectedCluster.mean_w_degree !== null
                      ? selectedCluster.mean_w_degree.toFixed(1)
                      : '—'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">Mean Clustering Coefficient:</span>
                  <span className="font-mono font-bold text-slate-900">
                    {selectedCluster.mean_clustering_coef !== undefined && selectedCluster.mean_clustering_coef !== null
                      ? selectedCluster.mean_clustering_coef.toFixed(3)
                      : '—'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">Mean Review Count:</span>
                  <span className="font-mono font-bold text-slate-900">
                    {selectedCluster.mean_n_of_reviews !== undefined && selectedCluster.mean_n_of_reviews !== null
                      ? `${selectedCluster.mean_n_of_reviews.toFixed(0)} reviews`
                      : '—'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-600">Mean Star Rating:</span>
                  <span className="font-mono font-bold text-slate-900">
                    {selectedCluster.mean_avg_review_rating !== undefined && selectedCluster.mean_avg_review_rating !== null
                      ? `★ ${selectedCluster.mean_avg_review_rating.toFixed(2)}`
                      : '—'}
                  </span>
                </div>
              </div>

              <div className="p-3 bg-amber-50/70 border border-amber-200 rounded-lg text-amber-900 text-[11px] leading-relaxed">
                <strong>Anomalous Topology Signature:</strong> Products in this partition exhibit concentrated review velocity (burstiness), elevated co-review degrees, and abnormal clustering coefficients characteristic of paid review seller groups.
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-400 py-6 text-center">Select a cluster on the left to inspect its characteristics.</p>
          )}
        </div>
      </div>
    </div>
  );
};

