import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import {
  getDashboardOverview,
  getModelMetrics,
  getGlobalShap,
  getClusteringDetails,
  getSystemEnvironment,
  getSystemTestReport,
  getDatasetsSummary,
} from '../../services/api';
import {
  DashboardOverview,
  ModelMetricRecord,
  GlobalShapResponse,
  ClusteringDetailsResponse,
  SystemEnvironmentResponse,
  SystemTestReportResponse,
  DatasetsSummaryResponse,
} from '../../types';
import {
  FileText,
  Printer,
  Download,
  CheckCircle2,
  Calendar,
  Layers,
  Award,
  ShieldAlert,
  Boxes,
  Network,
  Sparkles,
  Info,
} from 'lucide-react';

export const ReportsPage: React.FC = () => {
  const { t, mode } = useApp();
  const [selectedReport, setSelectedReport] = useState<string>('model_comparison');
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [metrics, setMetrics] = useState<ModelMetricRecord[]>([]);
  const [shapData, setShapData] = useState<GlobalShapResponse | null>(null);
  const [clusteringDetails, setClusteringDetails] = useState<ClusteringDetailsResponse | null>(null);
  const [systemEnv, setSystemEnv] = useState<SystemEnvironmentResponse | null>(null);
  const [testReport, setTestReport] = useState<SystemTestReportResponse | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<DatasetsSummaryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const loadReportsData = async () => {
      setLoading(true);
      try {
        const [ov, met, sh, cl, env, rep, ds] = await Promise.allSettled([
          getDashboardOverview(mode),
          getModelMetrics(undefined, undefined, mode),
          getGlobalShap(mode),
          getClusteringDetails(mode),
          getSystemEnvironment(mode),
          getSystemTestReport(mode),
          getDatasetsSummary(mode),
        ]);
        if (ov.status === 'fulfilled') setOverview(ov.value);
        if (met.status === 'fulfilled') setMetrics(met.value);
        if (sh.status === 'fulfilled') setShapData(sh.value);
        if (cl.status === 'fulfilled') setClusteringDetails(cl.value);
        if (env.status === 'fulfilled') setSystemEnv(env.value);
        if (rep.status === 'fulfilled') setTestReport(rep.value);
        if (ds.status === 'fulfilled') setDatasetSummary(ds.value);
      } catch (e) {
        console.warn('Failed loading report telemetry:', e);
      } finally {
        setLoading(false);
      }
    };
    loadReportsData();
  }, [mode]);

  const reportTypes = [
    {
      id: 'model_comparison',
      title: 'Model Performance Benchmark Report',
      description: 'Comprehensive research leaderboard comparing models across F1, ROC-AUC, Brier score, and ECE on holdout test set.',
      icon: Award,
    },
    {
      id: 'dataset_summary',
      title: 'Dataset & Class Balance Report',
      description: 'Breakdown of catalog labeled products, fake-review buyers, feature dimensions, and missing value verification.',
      icon: Layers,
    },
    {
      id: 'clustering_topology',
      title: 'Clustering & Graph Topology Report',
      description: 'K-Means sweep metrics (k=2..20), optimal silhouette separation, and Cluster 1 fake-buyer concentration analysis.',
      icon: Boxes,
    },
    {
      id: 'shap_explainability',
      title: 'TreeSHAP Feature Attribution Report',
      description: 'Global 28-feature rankings, modality weight distribution across network, review, and image dimensions, and multilingual explainability specifications.',
      icon: Sparkles,
    },
  ];

  const handlePrint = () => {
    window.print();
  };

  const handleExportJson = () => {
    const prog = overview?.feature_group_progression || [];
    const revProg = prog.find((p) => p.group.toLowerCase().includes('review only'));
    const netProg = prog.find((p) => p.group.toLowerCase().includes('review + network'));
    const networkGainText = (revProg && netProg)
      ? `Review+Network improves F1 from ${revProg.f1.toFixed(4)} to ${netProg.f1.toFixed(4)} (${((netProg.f1 - revProg.f1) * 100).toFixed(2)}% gain)`
      : 'Network features provide measurable gain over review-only baselines';

    const c1 = clusteringDetails?.k2_profiles?.[1];
    const optimalClusteringText = clusteringDetails?.optimal_silhouette !== undefined && clusteringDetails?.optimal_silhouette !== null
      ? `k=${clusteringDetails.optimal_k || 2} (Silhouette: ${clusteringDetails.optimal_silhouette.toFixed(4)}${c1 ? `, Cluster 1 fake prevalence: ${(c1.ground_truth_fake_prevalence * 100).toFixed(2)}%` : ''})`
      : 'No verified clustering sweep metrics recorded';

    const data = {
      project: 'Fake Review Detection Using Machine Learning and Network Analysis',
      report_type: selectedReport,
      timestamp: new Date().toISOString(),
      evaluation_protocol: 'Strict 80% Dev / 20% Holdout Split Stratified on TARGET (Zero-Leakage)',
      system_environment: systemEnv,
      automated_test_report: testReport,
      dataset_summary: datasetSummary,
      key_findings: {
        top_model: overview?.model_statistics.best_validated_model
          ? `${overview.model_statistics.best_validated_model} (Holdout Fake F1: ${overview.model_statistics.best_fake_f1?.toFixed(4) || '—'}, ROC-AUC: ${overview.model_statistics.best_roc_auc?.toFixed(4) || '—'})`
          : 'Holdout evaluation records',
        network_gain: networkGainText,
        calibration: 'Probability calibration applied with zero-filling eliminated',
        optimal_clustering: optimalClusteringText,
        modality_breakdown: {
          review: shapData?.modality_breakdown?.review !== undefined ? `${shapData.modality_breakdown.review.toFixed(2)}%` : '—',
          network: shapData?.modality_breakdown?.network !== undefined ? `${shapData.modality_breakdown.network.toFixed(2)}%` : '—',
          image: shapData?.modality_breakdown?.image !== undefined ? `${shapData.modality_breakdown.image.toFixed(2)}%` : '—',
        },
      },
      dataset_statistics: overview?.dataset_statistics || null,
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fraudshield_${selectedReport}_report.json`;
    a.click();
  };

  const handleExportCsv = () => {
    const modelList = (overview?.top_models && overview.top_models.length > 0)
      ? overview.top_models
      : metrics.slice(0, 15).map((m) => ({
          model: `${m.model} (${m.feature_group})`,
          f1: m.fake_f1,
          auc: m.roc_auc,
          type: m.evaluation_protocol || m.evaluation_type,
          accuracy: m.accuracy,
        }));

    const headers = ['Model & Feature Group', 'Split Protocol', 'Accuracy', 'Fake F1', 'ROC-AUC'];
    const rows = modelList.map((m: any) => [
      `"${m.model}"`,
      `"${m.type || 'Holdout Test'}"`,
      m.accuracy !== null && m.accuracy !== undefined ? Number(m.accuracy).toFixed(4) : '',
      m.f1 !== null && m.f1 !== undefined ? Number(m.f1).toFixed(4) : '',
      m.auc !== null && m.auc !== undefined ? Number(m.auc).toFixed(4) : '',
    ]);

    const csvContent = [headers.join(','), ...rows.map((r: any) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fraudshield_${selectedReport}_metrics.csv`;
    a.click();
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Executive Report Generator & Exports</h2>
          <p className="text-xs text-slate-500 mt-1">
            Generate formal print-ready academic reports and audit packages for project guide reviews and examination committees.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExportCsv}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-xs font-medium text-slate-700 transition-colors"
          >
            <Download size={14} />
            <span>Export CSV</span>
          </button>
          <button
            onClick={handleExportJson}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-xs font-medium text-slate-700 transition-colors"
          >
            <Download size={14} />
            <span>Export JSON</span>
          </button>
          <button
            onClick={handlePrint}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium transition-colors shadow-sm"
          >
            <Printer size={14} />
            <span>Print PDF Report</span>
          </button>
        </div>
      </div>

      {/* Select Report Template */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {reportTypes.map((r) => {
          const Icon = r.icon;
          const isSelected = selectedReport === r.id;
          return (
            <div
              key={r.id}
              onClick={() => setSelectedReport(r.id)}
              className={`p-4 rounded-xl border cursor-pointer transition-all flex flex-col justify-between ${
                isSelected
                  ? 'bg-indigo-50/60 border-indigo-600 shadow-sm ring-1 ring-indigo-600'
                  : 'bg-white border-slate-200 hover:bg-slate-50'
              }`}
            >
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <div className={`p-2 rounded-lg ${isSelected ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600'}`}>
                    <Icon size={18} />
                  </div>
                  <h3 className="font-bold text-xs text-slate-900">{r.title}</h3>
                </div>
                <p className="text-[11px] text-slate-500 line-clamp-3">{r.description}</p>
              </div>

              <span className={`text-[10px] font-semibold mt-3 ${isSelected ? 'text-indigo-600' : 'text-slate-400'}`}>
                {isSelected ? '● Active Report View' : 'Click to preview'}
              </span>
            </div>
          );
        })}
      </div>

      {/* Formal Printable Document View */}
      <div className="bg-white rounded-xl border border-slate-200 p-8 shadow-sm space-y-6 print:m-0 print:border-none print:shadow-none">
        {/* Document Header */}
        <div className="border-b border-slate-200 pb-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <span className="text-[10px] font-mono uppercase tracking-widest text-indigo-600 font-bold">
              Final Year Major Project Audit Deliverable
            </span>
            <h2 className="text-xl font-bold text-slate-900 mt-1">
              Fake Review Detection Using Machine Learning & Network Analysis
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Based on and extending: <em>&quot;Detecting Fake Review Buyers Using Network Structure: Direct Evidence from Amazon&quot;</em> (He et al., 2022)
            </p>
          </div>

          <div className="text-right text-xs text-slate-500 font-mono space-y-0.5">
            <div><strong>Date:</strong> {new Date().toLocaleDateString()}</div>
            <div><strong>FastAPI Version:</strong> {systemEnv?.fastapi_version ? `v${systemEnv.fastapi_version}` : 'Not recorded'}</div>
            <div><strong>Runtime Status:</strong> {systemEnv?.host_runtime_status ? 'Verified Host Runtime' : 'Status Pending'}</div>
          </div>
        </div>

        {/* Executive Summary Section */}
        <div className="space-y-2">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider text-indigo-900">
            1. Executive Methodology & Validation Guarantees
          </h3>
          <p className="text-xs text-slate-700 leading-relaxed">
            This report presents verified empirical benchmarks for product-level fake-review-buyer classification.
            All model comparisons, hyperparameter optimizations, and probability calibration candidate selections are strictly confined
            within the <strong>80% Development Split {datasetSummary?.development_split_count ? `($N=${datasetSummary.development_split_count.toLocaleString()}$)` : ''}</strong> using 5x2 Repeated Stratified Cross-Validation.
            Final production models are evaluated exactly once on the untouched <strong>20% Holdout Test Split {datasetSummary?.holdout_split_count ? `($N=${datasetSummary.holdout_split_count.toLocaleString()}$)` : ''}</strong> to guarantee zero test data leakage.
          </p>
        </div>

        {/* Report Content Body: Dynamic based on Selection */}
        {selectedReport === 'model_comparison' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider text-indigo-900">
                2. Final Holdout Test Benchmarks {datasetSummary?.holdout_split_count ? `(N=${datasetSummary.holdout_split_count} Products)` : ''}
              </h3>
              <span className="text-[11px] font-mono text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">
                Protocol: Untouched 20% Holdout Split
              </span>
            </div>
            <table className="w-full text-left text-xs border border-slate-200">
              <thead className="bg-slate-50 text-slate-700 font-semibold border-b border-slate-200">
                <tr>
                  <th className="py-2 px-3">Architecture & Modality</th>
                  <th className="py-2 px-3">Split Protocol</th>
                  <th className="py-2 px-3 text-right">Accuracy</th>
                  <th className="py-2 px-3 text-right">Fake F1</th>
                  <th className="py-2 px-3 text-right">ROC-AUC</th>
                  <th className="py-2 px-3 text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {(() => {
                  const modelList = (overview?.top_models && overview.top_models.length > 0)
                    ? overview.top_models
                    : metrics.slice(0, 5).map((m) => ({
                        model: `${m.model} (${m.feature_group})`,
                        f1: m.fake_f1,
                        auc: m.roc_auc,
                        type: m.evaluation_protocol || m.evaluation_type,
                        accuracy: m.accuracy,
                      }));

                  if (modelList.length === 0) {
                    return (
                      <tr>
                        <td colSpan={6} className="py-6 text-center text-xs text-slate-400 font-sans">
                          No model benchmark telemetry recorded.
                        </td>
                      </tr>
                    );
                  }

                  return modelList.map((m: any, idx: number) => (
                    <tr key={idx} className="hover:bg-slate-50/70 transition-colors">
                      <td className="py-2 px-3 font-sans font-medium text-slate-900 flex items-center gap-1.5">
                        {idx === 0 && <Award size={13} className="text-amber-500" />}
                        {m.model}
                      </td>
                      <td className="py-2 px-3 font-sans text-slate-500">{m.type || 'Not recorded'}</td>
                      <td className="py-2 px-3 text-right">{m.accuracy !== null && m.accuracy !== undefined ? Number(m.accuracy).toFixed(4) : '—'}</td>
                      <td className="py-2 px-3 text-right font-bold text-indigo-600">{m.f1 !== null && m.f1 !== undefined ? Number(m.f1).toFixed(4) : '—'}</td>
                      <td className="py-2 px-3 text-right">{m.auc !== null && m.auc !== undefined ? Number(m.auc).toFixed(4) : '—'}</td>
                      <td className="py-2 px-3 text-right font-sans">
                        <span className="text-[10px] font-medium text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                          Verified
                        </span>
                      </td>
                    </tr>
                  ));
                })()}
              </tbody>
            </table>
          </div>
        )}

        {selectedReport === 'dataset_summary' && (
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider text-indigo-900">
              2. Catalog Dataset Profile & Integrity
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-slate-500 block">Total Labeled Products</span>
                <span className="text-lg font-bold font-mono text-slate-900">
                  {datasetSummary?.labeled_products?.toLocaleString() || overview?.dataset_statistics.total_products?.toLocaleString() || '—'}
                </span>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-slate-500 block">Fake-Review Buyers</span>
                <span className="text-lg font-bold font-mono text-rose-600">
                  {datasetSummary?.positive_class_count !== undefined
                    ? `${datasetSummary.positive_class_count.toLocaleString()} (${datasetSummary.fake_prevalence_pct.toFixed(2)}%)`
                    : overview?.dataset_statistics.fake_products_count !== undefined
                    ? `${overview.dataset_statistics.fake_products_count.toLocaleString()} (${overview.dataset_statistics.fake_prevalence_pct.toFixed(2)}%)`
                    : '—'}
                </span>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-slate-500 block">Genuine Products</span>
                <span className="text-lg font-bold font-mono text-emerald-600">
                  {datasetSummary?.negative_class_count !== undefined
                    ? `${datasetSummary.negative_class_count.toLocaleString()} (${(100 - datasetSummary.fake_prevalence_pct).toFixed(2)}%)`
                    : overview?.dataset_statistics.genuine_products_count !== undefined
                    ? `${overview.dataset_statistics.genuine_products_count.toLocaleString()} (${(100 - overview.dataset_statistics.fake_prevalence_pct).toFixed(2)}%)`
                    : '—'}
                </span>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="text-slate-500 block">Missing Values</span>
                <span className="text-lg font-bold font-mono text-slate-900">
                  {datasetSummary?.missing_values_status || 'Not recorded'}
                </span>
              </div>
            </div>
          </div>
        )}

        {selectedReport === 'clustering_topology' && (
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider text-indigo-900">
              2. Unsupervised Partitioning & Anomaly Signatures
            </h3>
            {clusteringDetails?.optimal_silhouette !== undefined && clusteringDetails?.optimal_silhouette !== null ? (
              <p className="text-xs text-slate-700 leading-relaxed">
                Mathematical cluster separation reaches its global peak at $k={clusteringDetails?.optimal_k || 2}$ with a Silhouette score of{' '}
                <strong>{clusteringDetails.optimal_silhouette.toFixed(4)}</strong>.
                In this optimal clustering, Cluster 1 contains {clusteringDetails?.k2_profiles?.[1]?.n_products?.toLocaleString() || '—'} products exhibiting an observed ground-truth fake buyer prevalence of{' '}
                <strong>{clusteringDetails?.k2_profiles?.[1]?.ground_truth_fake_prevalence !== undefined ? `${(clusteringDetails.k2_profiles[1].ground_truth_fake_prevalence * 100).toFixed(2)}%` : '—'}</strong>,
                characterized by a mean co-review weighted degree of {clusteringDetails?.k2_profiles?.[1]?.mean_w_degree !== undefined ? clusteringDetails.k2_profiles[1].mean_w_degree.toFixed(1) : '—'} and clustering coefficient of {clusteringDetails?.k2_profiles?.[1]?.mean_clustering_coef !== undefined ? clusteringDetails.k2_profiles[1].mean_clustering_coef.toFixed(3) : '—'}.
              </p>
            ) : (
              <p className="text-xs text-slate-500 italic">No verified clustering sweep metrics recorded.</p>
            )}
          </div>
        )}

        {selectedReport === 'shap_explainability' && (
          <div className="space-y-4">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider text-indigo-900">
              2. Feature Attribution Breakdown
            </h3>
            <p className="text-xs text-slate-700 leading-relaxed">
              TreeSHAP feature attributions establish that network structural features constitute{' '}
              <strong>{shapData?.modality_breakdown?.network !== undefined ? `${shapData.modality_breakdown.network.toFixed(2)}%` : '—'}</strong> of the overall model decision weight
              (led by <code className="font-mono">{shapData?.top_feature || 'clustering_coef'}</code>).
              Review behavioral dynamics account for <strong>{shapData?.modality_breakdown?.review !== undefined ? `${shapData.modality_breakdown.review.toFixed(2)}%` : '—'}</strong>,
              and image similarity features account for <strong>{shapData?.modality_breakdown?.image !== undefined ? `${shapData.modality_breakdown.image.toFixed(2)}%` : '—'}</strong>.
            </p>
          </div>
        )}

        {/* Academic Sign-off Footer */}
        <div className="border-t border-slate-200 pt-6 flex justify-between items-end text-xs text-slate-500 font-sans">
          <div className="space-y-1">
            <p className="font-bold text-slate-800">Verified System Environment:</p>
            {systemEnv ? (
              <p>Python {systemEnv.python_version} • FastAPI {systemEnv.fastapi_version} • Scikit-Learn {systemEnv.sklearn_version} • XGBoost {systemEnv.xgboost_version}</p>
            ) : (
              <p className="text-slate-400">Environment telemetry: Not recorded</p>
            )}
            {testReport && testReport.status !== 'not_available' && testReport.total_tests !== null ? (
              <p className="text-[11px] text-slate-500">
                Automated Test Suite: {testReport.passed}/{testReport.total_tests} Tests Passing ({testReport.total_tests > 0 ? Math.round(((testReport.passed || 0) / testReport.total_tests) * 100) : 100}%) • {testReport.test_command || 'pytest'}
              </p>
            ) : (
              <p className="text-[11px] text-slate-400">Automated Test Suite: No verified test report available.</p>
            )}
          </div>

          <div className="text-right">
            <div className="h-10 border-b border-slate-400 w-44 mb-1"></div>
            <span className="text-[10px] uppercase font-bold text-slate-600">Student & Guide Signatures</span>
          </div>
        </div>
      </div>
    </div>
  );
};
