import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { getExperiments, getExperimentDetail } from '../../services/api';
import { ExperimentRecord } from '../../types';
import {
  FlaskConical,
  Search,
  Filter,
  Download,
  Info,
  CheckCircle2,
  Clock,
  ExternalLink,
  ChevronRight,
  X,
  FileCode,
} from 'lucide-react';

export const ExperimentsPage: React.FC = () => {
  const { t, mode } = useApp();
  const [records, setRecords] = useState<ExperimentRecord[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [search, setSearch] = useState<string>('');
  const [page, setPage] = useState<number>(1);
  const [pageSize, setPageSize] = useState<number>(50);
  const [activeExperiment, setActiveExperiment] = useState<ExperimentRecord | null>(null);

  const fetchRegistry = async () => {
    setLoading(true);
    try {
      const res = await getExperiments(selectedCategory, undefined, search, page, pageSize, mode);
      setRecords(res.records);
      setTotal(res.total);
    } catch (e) {
      console.warn('Failed to load experiments registry:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRegistry();
  }, [selectedCategory, search, page, pageSize, mode]);

  const handleExportCsv = () => {
    if (records.length === 0) return;
    const headers = Object.keys(records[0]).join(',');
    const rows = records.map((r) => Object.values(r).map((v) => `"${v ?? ''}"`).join(',')).join('\n');
    const csvContent = `data:text/csv;charset=utf-8,${headers}\n${rows}`;
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `experiment_registry_${selectedCategory.toLowerCase()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Research Experiment Registry (98 Entries)</h2>
          <p className="text-xs text-slate-500 mt-1">
            Machine-readable provenance catalog tracking split protocols, random seeds, hyperparameters, and independent verification status.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Category Filter */}
          <select
            value={selectedCategory}
            onChange={(e) => {
              setSelectedCategory(e.target.value);
              setPage(1);
            }}
            className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-slate-700 font-medium focus:ring-2 focus:ring-indigo-500 focus:outline-none"
          >
            <option value="All">All Categories ({total})</option>
            <option value="Final independent test">Final Independent Test Results (54)</option>
            <option value="Historical">Historical Baseline Results (24)</option>
            <option value="Development cross-validation">Dev 5x2 CV Results (18)</option>
            <option value="Synthetic">Synthetic Ring Experiments (1)</option>
            <option value="Demonstration">Demonstration Experiments (1)</option>
          </select>

          {/* Search Input */}
          <div className="relative">
            <Search size={14} className="absolute left-2.5 top-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search algorithm or ID..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              className="text-xs pl-8 pr-3 py-1.5 bg-slate-50 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none w-44 sm:w-56"
            />
          </div>

          {/* Export CSV */}
          <button
            onClick={handleExportCsv}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-xs font-medium text-slate-700 transition-colors"
          >
            <Download size={14} />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Main Registry Table */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-sm overflow-hidden">
        <div className="p-4 border-b border-slate-100 flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-700">
            Showing {records.length} of {total} registered experiments
          </span>
          <span className="text-slate-400 text-[11px]">Click row to open comprehensive audit metadata</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Experiment ID</th>
                <th className="py-2.5 px-3">Algorithm</th>
                <th className="py-2.5 px-3">Feature Group</th>
                <th className="py-2.5 px-3">Evaluation Type</th>
                <th className="py-2.5 px-3">Split Protocol</th>
                <th className="py-2.5 px-3 text-right">Fake F1</th>
                <th className="py-2.5 px-3 text-right">ROC-AUC</th>
                <th className="py-2.5 px-3 text-center">Status</th>
                <th className="py-2.5 px-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {records.map((r, idx) => {
                const isFinal = r.evaluation_type?.includes('Final');
                const isHistorical = r.evaluation_type?.includes('Historical');

                return (
                  <tr
                    key={`${r.experiment_id}_${idx}`}
                    onClick={() => setActiveExperiment(r)}
                    className="hover:bg-slate-50/80 transition-colors cursor-pointer group"
                  >
                    <td className="py-2.5 px-3 font-semibold text-slate-900 truncate max-w-[150px]">
                      {r.experiment_id}
                    </td>
                    <td className="py-2.5 px-3 font-sans text-slate-800">{r.algorithm}</td>
                    <td className="py-2.5 px-3 font-sans">
                      <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 text-[11px]">
                        {r.feature_group}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-sans text-[11px]">
                      <span
                        className={`px-2 py-0.5 rounded font-medium ${
                          isFinal
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : isHistorical
                            ? 'bg-amber-50 text-amber-700 border border-amber-200'
                            : 'bg-indigo-50 text-indigo-700 border border-indigo-200'
                        }`}
                      >
                        {r.evaluation_type}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-sans text-slate-500 text-[11px] truncate max-w-[150px]">
                      {r.split_protocol || 'Not recorded'}
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold text-indigo-600">
                      {r.fake_f1 !== undefined && r.fake_f1 !== null ? Number(r.fake_f1).toFixed(4) : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-700">
                      {r.roc_auc !== undefined && r.roc_auc !== null ? Number(r.roc_auc).toFixed(4) : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-center font-sans">
                      <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded">
                        <CheckCircle2 size={11} /> COMPLETED
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-center text-slate-400 group-hover:text-indigo-600">
                      <ChevronRight size={16} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Slide-over Audit Detail Modal */}
      {activeExperiment && (
        <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-xl border border-slate-200 max-w-2xl w-full p-6 space-y-4 max-h-[85vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <span className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider font-mono">
                  {activeExperiment.evaluation_type}
                </span>
                <h3 className="text-lg font-bold text-slate-900">{activeExperiment.experiment_id}</h3>
              </div>
              <button
                onClick={() => setActiveExperiment(null)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100"
              >
                <X size={18} />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-slate-500 block text-[10px] uppercase">Algorithm</span>
                <span className="font-bold text-slate-900 font-mono text-sm">{activeExperiment.algorithm}</span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-slate-500 block text-[10px] uppercase">Feature Group</span>
                <span className="font-bold text-indigo-600 font-mono text-sm">{activeExperiment.feature_group}</span>
              </div>
            </div>

            <div className="space-y-2 text-xs">
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 space-y-1.5">
                <div className="flex justify-between">
                  <span className="text-slate-500">Split Protocol:</span>
                  <span className="font-semibold text-slate-800">{activeExperiment.split_protocol || 'Not recorded'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Random Seed:</span>
                  <span className="font-mono font-bold text-slate-800">
                    {activeExperiment.random_seed !== undefined && activeExperiment.random_seed !== null ? activeExperiment.random_seed : 'Not recorded'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Dataset Fingerprint:</span>
                  <span className="font-mono text-slate-600 text-[10px] truncate max-w-[280px]">
                    {activeExperiment.dataset_fingerprint || 'Not recorded'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Artifact File:</span>
                  <span className="font-mono text-indigo-700 text-[10px]">
                    {activeExperiment.artifact_path || 'Not recorded'}
                  </span>
                </div>
              </div>

              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 space-y-1">
                <span className="text-slate-500 font-semibold block text-[11px]">Hyperparameters:</span>
                <code className="text-[11px] font-mono text-slate-700 block bg-white p-2 rounded border border-slate-200">
                  {activeExperiment.parameters || 'Not recorded'}
                </code>
              </div>

              <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-900 text-xs">
                <strong>Verification Status:</strong> {activeExperiment.verification_status || 'Not recorded'}
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setActiveExperiment(null)}
                className="px-4 py-2 bg-slate-900 text-white text-xs font-semibold rounded-lg hover:bg-slate-800"
              >
                Close Audit View
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

