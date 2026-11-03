import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { checkApiHealth, getModelRegistry, getDashboardOverview } from '../../services/api';
import { ModelRegistryItem, Language, AppMode } from '../../types';
import {
  Settings,
  Server,
  Database,
  Cpu,
  Globe,
  BookOpen,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Scale,
  ShieldCheck,
} from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const { language, setLanguage, mode, setMode, isOnline, t } = useApp();

  const [pingLatency, setPingLatency] = useState<number | null>(null);
  const [checking, setChecking] = useState<boolean>(false);
  const [lastCheck, setLastCheck] = useState<string>(new Date().toLocaleTimeString());
  const [models, setModels] = useState<ModelRegistryItem[]>([]);
  const [apiUrl, setApiUrl] = useState<string>(import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000');

  const testConnection = async () => {
    setChecking(true);
    const start = performance.now();
    try {
      await checkApiHealth();
      const latency = Math.round(performance.now() - start);
      setPingLatency(latency);
      setLastCheck(new Date().toLocaleTimeString());
    } catch {
      setPingLatency(null);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    testConnection();
    const fetchModels = async () => {
      try {
        const list = await getModelRegistry();
        setModels(list);
      } catch (e) {
        console.warn('Could not fetch model list:', e);
      }
    };
    fetchModels();
  }, []);

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Top Banner */}
      <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">System Configuration & Research Telemetry</h2>
          <p className="text-xs text-slate-500 mt-1">
            Manage FastAPI connection status, inspect registered model schemas, toggle languages, and review Indian regulatory compliance contexts.
          </p>
        </div>

        <button
          onClick={testConnection}
          disabled={checking}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium transition-colors shadow-sm disabled:opacity-50"
        >
          <RefreshCw size={14} className={checking ? 'animate-spin' : ''} />
          <span>Ping API Health</span>
        </button>
      </div>

      {/* Grid: API Status & Language Selector */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* API Health & Connection Card */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Server size={16} className="text-indigo-600" />
              FastAPI Backend Endpoint
            </h3>
            <span
              className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full ${
                isOnline ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-rose-500'}`} />
              {isOnline ? 'Online' : 'Offline'}
            </span>
          </div>

          <div className="space-y-3 text-xs">
            <div>
              <label className="text-slate-500 block mb-1">Configured Base URL (VITE_API_BASE_URL)</label>
              <input
                type="text"
                value={apiUrl}
                readOnly
                className="w-full text-xs font-mono px-3 py-2 border border-slate-200 rounded-lg bg-slate-50 text-slate-800"
              />
            </div>

            <div className="grid grid-cols-2 gap-2 text-center pt-1">
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-slate-500 block text-[10px] uppercase">Latency</span>
                <span className="text-lg font-bold font-mono text-indigo-600">
                  {pingLatency !== null ? `${pingLatency} ms` : '—'}
                </span>
              </div>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-slate-500 block text-[10px] uppercase">Last Checked</span>
                <span className="text-xs font-bold font-mono text-slate-800 mt-1 block">{lastCheck}</span>
              </div>
            </div>

            <div className="flex justify-between items-center py-1 border-t border-slate-100">
              <span className="text-slate-500">Execution Mode</span>
              <div className="flex gap-1">
                {(['research', 'demo'] as AppMode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`px-2.5 py-1 rounded text-xs font-medium capitalize transition-colors ${
                      mode === m ? 'bg-indigo-600 text-white font-semibold' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Language & Localization Card */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Globe size={16} className="text-indigo-600" />
              User Interface Localization
            </h3>
            <span className="text-xs font-mono text-slate-500 font-semibold uppercase">{language}</span>
          </div>

          <p className="text-xs text-slate-500">
            Select your preferred display language for the analytics console and explainable AI insights.
          </p>

          <div className="grid grid-cols-3 gap-3">
            {[
              { id: 'en', label: 'English', sub: 'Default' },
              { id: 'hi', label: 'हिंदी (Hindi)', sub: 'देवनागरी' },
              { id: 'hinglish', label: 'Hinglish', sub: 'Roman Hindi' },
            ].map((item) => (
              <button
                key={item.id}
                onClick={() => setLanguage(item.id as Language)}
                className={`p-3 rounded-xl border text-center transition-all ${
                  language === item.id
                    ? 'border-indigo-600 bg-indigo-50/60 shadow-sm ring-1 ring-indigo-600'
                    : 'border-slate-200 bg-white hover:bg-slate-50'
                }`}
              >
                <span className="text-xs font-bold text-slate-900 block">{item.label}</span>
                <span className="text-[10px] text-slate-400 font-mono mt-0.5 block">{item.sub}</span>
              </button>
            ))}
          </div>

          <div className="p-3 bg-slate-50 rounded-lg text-xs text-slate-600 border border-slate-100 space-y-1">
            <span className="font-semibold text-slate-800 block text-[11px]">Dynamic SHAP Translation:</span>
            <p className="text-[11px] leading-relaxed">
              FastAPI `/predict/product` endpoint natively returns multilingual explanations in English, Hindi, and Hinglish for all 21 registered models.
            </p>
          </div>
        </div>
      </div>

      {/* Registered Model Catalog Table */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-900">Registered Model Allowlist ({models.length} Models)</h3>
            <p className="text-xs text-slate-500">All registered artifacts verified on disk with strict schema contracts</p>
          </div>
          <span className="text-xs font-mono text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded font-semibold">
            Zero Silent Fallback
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-600 border-b border-slate-200 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Model Name</th>
                <th className="py-2.5 px-3">Feature Modality</th>
                <th className="py-2.5 px-3 text-center">Required Dimensions</th>
                <th className="py-2.5 px-3">Artifact File</th>
                <th className="py-2.5 px-3 text-center">Calibration</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {models.map((m) => {
                const isCalibrated = m.model_name.startsWith('calibrated_');
                return (
                  <tr key={m.model_name} className="hover:bg-slate-50/70 transition-colors">
                    <td className="py-2 px-3 font-sans font-medium text-slate-900">{m.model_name}</td>
                    <td className="py-2 px-3 font-sans capitalize">{m.feature_group}</td>
                    <td className="py-2 px-3 text-center text-indigo-600 font-bold">{m.required_features_count} feats</td>
                    <td className="py-2 px-3 text-slate-500 text-[11px]">{m.filename}</td>
                    <td className="py-2 px-3 text-center font-sans">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                          isCalibrated
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : 'bg-slate-100 text-slate-600'
                        }`}
                      >
                        {isCalibrated ? 'Isotonic Calibrated' : 'Raw Classifier'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Indian Regulatory & Responsible Use Accordion */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-4">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <Scale size={16} className="text-indigo-600" />
          Indian Regulatory Framework & Responsible AI Context
        </h3>

        <div className="space-y-3 text-xs text-slate-700 leading-relaxed">
          <p>
            In the Indian consumer protection landscape, fraudulent online reviews are governed under the{' '}
            <strong>Consumer Protection Act, 2019</strong> and the <strong>Consumer Protection (E-Commerce) Rules, 2020</strong>,
            which classify coordinated misleading reviews and non-disclosed promotional buying as unfair trade practices.
          </p>

          <p>
            Furthermore, the <strong>Bureau of Indian Standards (BIS)</strong> established standard{' '}
            <strong>IS 19000:2022</strong> (<em>Online Consumer Reviews — Principles and Requirements for their Collection, Moderation and Publication</em>),
            mandating transparency in review moderation, verification of genuine buyers, and detection of commercial review manipulation.
          </p>

          <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-slate-600 text-[11px]">
            <strong>Legal Disclaimer:</strong> This application is developed strictly for academic engineering research and demonstration.
            It generates algorithmic risk scores based on behavioral and topological statistical models. It does not constitute a certified legal audit or official regulatory compliance judgment.
          </div>
        </div>
      </div>
    </div>
  );
};

