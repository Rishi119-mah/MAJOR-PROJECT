import React, { useState, useEffect } from 'react';
import { getDashboardOverview } from '../../services/api';
import { DashboardOverview } from '../../types';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Package, AlertTriangle, CheckCircle, Award, Database } from 'lucide-react';

export const DashboardPage: React.FC = () => {
  const [data, setData] = useState<DashboardOverview | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await getDashboardOverview();
        setData(res);
      } catch (err: any) {
        console.error('Failed to load dashboard overview:', err);
        setError('Failed to connect to backend API. Please ensure the server is running.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-gray-500 text-sm">
        Loading dashboard metrics from backend...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-800 p-5 rounded-lg text-sm">
        <p className="font-semibold">Unable to load dashboard data</p>
        <p className="mt-1 text-xs">{error || 'No data returned from backend API.'}</p>
      </div>
    );
  }

  const ds = data.dataset_statistics;
  const ms = data.model_statistics;

  // Chart data: Feature group progression comparing baseline vs network-enhanced models
  const chartData = (data.feature_group_progression || []).map((item) => ({
    name: item.group,
    F1: parseFloat((item.f1 * 100).toFixed(1)),
    ROCAUC: parseFloat((item.auc * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* 1. Four Core Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Products */}
        <div className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Total Products</span>
            <div className="w-8 h-8 rounded-md bg-blue-50 text-blue-600 flex items-center justify-center">
              <Package size={16} />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-gray-900 font-mono">
              {ds.total_products.toLocaleString()}
            </span>
            <p className="text-xs text-gray-500 mt-1">Processed product records</p>
          </div>
        </div>

        {/* Fake-labelled Products */}
        <div className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Fake-Labelled</span>
            <div className="w-8 h-8 rounded-md bg-amber-50 text-amber-600 flex items-center justify-center">
              <AlertTriangle size={16} />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-gray-900 font-mono">
              {ds.fake_products_count.toLocaleString()}
            </span>
            <p className="text-xs text-gray-500 mt-1">
              {ds.fake_prevalence_pct ? `${ds.fake_prevalence_pct.toFixed(1)}% prevalence` : 'Manipulated products'}
            </p>
          </div>
        </div>

        {/* Genuine Products */}
        <div className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Genuine Products</span>
            <div className="w-8 h-8 rounded-md bg-emerald-50 text-emerald-600 flex items-center justify-center">
              <CheckCircle size={16} />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-gray-900 font-mono">
              {ds.genuine_products_count.toLocaleString()}
            </span>
            <p className="text-xs text-gray-500 mt-1">
              {(100 - (ds.fake_prevalence_pct || 0)).toFixed(1)}% verified normal
            </p>
          </div>
        </div>

        {/* Best Available Model F1-score */}
        <div className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Best Model F1-Score</span>
            <div className="w-8 h-8 rounded-md bg-blue-50 text-blue-600 flex items-center justify-center">
              <Award size={16} />
            </div>
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-blue-600 font-mono">
              {ms.best_fake_f1 ? ms.best_fake_f1.toFixed(4) : '—'}
            </span>
            <p className="text-xs text-gray-500 mt-1 truncate" title={ms.best_validated_model}>
              {ms.best_validated_model || 'Holdout evaluated'}
            </p>
          </div>
        </div>
      </div>

      {/* 2. One Simple Model-Performance Chart */}
      <div className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
        <div className="border-b border-gray-100 pb-3 mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-gray-900">Feature Modality Progression</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Comparison of detection performance as Network and Image features are added to review text baselines.
            </p>
          </div>
          <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-0.5 rounded border border-blue-100">
            Holdout Test Metrics (%)
          </span>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#4b5563' }} />
              <YAxis domain={[60, 100]} tick={{ fontSize: 12, fill: '#4b5563' }} />
              <Tooltip
                formatter={(value: any) => [`${value}%`]}
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '6px',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
              <Bar dataKey="F1" name="Fake F1-Score (%)" fill="#2563eb" radius={[4, 4, 0, 0]} />
              <Bar dataKey="ROCAUC" name="ROC-AUC (%)" fill="#64748b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. Small Dataset Information Section */}
      <div className="bg-white border border-gray-200 rounded-lg p-5 shadow-sm">
        <div className="flex items-center gap-2 border-b border-gray-100 pb-3 mb-3">
          <Database size={16} className="text-blue-600" />
          <h2 className="text-sm font-bold text-gray-900">Dataset & Evaluation Methodology</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-gray-600">
          <div>
            <span className="font-semibold text-gray-900 block">Dataset Source:</span>
            <p className="mt-1 text-gray-600">
              Amazon Electronics co-review interaction network and product-level aggregated metadata.
            </p>
          </div>
          <div>
            <span className="font-semibold text-gray-900 block">Feature Dimensions (28 Total):</span>
            <p className="mt-1 text-gray-600">
              12 Review Dynamics, 4 Bipartite Network Centralities (PageRank, degree, clustering coefficient), and 12 ResNet Image Similarities.
            </p>
          </div>
          <div>
            <span className="font-semibold text-gray-900 block">Validation Protocol:</span>
            <p className="mt-1 text-gray-600">
              Strict 80% Development / 20% Holdout test split, stratified on target labels with zero data leakage.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
