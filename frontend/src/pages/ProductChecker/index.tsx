import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import {
  getModelRegistry,
  predictProduct,
  getProductDetails,
} from '../../services/api';
import { ModelRegistryItem, PredictionResponse } from '../../types';
import {
  Search,
  CheckCircle,
  AlertTriangle,
  Layers,
  ArrowRight,
} from 'lucide-react';

export const ProductCheckerPage: React.FC = () => {
  const { selectedModel, setSelectedModel } = useApp();
  const [models, setModels] = useState<ModelRegistryItem[]>([]);
  const [productIdInput, setProductIdInput] = useState<string>('1080');
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load available models from FastAPI registry
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const list = await getModelRegistry();
        if (Array.isArray(list) && list.length > 0) {
          setModels(list);
          if (!list.some((m) => m.model_name === selectedModel)) {
            setSelectedModel(list[0].model_name);
          }
        }
      } catch (err) {
        console.warn('Failed to load model registry:', err);
      }
    };
    fetchModels();
  }, []);

  const handleCheckProduct = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();

    const cleanId = productIdInput.trim();
    const numId = Number(cleanId);
    if (!cleanId || isNaN(numId) || numId <= 0) {
      setError('Please enter a valid numeric Product ID.');
      setPrediction(null);
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      // 1. Fetch exact features for this Product ID from backend
      const productData = await getProductDetails(numId);
      if (!productData || !productData.features) {
        throw new Error(`Product ID #${cleanId} not found in the verified dataset.`);
      }

      // 2. Submit exact corresponding features to selected model
      const result = await predictProduct({
        product_ID: numId,
        displayed_rating: productData.displayed_rating || 4.5,
        model_name: selectedModel,
        features: productData.features,
      });

      setPrediction(result);
    } catch (err: any) {
      console.error('Prediction failed:', err);
      setError(err.message || 'Failed to predict product fraud risk. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const sampleProducts = [
    { id: '1080', label: 'Product #1080 (High Risk Coordinated)' },
    { id: '205', label: 'Product #205 (Genuine Normal Reviews)' },
    { id: '501', label: 'Product #501 (Moderate Risk Clustering)' },
  ];

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Search & Configuration Card */}
      <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm space-y-5">
        <div>
          <h2 className="text-base font-bold text-gray-900">Product Review Risk Checker</h2>
          <p className="text-xs text-gray-500 mt-1">
            Analyze review dynamics and co-review network centralities using calibrated machine learning models.
          </p>
        </div>

        <form onSubmit={handleCheckProduct} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Product ID Input */}
            <div>
              <label htmlFor="product-id-input" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1.5">
                Product ID
              </label>
              <div className="relative">
                <input
                  id="product-id-input"
                  type="text"
                  value={productIdInput}
                  onChange={(e) => {
                    setProductIdInput(e.target.value);
                    if (prediction) setPrediction(null);
                    if (error) setError(null);
                  }}
                  placeholder="e.g. 1080, 205, 501"
                  className="w-full px-3.5 py-2.5 bg-gray-50 border border-gray-300 rounded-md text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:bg-white font-mono"
                />
              </div>
            </div>

            {/* Model Selection Dropdown */}
            <div>
              <label htmlFor="model-select" className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1.5">
                Evaluation Model
              </label>
              <select
                id="model-select"
                value={selectedModel}
                onChange={(e) => {
                  setSelectedModel(e.target.value);
                  if (prediction) setPrediction(null);
                }}
                className="w-full px-3.5 py-2.5 bg-gray-50 border border-gray-300 rounded-md text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-600 focus:bg-white"
              >
                {models.map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    {m.model_name} ({m.feature_group})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Quick Sample Selector */}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <span className="text-xs text-gray-500">Quick Samples:</span>
            {sampleProducts.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => {
                  setProductIdInput(p.id);
                  if (prediction) setPrediction(null);
                  if (error) setError(null);
                }}
                className="text-xs px-2.5 py-1 rounded bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors font-mono"
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Submit Button */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={loading}
              className="w-full sm:w-auto px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-medium text-sm rounded-md transition-colors flex items-center justify-center gap-2"
            >
              <Search size={16} />
              <span>{loading ? 'Analyzing Product...' : 'Check Product'}</span>
            </button>
          </div>
        </form>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-lg text-xs flex items-start gap-2">
          <AlertTriangle size={16} className="shrink-0 mt-0.5 text-red-600" />
          <span>{error}</span>
        </div>
      )}

      {/* Prediction Output Section */}
      {prediction && (
        <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm space-y-6">
          <div className="border-b border-gray-100 pb-3 flex items-center justify-between">
            <div>
              <span className="text-xs font-mono text-gray-500 uppercase tracking-wider">Analysis Result</span>
              <h3 className="text-lg font-bold text-gray-900 mt-0.5">
                Product ID #{prediction.product_ID || productIdInput}
              </h3>
            </div>
            <span className="text-xs text-gray-500 font-mono">
              Model: {prediction.model_name}
            </span>
          </div>

          {/* 4 Core Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Predicted Class */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <span className="text-xs text-gray-500 uppercase font-semibold">Predicted Class</span>
              <div className="mt-2">
                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-bold uppercase ${
                    prediction.is_fake_prediction
                      ? 'bg-red-100 text-red-800 border border-red-200'
                      : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                  }`}
                >
                  {prediction.is_fake_prediction ? (
                    <>
                      <AlertTriangle size={13} />
                      Fake Review Manipulation
                    </>
                  ) : (
                    <>
                      <CheckCircle size={13} />
                      Genuine Review Profile
                    </>
                  )}
                </span>
              </div>
            </div>

            {/* Fake Probability */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <span className="text-xs text-gray-500 uppercase font-semibold">Fake Probability</span>
              <div className="mt-2">
                <span className="text-2xl font-bold font-mono text-gray-900">
                  {(prediction.p_fake * 100).toFixed(1)}%
                </span>
                <span className="text-xs text-gray-500 block mt-0.5">
                  {prediction.probability_type} likelihood
                </span>
              </div>
            </div>

            {/* Risk Score out of 100 */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <span className="text-xs text-gray-500 uppercase font-semibold">Risk Score</span>
              <div className="mt-2">
                <span className="text-2xl font-bold font-mono text-blue-600">
                  {prediction.risk_score} <span className="text-xs font-normal text-gray-500">/ 100</span>
                </span>
                <span className="text-xs text-gray-500 block mt-0.5">Continuous risk index</span>
              </div>
            </div>

            {/* Risk Category */}
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
              <span className="text-xs text-gray-500 uppercase font-semibold">Risk Category</span>
              <div className="mt-2">
                <span
                  className={`inline-block px-2.5 py-1 rounded text-xs font-bold font-mono uppercase ${
                    prediction.risk_tier === 'HIGH_RISK'
                      ? 'bg-red-100 text-red-800 border border-red-200'
                      : prediction.risk_tier === 'MODERATE_RISK'
                      ? 'bg-amber-100 text-amber-800 border border-amber-200'
                      : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                  }`}
                >
                  {prediction.risk_tier.replace('_', ' ')}
                </span>
                <span className="text-xs text-gray-500 block mt-0.5">
                  Rating: {prediction.trust_adjusted_rating.toFixed(2)} ★ (adjusted)
                </span>
              </div>
            </div>
          </div>

          {/* English Explanation */}
          {prediction.explanation_english && (
            <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg text-xs text-gray-700 leading-relaxed space-y-1">
              <span className="font-semibold text-gray-900 block">Model Explanation:</span>
              <p>{prediction.explanation_english}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
