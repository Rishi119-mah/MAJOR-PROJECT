import React, { useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import {
  getModelRegistry,
  predictProduct,
  listProducts,
  getProductDetails,
} from '../../services/api';
import { ModelRegistryItem, PredictionResponse, ProductCatalogItem } from '../../types';
import { CircularRiskScore } from '../../components/common/CircularRiskScore';
import { RiskBadge } from '../../components/common/RiskBadge';
import {
  ShieldAlert,
  Search,
  Sparkles,
  Layers,
  ArrowRight,
  Printer,
  Download,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Sliders,
  Info,
} from 'lucide-react';

export const ProductCheckerPage: React.FC = () => {
  const { t, selectedModel, setSelectedModel, language, mode } = useApp();

  const [models, setModels] = useState<ModelRegistryItem[]>([]);
  const [activeTab, setActiveTab] = useState<'catalog' | 'manual'>('catalog');
  const [catalogProducts, setCatalogProducts] = useState<any[]>([]);
  const [searchId, setSearchId] = useState<string>('1080');
  const [loadedProductId, setLoadedProductId] = useState<string | null>(null);
  const [fetchingProduct, setFetchingProduct] = useState<boolean>(false);
  const [displayedRating, setDisplayedRating] = useState<number>(4.8);
  const [featureInputs, setFeatureInputs] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Request sequence and cancellation refs for race-condition prevention
  const activeRequestIdRef = React.useRef<number>(0);
  const abortControllerRef = React.useRef<AbortController | null>(null);

  // Load registered models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const list = await getModelRegistry(mode);
        if (Array.isArray(list)) {
          setModels(list);
          if (list.length > 0 && !list.some((m) => m.model_name === selectedModel)) {
            setSelectedModel(list[0].model_name);
          }
        }
      } catch (e) {
        console.warn('Failed to load model registry:', e);
      }
    };
    fetchModels();
  }, [mode]);

  // Fetch product features from backend API with race-condition prevention
  const handleSearchIdChange = (newVal: string) => {
    setSearchId(newVal);
    if (activeTab === 'catalog') {
      // Immediate full state invalidation on product ID edit
      setFeatureInputs({});
      setPrediction(null);
      setLoadedProductId(null);
      setError(null);
    }
  };

  const fetchProductById = async (id: string | number) => {
    const numId = Number(id);
    if (isNaN(numId) || numId <= 0) {
      setError('Please provide a valid numeric product ID.');
      setLoadedProductId(null);
      setFeatureInputs({});
      setPrediction(null);
      return;
    }

    // Abort prior in-flight request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;
    const reqId = ++activeRequestIdRef.current;

    setFetchingProduct(true);
    setError(null);
    setPrediction(null);
    setFeatureInputs({});
    setLoadedProductId(null);

    try {
      const details = await getProductDetails(numId, controller.signal, mode);
      // Discard stale response if a newer selection has occurred or ID mismatch
      if (reqId !== activeRequestIdRef.current) {
        return;
      }
      if (String(details.product_ID) !== String(numId)) {
        return;
      }
      setSearchId(String(details.product_ID));
      setLoadedProductId(String(details.product_ID));
      setDisplayedRating(details.displayed_rating || 4.2);
      setFeatureInputs(details.features || {});
    } catch (err: any) {
      if (err.name === 'AbortError' || reqId !== activeRequestIdRef.current) {
        return; // Silent discard for superseded request
      }
      setError(`Product ID ${id} not found in catalog. ${err.message || ''}`);
      setLoadedProductId(null);
      setFeatureInputs({});
    } finally {
      if (reqId === activeRequestIdRef.current) {
        setFetchingProduct(false);
      }
    }
  };

  // Load catalog product samples for instant selection
  useEffect(() => {
    const fetchCatalog = async () => {
      setPrediction(null);
      setError(null);
      try {
        const res = await listProducts(1, 15, undefined, mode);
        setCatalogProducts(res.items);
        if (res.items.length > 0) {
          fetchProductById(res.items[0].product_ID);
        }
      } catch (e) {
        console.warn('Failed to load catalog products:', e);
      }
    };
    fetchCatalog();
  }, [mode]);

  const selectCatalogProduct = (product: any) => {
    const pid = String(product.product_ID);
    setSearchId(pid);
    setFeatureInputs({});
    setPrediction(null);
    setLoadedProductId(null);
    setError(null);
    fetchProductById(product.product_ID);
  };

  const currentModel = Array.isArray(models)
    ? models.find((m) => m.model_name === selectedModel) || models[0]
    : undefined;

  // Strict pre-prediction validation guard
  const allRequiredFeaturesPresent = Boolean(
    currentModel &&
    currentModel.required_features.length > 0 &&
    currentModel.required_features.every(
      (feat) => featureInputs[feat] !== undefined && featureInputs[feat] !== null && !isNaN(Number(featureInputs[feat]))
    )
  );

  const canPredict = Boolean(
    !loading &&
    !fetchingProduct &&
    currentModel &&
    allRequiredFeaturesPresent &&
    (activeTab === 'manual' || (loadedProductId !== null && loadedProductId === searchId.trim()))
  );

  const handlePredict = async () => {
    if (!canPredict || !currentModel) return;

    // Stale feature retrieval safeguard: prevent submitting mismatching product features
    if (activeTab === 'catalog') {
      if (!loadedProductId || loadedProductId !== searchId.trim()) {
        setError(
          `Stale feature protection: Active features belong to product '${loadedProductId || 'none'}', but Product ID '${searchId}' was entered. Please click 'Fetch Product' to load its verified features, or switch to 'Manual Numeric Entry'.`
        );
        return;
      }
    }

    setLoading(true);
    setError(null);

    // Build payload strictly adhering to model's required features
    const payloadFeatures: Record<string, number> = {};
    const missing: string[] = [];

    currentModel.required_features.forEach((feat) => {
      if (featureInputs[feat] !== undefined && !isNaN(featureInputs[feat])) {
        payloadFeatures[feat] = Number(featureInputs[feat]);
      } else {
        missing.push(feat);
      }
    });

    if (missing.length > 0) {
      setError(`Missing ${missing.length} required features for '${currentModel.model_name}': ${missing.slice(0, 4).join(', ')}${missing.length > 4 ? '...' : ''}. Please supply all features.`);
      setLoading(false);
      return;
    }

    try {
      const expectedId = Number(searchId) || undefined;
      const res = await predictProduct({
        product_ID: expectedId,
        displayed_rating: displayedRating,
        model_name: currentModel.model_name,
        features: payloadFeatures,
      }, mode);

      if (expectedId !== undefined && res.product_ID !== undefined && res.product_ID !== expectedId) {
        setError(`Response mismatch: Expected prediction for product ${expectedId}, received product ${res.product_ID}.`);
        return;
      }

      setPrediction(res);
    } catch (err: any) {
      setError(err.message || 'Prediction failed. Verify input values.');
    } finally {
      setLoading(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  const handleExportJson = () => {
    if (!prediction) return;
    const blob = new Blob([JSON.stringify(prediction, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fraudshield_prediction_${prediction.product_ID || 'product'}.json`;
    a.click();
  };

  return (
    <div className="space-y-6">
      {/* Page Title & Intro */}
      <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">Product-Level Fake Review Risk Checker</h2>
          <p className="text-xs text-slate-500 mt-1">
            Analyze coordinated fake-review-buyer behavior using calibrated classifiers, bipartite graph centralities, and multimodal feature embeddings.
          </p>
        </div>

        {/* Model Selection Dropdown */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-semibold text-slate-600 uppercase tracking-wider shrink-0">Model:</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="text-xs font-mono font-medium bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <optgroup label="Calibrated Production Models (Recommended)">
              {models
                .filter((m) => m.model_name.startsWith('calibrated_'))
                .map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    ★ {m.model_name} ({m.required_features_count} feats)
                  </option>
                ))}
            </optgroup>
            <optgroup label="Tuned Multimodal (28 Features)">
              {models
                .filter((m) => m.model_name.startsWith('tuned_'))
                .map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    {m.model_name} ({m.required_features_count} feats)
                  </option>
                ))}
            </optgroup>
            <optgroup label="Review + Network (16 Features)">
              {models
                .filter((m) => m.model_name.startsWith('review_network_') && !m.model_name.includes('image'))
                .map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    {m.model_name} ({m.required_features_count} feats)
                  </option>
                ))}
            </optgroup>
            <optgroup label="Review Only (12 Features)">
              {models
                .filter((m) => m.model_name.startsWith('review_only_'))
                .map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    {m.model_name} ({m.required_features_count} feats)
                  </option>
                ))}
            </optgroup>
            <optgroup label="Network Only (4 Features)">
              {models
                .filter((m) => m.model_name.startsWith('network_only_'))
                .map((m) => (
                  <option key={m.model_name} value={m.model_name}>
                    {m.model_name} ({m.required_features_count} feats)
                  </option>
                ))}
            </optgroup>
          </select>
        </div>
      </div>

      {/* Main Grid: Input Form (Left) & Result Panel (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Input Form (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm space-y-4">
            {/* Tabs: Catalog vs Manual */}
            <div className="flex border-b border-slate-200">
              <button
                onClick={() => setActiveTab('catalog')}
                className={`flex-1 py-2 text-xs font-semibold border-b-2 transition-colors ${
                  activeTab === 'catalog'
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-slate-500 hover:text-slate-800'
                }`}
              >
                Catalog Lookup
              </button>
              <button
                onClick={() => setActiveTab('manual')}
                className={`flex-1 py-2 text-xs font-semibold border-b-2 transition-colors ${
                  activeTab === 'manual'
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-slate-500 hover:text-slate-800'
                }`}
              >
                Manual Numeric Entry
              </button>
            </div>

            {/* Catalog Samples Selector */}
            {activeTab === 'catalog' && (
              <div className="space-y-3">
                <label className="text-xs font-semibold text-slate-700 block">
                  Select Product from Catalog Samples:
                </label>
                <div className="grid grid-cols-3 gap-2 max-h-48 overflow-y-auto p-1 border border-slate-100 rounded-lg">
                  {catalogProducts.map((p) => (
                    <button
                      key={p.product_ID}
                      onClick={() => selectCatalogProduct(p)}
                      className={`p-2 rounded-lg text-left text-xs border transition-all ${
                        String(p.product_ID) === searchId
                          ? 'bg-indigo-50 border-indigo-300 font-bold text-indigo-900 shadow-xs'
                          : 'bg-white border-slate-200 hover:bg-slate-50 text-slate-700'
                      }`}
                    >
                      <div className="font-mono text-[11px] font-bold">ID: {p.product_ID}</div>
                      <div className="text-[10px] text-slate-500">★ {p.avg_review_rating?.toFixed(1) || '4.0'}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Product ID & Rating Inputs */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-medium text-slate-600">Product ID</label>
                  {loadedProductId && loadedProductId === searchId.trim() && (
                    <span className="text-[10px] text-emerald-700 font-mono font-medium">✓ Synced</span>
                  )}
                  {loadedProductId && loadedProductId !== searchId.trim() && (
                    <span className="text-[10px] text-amber-600 font-mono font-medium">⚠ Out of sync</span>
                  )}
                </div>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={searchId}
                    onChange={(e) => handleSearchIdChange(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        fetchProductById(searchId);
                      }
                    }}
                    placeholder="e.g. 1080"
                    className="flex-1 text-xs font-mono px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-slate-50"
                  />
                  <button
                    type="button"
                    onClick={() => fetchProductById(searchId)}
                    disabled={fetchingProduct}
                    className="px-2.5 py-2 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 rounded-lg text-xs font-medium transition-colors flex items-center gap-1 shrink-0"
                    title="Fetch product features from catalog database"
                  >
                    <Search size={13} className={fetchingProduct ? 'animate-spin' : ''} />
                    <span>{fetchingProduct ? 'Fetching...' : 'Fetch'}</span>
                  </button>
                </div>
              </div>
              <div>
                <label className="text-xs font-medium text-slate-600 block mb-1">Observed Rating (1.0 - 5.0)</label>
                <input
                  type="number"
                  step="0.1"
                  min="1.0"
                  max="5.0"
                  value={displayedRating}
                  onChange={(e) => setDisplayedRating(Number(e.target.value))}
                  className="w-full text-xs font-mono px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-slate-50"
                />
              </div>
            </div>

            {/* Model Feature Preview & Values */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                  Required Features ({currentModel?.required_features_count || 0})
                </label>
                <span className="text-[10px] text-slate-500">No zero-filling</span>
              </div>

              <div className="max-h-60 overflow-y-auto space-y-2 pr-1">
                {currentModel?.required_features.map((feat) => (
                  <div key={feat} className="flex items-center justify-between text-xs bg-slate-50 p-2 rounded-lg border border-slate-100">
                    <span className="font-mono text-[11px] text-slate-600 truncate max-w-[180px]" title={feat}>
                      {feat}
                    </span>
                    <input
                      type="number"
                      step="any"
                      value={featureInputs[feat] !== undefined ? featureInputs[feat] : ''}
                      onChange={(e) =>
                        setFeatureInputs({
                          ...featureInputs,
                          [feat]: parseFloat(e.target.value),
                        })
                      }
                      placeholder="0.0"
                      className="w-24 text-right text-xs font-mono px-2 py-1 border border-slate-200 rounded bg-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start gap-2">
                <AlertTriangle size={15} className="text-rose-600 shrink-0 mt-0.5" />
                <span className="leading-snug">{error}</span>
              </div>
            )}

            {/* Analyze Button */}
            <button
              onClick={handlePredict}
              disabled={!canPredict}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-medium text-xs shadow-sm shadow-indigo-600/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <RotateCcw size={14} className="animate-spin" />
                  <span>Evaluating Model Pipeline...</span>
                </>
              ) : (
                <>
                  <Sparkles size={14} />
                  <span>Analyze Product Fraud Risk</span>
                </>
              )}
            </button>

            {/* Validation helper status */}
            {!canPredict && !loading && (
              <div className="text-[11px] text-slate-500 bg-slate-50 p-2.5 rounded-lg border border-slate-200 space-y-1">
                {activeTab === 'catalog' && (!loadedProductId || loadedProductId !== searchId.trim()) && (
                  <p className="text-amber-700 font-medium">
                    ⚠ Active inputs do not match Product #{searchId}. Click &apos;Fetch&apos; to load verified catalog features.
                  </p>
                )}
                {!allRequiredFeaturesPresent && (
                  <p className="text-slate-600">
                    Enter valid numeric values for all {currentModel?.required_features_count || 'required'} model features to enable prediction.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Prediction Results & Explanations (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          {prediction ? (
            <div className="bg-white rounded-xl border border-slate-200/80 p-6 shadow-sm space-y-6">
              {/* Top Header of Result */}
              <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-lg font-bold text-slate-900">
                      Product #{prediction.product_ID || searchId} Evaluation
                    </h3>
                    <RiskBadge tier={prediction.risk_tier} score={prediction.risk_score} size="md" />
                  </div>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Evaluated via <code className="font-mono text-indigo-600 font-semibold">{prediction.model_name}</code> (v{prediction.model_version})
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={handleExportJson}
                    className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-600 text-xs flex items-center gap-1"
                    title="Export JSON"
                  >
                    <Download size={14} />
                    <span className="hidden sm:inline">JSON</span>
                  </button>
                  <button
                    onClick={handlePrint}
                    className="p-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-600 text-xs flex items-center gap-1"
                    title="Print Report"
                  >
                    <Printer size={14} />
                    <span className="hidden sm:inline">Print</span>
                  </button>
                </div>
              </div>

              {/* Core Metrics Visualizer */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                {/* Circular Gauge */}
                <div className="flex flex-col items-center justify-center p-4 bg-slate-50/70 rounded-xl border border-slate-100">
                  <CircularRiskScore score={prediction.risk_score} tier={prediction.risk_tier} size={170} />
                  <span className="text-[11px] text-slate-500 mt-2 font-medium">Continuous Fraud Risk Score (0-100)</span>
                </div>

                {/* Probability & Shrinkage Breakdown */}
                <div className="space-y-3 text-xs">
                  <div className="p-3 rounded-lg border border-slate-100 bg-slate-50/60">
                    <span className="text-slate-500 block text-[11px]">Calibrated Posterior Probability</span>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className="text-2xl font-bold font-mono text-slate-900">
                        {(prediction.p_fake * 100).toFixed(2)}%
                      </span>
                      <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 font-semibold">
                        {prediction.probability_type} • {prediction.calibration_method}
                      </span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg border border-slate-100 bg-slate-50/60">
                    <span className="text-slate-500 block text-[11px]">Bayesian Trust-Adjusted Rating Shrinkage</span>
                    <div className="flex items-baseline gap-3 mt-1">
                      <div className="text-slate-400 line-through text-sm font-mono font-medium">
                        ★ {displayedRating.toFixed(2)}
                      </div>
                      <ArrowRight size={14} className="text-slate-400" />
                      <div className="text-2xl font-bold font-mono text-indigo-600">
                        ★ {prediction.trust_adjusted_rating.toFixed(2)}
                      </div>
                    </div>
                    <span className="text-[10px] text-slate-400 mt-0.5 block">
                      Penalizes displayed rating proportional to fake review probability.
                    </span>
                  </div>
                </div>
              </div>

              {/* Multilingual Explanation Card */}
              <div className="space-y-3 pt-4 border-t border-slate-100">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkles size={14} className="text-indigo-600" />
                    Multilingual Explainability
                  </h4>
                  <span className="text-[10px] text-slate-400 font-mono">TreeSHAP Attribution</span>
                </div>

                <div className="space-y-2">
                  <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80">
                    <span className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider block mb-1">
                      English Explanation
                    </span>
                    <p className="text-xs text-slate-800 leading-relaxed font-sans">{prediction.explanation_english}</p>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80">
                    <span className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider block mb-1">
                      हिंदी व्याख्या (Hindi Explanation)
                    </span>
                    <p className="text-xs text-slate-800 leading-relaxed font-sans">{prediction.explanation_hindi}</p>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80">
                    <span className="text-[10px] font-bold text-indigo-600 uppercase tracking-wider block mb-1">
                      Hinglish Explanation
                    </span>
                    <p className="text-xs text-slate-800 leading-relaxed font-sans">{prediction.explanation_hinglish}</p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div className="bg-white rounded-xl border border-dashed border-slate-300 p-12 text-center flex flex-col items-center justify-center space-y-3 h-full min-h-[400px]">
              <div className="w-12 h-12 rounded-full bg-indigo-50 text-indigo-600 flex items-center justify-center">
                <ShieldAlert size={24} />
              </div>
              <h3 className="text-sm font-bold text-slate-900">No Product Evaluated Yet</h3>
              <p className="text-xs text-slate-500 max-w-sm">
                Select a product from the catalog on the left or enter manual features, then click &quot;Analyze Product Fraud Risk&quot; to view real-time risk scores and explanations.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

