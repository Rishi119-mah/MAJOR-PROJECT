/**
 * Centralized API Client for Fake Review Detection Backend.
 * Communicates with FastAPI on VITE_API_BASE_URL (default http://localhost:8000).
 */

import {
  DashboardOverview,
  ModelRegistryItem,
  PredictionPayload,
  PredictionResponse,
  ProductListResponse,
  ExperimentListResponse,
  ExperimentRecord,
  GlobalShapResponse,
  ClusteringDetailsResponse,
  NetworkSummaryResponse,
  NetworkGraphResponse,
  HealthResponse,
  SystemEnvironmentResponse,
  SystemTestReportResponse,
  DatasetsSummaryResponse,
  AppMode,
} from '../types';
import {
  DEMO_DASHBOARD_OVERVIEW,
  DEMO_PRODUCTS,
  DEMO_PREDICTIONS,
  DEMO_MODEL_METRICS,
  DEMO_GLOBAL_SHAP,
  DEMO_CLUSTERING_DETAILS,
  DEMO_NETWORK_SUMMARY,
  DEMO_NETWORK_GRAPH,
  DEMO_SYSTEM_ENVIRONMENT,
  DEMO_SYSTEM_TEST_REPORT,
  DEMO_DATASETS_SUMMARY,
} from './demoFixtures';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers || {}),
      },
    });

    if (!response.ok) {
      let errorDetail = `HTTP ${response.status} ${response.statusText}`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        // Fallback to status text
      }
      throw new Error(errorDetail);
    }

    return (await response.json()) as T;
  } catch (error: any) {
    console.error(`API Error on ${endpoint}:`, error);
    throw error;
  }
}

// ──────────────────────────────────────────────
// API Endpoints with Explicit Demo / Research Data Isolation
// ──────────────────────────────────────────────

export async function checkApiHealth(): Promise<HealthResponse> {
  return fetchJson<HealthResponse>('/health');
}

export async function getDashboardOverview(mode: AppMode = 'research'): Promise<DashboardOverview> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_DASHBOARD_OVERVIEW);
  }
  return fetchJson<DashboardOverview>('/dashboard/overview');
}

export async function getModelRegistry(mode: AppMode = 'research'): Promise<ModelRegistryItem[]> {
  if (mode === 'demo') {
    return Promise.resolve([
      {
        model_name: '[DEMO] synthetic_xgboost',
        filename: 'demo_synthetic_xgboost.joblib',
        feature_group: 'review_network_image',
        required_features_count: 28,
        required_features: ['avg_review_rating', 'clustering_coef'],
        description: 'Demonstration synthetic XGBoost model',
      },
      {
        model_name: '[DEMO] synthetic_lightgbm',
        filename: 'demo_synthetic_lightgbm.joblib',
        feature_group: 'review_network_image',
        required_features_count: 28,
        required_features: ['avg_review_rating', 'clustering_coef'],
        description: 'Demonstration synthetic LightGBM model',
      },
    ]);
  }
  return fetchJson<ModelRegistryItem[]>('/models/registry');
}

export async function getModelMetrics(
  modelName?: string,
  featureGroup?: string,
  mode: AppMode = 'research'
): Promise<any[]> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_MODEL_METRICS);
  }
  const params = new URLSearchParams();
  if (modelName) params.append('model_name', modelName);
  if (featureGroup) params.append('feature_group', featureGroup);
  const qs = params.toString() ? `?${params.toString()}` : '';
  return fetchJson<any[]>(`/models/metrics${qs}`);
}

export async function predictProduct(
  payload: PredictionPayload,
  mode: AppMode = 'research'
): Promise<PredictionResponse> {
  if (mode === 'demo') {
    const id = payload.product_ID || 9001;
    if (DEMO_PREDICTIONS[id]) {
      return Promise.resolve(DEMO_PREDICTIONS[id]);
    }
    // Generate synthetic response for demo mode
    return Promise.resolve({
      product_ID: id,
      trust_adjusted_rating: 3.75,
      p_fake: 0.65,
      risk_score: 65.0,
      risk_tier: 'HIGH_RISK',
      probability_type: 'calibrated',
      calibration_method: 'isotonic',
      model_name: '[DEMO] synthetic_xgboost',
      model_version: '2.1.0-demo',
      is_fake_prediction: true,
      explanation_english: '[DEMO] Demonstration prediction generated from synthetic fixtures.',
      explanation_hindi: '[DEMO] सिंथेटिक फिक्स्चर से उत्पन्न प्रदर्शन भविष्यवाणी।',
      explanation_hinglish: '[DEMO] Synthetic demo fixtures se prediction calculate kiya gaya.',
    });
  }
  return fetchJson<PredictionResponse>('/predict/product', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function getProductDetails(
  productId: number,
  signal?: AbortSignal,
  mode: AppMode = 'research'
): Promise<any> {
  if (mode === 'demo') {
    const match = DEMO_PRODUCTS.items.find((p) => p.product_ID === productId);
    if (!match) {
      throw new Error(`[DEMO] Product ID ${productId} not found in demonstration catalog.`);
    }
    const feats: Record<string, number> = {};
    Object.keys(match).forEach((k) => {
      if (k !== 'product_ID' && typeof match[k] === 'number') {
        feats[k] = match[k];
      }
    });
    return Promise.resolve({
      product_ID: productId,
      displayed_rating: match.avg_review_rating || 4.2,
      risk_score: match.clustering_coef > 0.3 ? 85.0 : 15.0,
      risk_tier: match.clustering_coef > 0.3 ? 'HIGH_RISK' : 'LOW_RISK',
      trust_adjusted_rating: match.clustering_coef > 0.3 ? 3.65 : 4.10,
      features: feats,
      is_demo_fixture: true,
    });
  }
  return fetchJson<any>(`/products/${productId}`, { signal });
}

export async function listProducts(
  page = 1,
  pageSize = 20,
  search?: string,
  mode: AppMode = 'research'
): Promise<ProductListResponse> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_PRODUCTS);
  }
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (search) params.append('search', search);
  return fetchJson<ProductListResponse>(`/products?${params.toString()}`);
}

export async function getExperiments(
  evaluationType?: string,
  featureGroup?: string,
  search?: string,
  page = 1,
  pageSize = 50,
  mode: AppMode = 'research'
): Promise<ExperimentListResponse> {
  if (mode === 'demo') {
    return Promise.resolve({
      total: DEMO_DASHBOARD_OVERVIEW.recent_experiments.length,
      page: 1,
      page_size: pageSize,
      records: DEMO_DASHBOARD_OVERVIEW.recent_experiments,
    });
  }
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (evaluationType && evaluationType !== 'All') params.append('evaluation_type', evaluationType);
  if (featureGroup && featureGroup !== 'All') params.append('feature_group', featureGroup);
  if (search) params.append('search', search);

  return fetchJson<ExperimentListResponse>(`/experiments?${params.toString()}`);
}

export async function getExperimentDetail(experimentId: string, mode: AppMode = 'research'): Promise<ExperimentRecord> {
  return fetchJson<ExperimentRecord>(`/experiments/${encodeURIComponent(experimentId)}`);
}

export async function getGlobalShap(mode: AppMode = 'research'): Promise<GlobalShapResponse> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_GLOBAL_SHAP);
  }
  return fetchJson<GlobalShapResponse>('/explainability/global');
}

export async function getClusteringDetails(mode: AppMode = 'research'): Promise<ClusteringDetailsResponse> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_CLUSTERING_DETAILS);
  }
  return fetchJson<ClusteringDetailsResponse>('/clustering/details');
}

export async function getClusteringSummary(mode: AppMode = 'research'): Promise<any[]> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_CLUSTERING_DETAILS.sweep_metrics);
  }
  return fetchJson<any[]>('/clustering/summary');
}

export async function getNetworkSummary(mode: AppMode = 'research'): Promise<NetworkSummaryResponse> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_NETWORK_SUMMARY);
  }
  return fetchJson<NetworkSummaryResponse>('/network/summary');
}

export async function getNetworkGraph(noiseLevel = 0.1, mode: AppMode = 'research'): Promise<NetworkGraphResponse> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_NETWORK_GRAPH);
  }
  return fetchJson<NetworkGraphResponse>(`/network/graph?noise_level=${noiseLevel}`);
}

export async function getDatasetsSummary(mode: AppMode = 'research'): Promise<DatasetsSummaryResponse> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_DATASETS_SUMMARY);
  }
  return fetchJson<DatasetsSummaryResponse>('/datasets/summary');
}

export async function getSystemEnvironment(mode: AppMode = 'research'): Promise<SystemEnvironmentResponse> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_SYSTEM_ENVIRONMENT);
  }
  return fetchJson<SystemEnvironmentResponse>('/system/environment');
}

export async function getSystemTestReport(mode: AppMode = 'research'): Promise<SystemTestReportResponse> {
  if (mode === 'demo') {
    return Promise.resolve(DEMO_SYSTEM_TEST_REPORT);
  }
  return fetchJson<SystemTestReportResponse>('/system/test-report');
}


