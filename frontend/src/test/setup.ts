import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock ResizeObserver for Recharts and React Flow
class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

window.ResizeObserver = MockResizeObserver as any;

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
});

// Mock window.print
window.print = () => {};

// Global fetch mock to prevent Node 24 Undici RequestInit / AbortSignal issues in jsdom
globalThis.fetch = vi.fn().mockImplementation((url: string) => {
  if (url.includes('/health')) {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ status: 'ok', version: '2.1.0', timestamp: '2026-11-02T00:00:00Z' }),
    });
  }
  if (url.includes('/models/registry')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve([
          {
            model_name: 'calibrated_review_network_image_xgboost',
            filename: 'calibrated_review_network_image_xgboost.joblib',
            feature_group: 'review_network_image',
            required_features_count: 28,
            required_features: ['avg_review_rating', 'clustering_coef'],
            description: 'Calibrated Multimodal XGBoost Classifier',
          },
        ]),
    });
  }
  if (url.includes('/dashboard/overview')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          status: 'ok',
          dataset_statistics: {
            total_products: 3327,
            fake_products_count: 1400,
            genuine_products_count: 1927,
            fake_prevalence_pct: 42.08,
            features_count: 28,
          },
          model_statistics: {
            registered_models_count: 21,
            best_validated_model: 'lightgbm (review_network_image)',
            best_fake_f1: 0.8193,
            best_roc_auc: 0.9191,
          },
          feature_groups: [],
          feature_group_progression: [
            { group: 'Review Only', f1: 0.7673, auc: 0.8774, features: 12 },
            { group: 'Multimodal (All)', f1: 0.8179, auc: 0.9199, features: 28 },
          ],
          top_models: [
            { model: 'LightGBM (review_network_image)', f1: 0.8193, auc: 0.9191, type: 'Final Holdout Test' },
            { model: 'XGBoost (review_network_image)', f1: 0.8179, auc: 0.9199, type: 'Final Holdout Test' },
          ],
          recent_experiments: [],
          system_status: {
            api_version: '2.1.0',
            environment: 'test',
            models_loaded: 21,
            timestamp: '2026-11-02T00:00:00Z',
          },
        }),
    });
  }
  if (url.includes('/explainability/global')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          features: [
            { feature: 'clustering_coef', mean_abs_shap: 0.42, percent_importance: 16.07, modality: 'network' },
            { feature: 'avg_review_rating', mean_abs_shap: 0.35, percent_importance: 12.5, modality: 'review' },
          ],
          modality_breakdown: { network: 35.42, review: 41.98, image: 22.61 },
          total_features: 28,
          top_feature: 'clustering_coef',
        }),
    });
  }
  if (url.includes('/models/metrics')) {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve([]),
    });
  }
  if (url.includes('/products')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          total: 1,
          page: 1,
          page_size: 20,
          items: [
            {
              product_ID: 1080,
              avg_review_rating: 4.8,
              clustering_coef: 0.4,
              w_degree: 200,
              pagerank: 0.0005,
              n_of_reviews: 150,
              rating_entropy: 0.4,
              review_burstiness: 0.6,
            },
          ],
        }),
    });
  }
  if (url.includes('/clustering/details')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          sweep_metrics: [
            { k: 2, inertia: 18500, silhouette: 0.2474, davies_bouldin: 1.45, calinski_harabasz: 980 },
          ],
          k2_profiles: [
            {
              cluster_ID: 0,
              n_products: 2672,
              share_of_total: 0.803,
              ground_truth_fake_prevalence: 0.3462,
              ground_truth_fake_count: 925,
              mean_p_fake_predicted: 0.34,
              pct_predicted_fake_threshold_05: 0.33,
              pct_predicted_fake_threshold_07: 0.28,
              mean_avg_review_rating: 4.12,
              mean_n_of_reviews: 245,
              mean_clustering_coef: 0.30,
              mean_w_degree: 74.0,
              ground_truth_genuine_prevalence: 0.6538,
            },
            {
              cluster_ID: 1,
              n_products: 655,
              share_of_total: 0.197,
              ground_truth_fake_prevalence: 0.7924,
              ground_truth_fake_count: 519,
              mean_p_fake_predicted: 0.82,
              pct_predicted_fake_threshold_05: 0.84,
              pct_predicted_fake_threshold_07: 0.79,
              mean_avg_review_rating: 4.41,
              mean_n_of_reviews: 984,
              mean_clustering_coef: 0.378,
              mean_w_degree: 659.2,
              ground_truth_genuine_prevalence: 0.2076,
            },
          ],
          k20_profiles: [],
          optimal_k: 2,
          optimal_silhouette: 0.2474,
        }),
    });
  }
  if (url.includes('/network/summary')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          benchmarks: [],
          ring_evaluations: [],
          node_features: ['pagerank', 'w_degree', 'clustering_coef'],
          real_network_available: false,
          real_network_note: 'Test network',
        }),
    });
  }
  if (url.includes('/network/graph')) {
    return Promise.resolve({
      ok: true,
      json: () =>
        Promise.resolve({
          nodes: [],
          edges: [],
          stats: { total_nodes: 0, total_edges: 0, reviewer_nodes: 0, product_nodes: 0, ring_reviewers: 0, target_products: 0 },
          mode: 'test',
        }),
    });
  }
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
  });
}) as any;
