import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import { DashboardPage } from '../pages/Dashboard';
import { ProductCheckerPage } from '../pages/ProductChecker';
import { ModelComparisonPage } from '../pages/ModelComparison';
import { ClusteringPage } from '../pages/Clustering';
import { AppProvider, useApp } from '../context/AppContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as api from '../services/api';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <AppProvider>{children}</AppProvider>
    </QueryClientProvider>
  );
};

// Component to toggle mode programmatically
const ModeSwitcher: React.FC = () => {
  const { mode, setMode } = useApp();
  return (
    <div>
      <span data-testid="mode-display">{mode}</span>
      <button data-testid="set-demo-btn" onClick={() => setMode('demo')}>
        Demo
      </button>
      <button data-testid="set-research-btn" onClick={() => setMode('research')}>
        Research
      </button>
    </div>
  );
};

describe('Frontend Mode Switching, Race Conditions & Fallback Removal', () => {
  it('Dashboard displays error state and no static mock fallback when API fails', async () => {
    const spy = vi.spyOn(api, 'getDashboardOverview').mockRejectedValueOnce(new Error('Network failure 500'));

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <DashboardPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/FastAPI Backend Disconnected/i)).toBeInTheDocument();
    });

    // Verify hardcoded fake fallback F1 of 0.8745 is NOT present
    expect(screen.queryByText(/0.8745/)).not.toBeInTheDocument();
    spy.mockRestore();
  });

  it('Dashboard renders exact dynamic F1 and metrics when API succeeds', async () => {
    const mockOverview = {
      status: 'ok',
      dataset_statistics: {
        total_products: 4500,
        fake_products_count: 1800,
        genuine_products_count: 2700,
        fake_prevalence_pct: 40.0,
        features_count: 28,
      },
      model_statistics: {
        registered_models_count: 75,
        best_validated_model: 'custom_catboost_multimodal',
        best_fake_f1: 0.8345,
        best_roc_auc: 0.9234,
      },
      feature_groups: [],
      feature_group_progression: [
        { group: 'Review Only', f1: 0.7600, auc: 0.8600, features: 12, protocol: 'Holdout' },
        { group: 'Review + Network', f1: 0.8100, auc: 0.9000, features: 16, protocol: 'Holdout' },
      ],
      top_models: [
        { model: 'Custom CatBoost', f1: 0.8345, auc: 0.9234, type: 'Final Holdout Test', accuracy: 0.8500 },
      ],
      protocol_summary: {},
      clustering_summary: { optimal_k: 2, optimal_silhouette: 0.2474 },
      shap_summary: { network_pct: 35.42 },
      recent_experiments: [],
      system_status: { api_version: '2.1.0', environment: 'test', models_loaded: 75, timestamp: '' },
    };

    const spy = vi.spyOn(api, 'getDashboardOverview').mockResolvedValueOnce(mockOverview as any);

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <DashboardPage />
      </Wrapper>
    );

    await waitFor(() => {
      const elements = screen.getAllByText(/custom_catboost_multimodal/i);
      expect(elements.length).toBeGreaterThan(0);
    });

    expect(screen.getByText('0.8345')).toBeInTheDocument();
    expect(screen.getByText('4,500')).toBeInTheDocument();
    spy.mockRestore();
  });

  it('Model Comparison displays dashes "—" when confusion matrix is missing for CV rows, and exact counts when present', async () => {
    const mockRecords = [
      {
        experiment_id: 'EXP-CV-01',
        model: 'LightGBM CV Model',
        feature_group: 'review_network_image',
        fake_f1: 0.8150,
        roc_auc: 0.9120,
        accuracy: 0.8390,
        tp: null,
        tn: null,
        fp: null,
        fn: null,
        evaluation_protocol: 'Development cross-validation',
      },
      {
        experiment_id: 'EXP-HOLD-02',
        model: 'XGBoost Holdout Model',
        feature_group: 'review_network_image',
        fake_f1: 0.8179,
        roc_auc: 0.9199,
        accuracy: 0.8408,
        tp: 227,
        tn: 342,
        fp: 45,
        fn: 68,
        evaluation_protocol: 'Final independent test (N=682)',
      },
    ];

    const spy = vi.spyOn(api, 'getModelMetrics').mockResolvedValue(mockRecords as any);

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ModelComparisonPage />
      </Wrapper>
    );

    // Initial selected row is EXP-CV-01 (first record with null matrix)
    await waitFor(() => {
      const items = screen.getAllByText(/LightGBM CV Model/i);
      expect(items.length).toBeGreaterThan(0);
    });

    // Check that dashed fallbacks "—" are displayed in the matrix inspector, NOT hardcoded 342/45
    const dashes = screen.getAllByText('—');
    expect(dashes.length).toBeGreaterThanOrEqual(4);

    // Click second model row in table to inspect exact confusion matrix counts
    const row2 = screen.getByText('Final independent test (N=682)');
    fireEvent.click(row2);

    await waitFor(() => {
      expect(screen.getByText('342')).toBeInTheDocument();
      expect(screen.getByText('227')).toBeInTheDocument();
      expect(screen.getByText('45')).toBeInTheDocument();
      expect(screen.getByText('68')).toBeInTheDocument();
    });

    spy.mockRestore();
  });

  it('Clustering Page renders dynamic metrics without hardcoded static profiles', async () => {
    const mockClustering = {
      sweep_metrics: [
        { k: 2, inertia: 15000, silhouette: 0.2850, davies_bouldin: 1.30, calinski_harabasz: 1100 },
      ],
      k2_profiles: [
        {
          cluster_ID: 0,
          n_products: 2500,
          share_of_total: 0.65,
          ground_truth_fake_prevalence: 0.25,
          ground_truth_fake_count: 625,
          mean_p_fake_predicted: 0.24,
          pct_predicted_fake_threshold_05: 0.22,
          pct_predicted_fake_threshold_07: 0.18,
          mean_avg_review_rating: 4.05,
          mean_n_of_reviews: 180,
          mean_clustering_coef: 0.22,
          mean_w_degree: 52.4,
          ground_truth_genuine_prevalence: 0.75,
        },
        {
          cluster_ID: 1,
          n_products: 1350,
          share_of_total: 0.35,
          ground_truth_fake_prevalence: 0.88,
          ground_truth_fake_count: 1188,
          mean_p_fake_predicted: 0.85,
          pct_predicted_fake_threshold_05: 0.87,
          pct_predicted_fake_threshold_07: 0.82,
          mean_avg_review_rating: 4.45,
          mean_n_of_reviews: 820,
          mean_clustering_coef: 0.41,
          mean_w_degree: 712.5,
          ground_truth_genuine_prevalence: 0.12,
        },
      ],
      k20_profiles: [],
      optimal_k: 2,
      optimal_silhouette: 0.2850,
    };

    const spy = vi.spyOn(api, 'getClusteringDetails').mockResolvedValue(mockClustering as any);

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ClusteringPage />
      </Wrapper>
    );

    await waitFor(() => {
      const silScores = screen.getAllByText(/0.2850/);
      expect(silScores.length).toBeGreaterThan(0);
      const fakePrev = screen.getAllByText(/88.0%/);
      expect(fakePrev.length).toBeGreaterThan(0);
      const degree = screen.getAllByText(/712.5/);
      expect(degree.length).toBeGreaterThan(0);
    });

    spy.mockRestore();
  });

  it('Product Checker race condition test: newer product query overrides older delayed query', async () => {
    let resolveQuery101: ((val: any) => void) | null = null;
    let resolveQuery202: ((val: any) => void) | null = null;

    const spyList = vi.spyOn(api, 'listProducts').mockResolvedValue({
      total: 2,
      page: 1,
      page_size: 20,
      items: [
        { product_ID: 101, avg_review_rating: 4.5, clustering_coef: 0.1, w_degree: 50, pagerank: 0.0001, n_of_reviews: 100, rating_entropy: 0.5, review_burstiness: 0.2 },
        { product_ID: 202, avg_review_rating: 4.8, clustering_coef: 0.4, w_degree: 600, pagerank: 0.0008, n_of_reviews: 1200, rating_entropy: 0.3, review_burstiness: 0.9 },
      ],
    } as any);

    const spyDetails = vi.spyOn(api, 'getProductDetails').mockImplementation(((productId: number) => {
      if (productId === 101) {
        return new Promise((resolve) => {
          resolveQuery101 = resolve;
        });
      }
      if (productId === 202) {
        return new Promise((resolve) => {
          resolveQuery202 = resolve;
        });
      }
      return Promise.resolve({} as any);
    }) as any);

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ProductCheckerPage />
      </Wrapper>
    );

    // Initial mount auto-selects first item (101) and initiates getProductDetails(101)
    await waitFor(() => {
      expect(screen.getByText('ID: 202')).toBeInTheDocument();
    });

    // Rapidly click second catalog card (202) before 101 finishes
    const btn202 = screen.getByText('ID: 202');
    fireEvent.click(btn202);

    // Now resolve query 202 first
    act(() => {
      if (resolveQuery202) {
        resolveQuery202({
          product_ID: 202,
          displayed_rating: 4.8,
          features: { clustering_coef: 0.42 },
        });
      }
    });

    // Check that Product 202 is loaded and active
    await waitFor(() => {
      expect(screen.getByDisplayValue('202')).toBeInTheDocument();
      expect(screen.getByDisplayValue('0.42')).toBeInTheDocument();
    });

    // Now resolve query 101 late
    act(() => {
      if (resolveQuery101) {
        resolveQuery101({
          product_ID: 101,
          displayed_rating: 4.5,
          features: { clustering_coef: 0.11 },
        });
      }
    });

    // Product 202 (value 202, clustering_coef 0.42) must STILL remain active and NOT overwritten by Product 101
    expect(screen.getByDisplayValue('202')).toBeInTheDocument();
    expect(screen.getByDisplayValue('0.42')).toBeInTheDocument();
    expect(screen.queryByDisplayValue('101')).not.toBeInTheDocument();
    expect(screen.queryByDisplayValue('0.11')).not.toBeInTheDocument();

    spyList.mockRestore();
    spyDetails.mockRestore();
  });

  it('Mode Switch: Demo mode returns demonstration fixtures tagged with [DEMO]', async () => {
    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ModeSwitcher />
      </Wrapper>
    );

    expect(screen.getByTestId('mode-display').textContent).toBe('research');

    // Switch to demo mode
    fireEvent.click(screen.getByTestId('set-demo-btn'));
    expect(screen.getByTestId('mode-display').textContent).toBe('demo');

    // Call getDashboardOverview with mode='demo'
    const demoOverview = await api.getDashboardOverview('demo');
    expect(demoOverview.model_statistics.best_validated_model).toContain('[DEMO]');
    expect(demoOverview.top_models?.[0]?.model).toContain('[DEMO]');
  });
});
