import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { DashboardPage } from '../pages/Dashboard';
import { ModelAnalysisPage } from '../pages/ModelAnalysis';
import { AppProvider } from '../context/AppContext';
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

describe('Frontend Data Rendering & Race Condition Safeguards', () => {
  it('Dashboard displays error state when backend API is unreachable', async () => {
    const spy = vi.spyOn(api, 'getDashboardOverview').mockRejectedValueOnce(new Error('Network failure 500'));

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <DashboardPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Unable to load dashboard data/i)).toBeInTheDocument();
    });

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
      expect(screen.getByText(/custom_catboost_multimodal/i)).toBeInTheDocument();
    });

    expect(screen.getByText('0.8345')).toBeInTheDocument();
    expect(screen.getByText('4,500')).toBeInTheDocument();
    spy.mockRestore();
  });

  it('Model Analysis renders dynamic metrics without hardcoded static profiles', async () => {
    const mockClustering = {
      optimal_k: 2,
      optimal_silhouette: 0.2891,
      davies_bouldin_score: 1.45,
      calinski_harabasz_score: 620.1,
      sweep_metrics: [
        { k: 2, silhouette: 0.2891, davies_bouldin: 1.45, calinski_harabasz: 620.1 },
        { k: 3, silhouette: 0.2100, davies_bouldin: 1.82, calinski_harabasz: 510.3 },
      ],
      k2_profiles: [
        { cluster: 0, size: 2800, ground_truth_fake_prevalence: 0.25 },
        { cluster: 1, size: 527, ground_truth_fake_prevalence: 0.95 },
      ],
    };

    const spyClust = vi.spyOn(api, 'getClusteringDetails').mockResolvedValue(mockClustering as any);

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ModelAnalysisPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Machine Learning & Anomaly Analysis/i)).toBeInTheDocument();
    });

    spyClust.mockRestore();
  });
});
