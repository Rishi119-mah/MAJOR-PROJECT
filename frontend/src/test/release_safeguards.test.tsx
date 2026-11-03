import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { ProductCheckerPage } from '../pages/ProductChecker';
import { ReportsPage } from '../pages/Reports';
import { NetworkExplorerPage } from '../pages/NetworkExplorer';
import { ExperimentsPage } from '../pages/Experiments';
import { ModelComparisonPage } from '../pages/ModelComparison';
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

describe('Release Sign-off Safeguards & Fallback Elimination Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('ProductChecker: invalidates state on searchId change and disables analyze button', async () => {
    vi.spyOn(api, 'getModelRegistry').mockResolvedValue([
      {
        model_name: 'test_model',
        filename: 'test.joblib',
        feature_group: 'review_only',
        required_features_count: 2,
        required_features: ['mean_rating', 'review_count'],
        description: 'Test Model',
      },
    ]);

    vi.spyOn(api, 'listProducts').mockResolvedValue({
      total: 1,
      page: 1,
      page_size: 15,
      items: [{ product_ID: 1080, displayed_rating: 4.5, fake_f1: 0.8 }],
    });

    vi.spyOn(api, 'getProductDetails').mockResolvedValue({
      product_ID: 1080,
      displayed_rating: 4.5,
      features: { mean_rating: 4.5, review_count: 50 },
    });

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ProductCheckerPage />
      </Wrapper>
    );

    // Wait for initial product load
    await waitFor(() => {
      expect(screen.getByText(/✓ Synced/i)).toBeInTheDocument();
    });

    // Verify analyze button is enabled when synced
    const analyzeBtn = screen.getByRole('button', { name: /Analyze Product Fraud Risk/i });
    expect(analyzeBtn).not.toBeDisabled();

    // Now user types a new product ID in search input
    const input = screen.getByPlaceholderText('e.g. 1080');
    fireEvent.change(input, { target: { value: '9999' } });

    // State invalidation check: analyze button must be disabled immediately
    await waitFor(() => {
      expect(analyzeBtn).toBeDisabled();
      expect(screen.queryByText(/✓ Synced/i)).not.toBeInTheDocument();
      expect(screen.getByText(/⚠ Active inputs do not match Product #9999/i)).toBeInTheDocument();
    });
  });

  it('ReportsPage: handles missing test report and environment telemetry gracefully without static claims', async () => {
    vi.spyOn(api, 'getDashboardOverview').mockResolvedValue({
      status: 'ok',
      dataset_statistics: {
        total_products: 3327,
        fake_products_count: 852,
        genuine_products_count: 2475,
        fake_prevalence_pct: 25.61,
        features_count: 28,
      },
      model_statistics: {
        registered_models_count: 21,
        best_validated_model: 'lightgbm',
        best_fake_f1: 0.8193,
        best_roc_auc: 0.9191,
      },
      feature_groups: [],
      feature_group_progression: [],
      top_models: [],
      protocol_summary: {},
      clustering_summary: { optimal_k: 2, optimal_silhouette: 0.2474 },
      shap_summary: { network_pct: 35.4 },
      recent_experiments: [],
      system_status: { api_version: '2.1.0', environment: 'production', models_loaded: 21, timestamp: '' },
    });

    vi.spyOn(api, 'getModelMetrics').mockResolvedValue([]);
    vi.spyOn(api, 'getGlobalShap').mockResolvedValue({
      features: [],
      modality_breakdown: { network: 35.4, review: 42.1, image: 22.5 },
      total_features: 28,
      top_feature: 'clustering_coef',
    });
    vi.spyOn(api, 'getClusteringDetails').mockResolvedValue({
      sweep_metrics: [],
      k2_profiles: [],
      k20_profiles: [],
      optimal_k: null,
      optimal_silhouette: null,
    });

    // Backend returns not_available test report and null environment
    vi.spyOn(api, 'getSystemTestReport').mockResolvedValue({
      status: 'not_available',
      message: 'No verified test report available.',
      total_tests: null,
      passed: null,
      failed: null,
      skipped: null,
      test_execution_date: null,
      environment: null,
      test_command: null,
    });
    vi.spyOn(api, 'getSystemEnvironment').mockRejectedValue(new Error('Environment endpoint offline'));
    vi.spyOn(api, 'getDatasetsSummary').mockResolvedValue({
      labeled_products: 3327,
      features_count: 28,
      positive_class_count: 852,
      negative_class_count: 2475,
      fake_prevalence_pct: 25.61,
      missing_values_count: 0,
      missing_values_status: '0 (100% Clean)',
      development_split_count: 2661,
      holdout_split_count: 666,
      available_feature_groups: [],
    });

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ReportsPage />
      </Wrapper>
    );

    // Must display dynamic graceful empty states and NOT static hardcoded claims
    await waitFor(() => {
      expect(screen.getByText(/Automated Test Suite: No verified test report available/i)).toBeInTheDocument();
      expect(screen.getByText(/Environment telemetry: Not recorded/i)).toBeInTheDocument();
    });

    // Verify static "35/35 Tests Passing" is NOT rendered
    expect(screen.queryByText(/35\/35 Tests Passing/i)).not.toBeInTheDocument();
  });

  it('NetworkExplorer: displays empty state when graph benchmarks are unavailable', async () => {
    vi.spyOn(api, 'getNetworkSummary').mockResolvedValue({
      benchmarks: [],
      ring_evaluations: [],
      node_features: [],
      real_network_available: false,
      real_network_note: 'Not bundled',
    });

    vi.spyOn(api, 'getNetworkGraph').mockResolvedValue({
      nodes: [],
      edges: [],
      stats: { total_nodes: 0, total_edges: 0, reviewer_nodes: 0, product_nodes: 0, ring_reviewers: 0, target_products: 0 },
      mode: 'test',
    });

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <NetworkExplorerPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/No verified graph benchmark results available/i)).toBeInTheDocument();
    });

    // Verify hardcoded CatBoost fallback is NOT displayed
    expect(screen.queryByText(/Tabular CatBoost \(Centrality\)/i)).not.toBeInTheDocument();
  });

  it('ExperimentsPage: displays "Not recorded" for missing metadata instead of invented defaults', async () => {
    vi.spyOn(api, 'getExperiments').mockResolvedValue({
      total: 1,
      page: 1,
      page_size: 20,
      records: [
        {
          experiment_id: 'EXP_TEST_001',
          algorithm: 'xgboost',
          feature_group: 'review_only',
          split_protocol: undefined,
          random_seed: undefined,
          dataset_fingerprint: undefined,
          artifact_path: undefined,
          parameters: undefined,
          verification_status: undefined,
          fake_f1: 0.78,
          roc_auc: 0.88,
          evaluation_type: 'Holdout Test',
        },
      ],
    });

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ExperimentsPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('EXP_TEST_001')).toBeInTheDocument();
    });

    // Click to open audit view modal
    fireEvent.click(screen.getByText('EXP_TEST_001'));

    await waitFor(() => {
      // Must display 'Not recorded' for all missing fields
      const notRecordedBadges = screen.getAllByText(/Not recorded/i);
      expect(notRecordedBadges.length).toBeGreaterThanOrEqual(3);
    });

    // Ensure invented random seed 42 and generic sha256 fingerprint are NOT in the document
    expect(screen.queryByText(/sha256:7305ac4d8f3e/i)).not.toBeInTheDocument();
  });

  it('ModelComparison: displays "—" for missing metrics and "Not recorded" for missing protocol', async () => {
    vi.spyOn(api, 'getModelMetrics').mockResolvedValue([
      {
        model: 'IncompleteModel',
        feature_group: 'review_only',
        evaluation_type: null,
        accuracy: null,
        fake_precision: null,
        fake_recall: null,
        fake_f1: 0.75,
        roc_auc: null,
        pr_auc: null,
        brier_score: null,
      },
    ]);

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ModelComparisonPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('IncompleteModel')).toBeInTheDocument();
      expect(screen.getByText('Not recorded')).toBeInTheDocument();
    });

    // Multiple dashes for null metrics in row
    const dashes = screen.getAllByText('—');
    expect(dashes.length).toBeGreaterThanOrEqual(4);
  });
});
