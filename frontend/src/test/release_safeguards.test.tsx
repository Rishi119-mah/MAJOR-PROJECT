import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { ProductCheckerPage } from '../pages/ProductChecker';
import { NetworkAnalysisPage } from '../pages/NetworkAnalysis';
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

describe('Release Sign-off Safeguards & Fallback Elimination Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('ProductChecker: renders clean input and check button without unverified defaults', async () => {
    vi.spyOn(api, 'getModelRegistry').mockResolvedValue([
      {
        model_name: 'test_model',
        filename: 'test.joblib',
        feature_group: 'review_only',
        required_features_count: 2,
        required_features: ['avg_review_rating', 'clustering_coef'],
        description: 'Test Model',
      },
    ]);

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ProductCheckerPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Product Review Risk Checker/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Check Product/i })).toBeInTheDocument();
    });

    expect(screen.getByDisplayValue('1080')).toBeInTheDocument();
  });

  it('NetworkAnalysis: displays synthetic graph disclosure and legend', async () => {
    vi.spyOn(api, 'getNetworkGraph').mockResolvedValue({
      nodes: [
        { id: 'u1', label: 'Reviewer #1', type: 'reviewer', degree: 3, position: { x: 0, y: 0 }, is_suspicious: false },
        { id: 'p1', label: 'Product #1080', type: 'product', degree: 5, position: { x: 100, y: 100 }, is_suspicious: true },
      ],
      edges: [
        { id: 'e1', source: 'u1', target: 'p1', is_fake: true, weight: 1.0 },
      ],
      stats: {
        total_nodes: 2,
        reviewer_nodes: 1,
        product_nodes: 1,
        total_edges: 1,
        ring_reviewers: 1,
        target_products: 1,
      },
      mode: 'synthetic',
    });

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <NetworkAnalysisPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Reviewer-Product Network Analysis/i)).toBeInTheDocument();
      expect(screen.getByText(/Synthetic Demonstration Graph Disclosure/i)).toBeInTheDocument();
    });
  });

  it('ModelAnalysis: displays model metrics table and tab headers', async () => {
    vi.spyOn(api, 'getModelMetrics').mockResolvedValue([
      {
        experiment_id: 'EXP-1',
        model: 'LightGBM Model',
        feature_group: 'review_network_image',
        accuracy: 0.85,
        fake_precision: 0.82,
        fake_recall: 0.81,
        fake_f1: 0.815,
        roc_auc: 0.92,
        evaluation_type: 'Final independent test',
      },
    ]);

    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ModelAnalysisPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Machine Learning & Anomaly Analysis/i)).toBeInTheDocument();
      expect(screen.getByText(/Holdout Model Leaderboard/i)).toBeInTheDocument();
    });
  });
});
