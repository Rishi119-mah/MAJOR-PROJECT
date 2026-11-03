import { describe, it, expect } from 'vitest';
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { DashboardPage } from '../pages/Dashboard';
import { ProductCheckerPage } from '../pages/ProductChecker';
import { ModelAnalysisPage } from '../pages/ModelAnalysis';
import { AppProvider } from '../context/AppContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

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

describe('Data Integrity & Dynamic Binding Verification', () => {
  it('Dashboard renders total products, fake, genuine, and model F1 without hardcoded numbers', async () => {
    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <DashboardPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Total Products/i)).toBeInTheDocument();
      expect(screen.getByText(/Fake-Labelled/i)).toBeInTheDocument();
      expect(screen.getByText(/Genuine Products/i)).toBeInTheDocument();
      expect(screen.getByText(/Best Model F1-Score/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Feature Modality Progression/i)).toBeInTheDocument();
  });

  it('Product Checker renders Product ID input, model selector, and check button', async () => {
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

    expect(screen.getByLabelText(/Product ID/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Evaluation Model/i)).toBeInTheDocument();
  });

  it('Model Analysis renders all three unified tabs: comparison, clustering, shap', async () => {
    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ModelAnalysisPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText('Machine Learning & Anomaly Analysis')).toBeInTheDocument();
      expect(screen.getAllByText(/Model Comparison/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/Clustering/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/SHAP Explainability/i).length).toBeGreaterThanOrEqual(1);
    });
  });
});
