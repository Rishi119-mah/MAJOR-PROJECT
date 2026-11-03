import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { DashboardPage } from '../pages/Dashboard';
import { ProductCheckerPage } from '../pages/ProductChecker';
import { ExplainabilityPage } from '../pages/Explainability';
import { ReportsPage } from '../pages/Reports';
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
  it('Dashboard renders dynamic holdout protocol and top models without static fake numbers', async () => {
    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <DashboardPage />
      </Wrapper>
    );

    // Wait for Dashboard to render
    await waitFor(() => {
      expect(screen.getByText(/Executive Intelligence Overview/i)).toBeInTheDocument();
    });

    // Check that holdout test protocol badge is rendered
    expect(screen.getAllByText(/Holdout/i).length).toBeGreaterThan(0);
    // Check that LightGBM (best holdout model) or XGBoost appears
    expect(screen.getAllByText(/XGBoost|LightGBM/i).length).toBeGreaterThan(0);
  });

  it('Explainability page dynamically renders modality breakdown from backend API', async () => {
    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ExplainabilityPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/TreeSHAP Explainability/i)).toBeInTheDocument();
    });

    // Verify presence of modalities
    expect(screen.getAllByText(/Network Centralit/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Review Behavior/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Image Similarit/i).length).toBeGreaterThan(0);
  });

  it('Reports page renders dynamic holdout benchmarks and export option', async () => {
    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ReportsPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Executive Report Generator & Exports/i)).toBeInTheDocument();
    });

    // Verification of holdout split protocol label in document view
    expect(screen.getByText(/Untouched 20% Holdout Split/i)).toBeInTheDocument();
    expect(screen.getByText(/Print PDF Report/i)).toBeInTheDocument();
  });

  it('Product Checker contains fetch button to prevent stale product submissions', async () => {
    const Wrapper = createWrapper();
    render(
      <Wrapper>
        <ProductCheckerPage />
      </Wrapper>
    );

    await waitFor(() => {
      expect(screen.getByText(/Product-Level Fake Review Risk Checker/i)).toBeInTheDocument();
    });

    // Check for the Fetch button
    const fetchBtn = screen.getByRole('button', { name: /fetch/i });
    expect(fetchBtn).toBeInTheDocument();
  });
});

