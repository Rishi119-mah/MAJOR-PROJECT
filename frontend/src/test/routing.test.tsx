import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppProvider } from '../context/AppContext';
import { routes } from '../router';

const renderTestApp = (initialRoute = '/dashboard') => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  const testRouter = createMemoryRouter(routes, {
    initialEntries: [initialRoute],
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <AppProvider>
        <RouterProvider router={testRouter} />
      </AppProvider>
    </QueryClientProvider>
  );
};

describe('App Root and Navigation Flow', () => {
  it('renders application with sidebar and student developer branding', () => {
    renderTestApp();
    expect(screen.getByText('FraudShield ML')).toBeInTheDocument();
    expect(screen.getByText(/v2.1.0 • Rishi Maheshwari/i)).toBeInTheDocument();
  });

  it('renders exactly the four core navigation links in sidebar', () => {
    renderTestApp();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Product Checker')).toBeInTheDocument();
    expect(screen.getByText('Model Analysis')).toBeInTheDocument();
    expect(screen.getByText('Network Analysis')).toBeInTheDocument();
  });

  it('navigates cleanly to Product Checker', () => {
    renderTestApp('/product-checker');
    expect(screen.getByText('Product Review Risk Checker')).toBeInTheDocument();
    expect(screen.getByText('Check Product')).toBeInTheDocument();
  });

  it('navigates cleanly to Model Analysis', () => {
    renderTestApp('/model-analysis');
    expect(screen.getByText('Machine Learning & Anomaly Analysis')).toBeInTheDocument();
    expect(screen.getByText('Model Comparison')).toBeInTheDocument();
    expect(screen.getByText('Clustering')).toBeInTheDocument();
    expect(screen.getByText('SHAP Explainability')).toBeInTheDocument();
  });

  it('navigates cleanly to Network Analysis', () => {
    renderTestApp('/network-analysis');
    expect(screen.getByText('Reviewer-Product Network Analysis')).toBeInTheDocument();
  });
});
