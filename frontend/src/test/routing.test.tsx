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
  it('renders application with sidebar and branding', () => {
    renderTestApp();
    expect(screen.getByText('FraudShield ML')).toBeInTheDocument();
    expect(screen.getByText(/v2.1.0 • Antigravity/i)).toBeInTheDocument();
  });

  it('renders all nine navigation links in sidebar', () => {
    renderTestApp();
    expect(screen.getAllByText('Overview Dashboard').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Product Checker')).toBeInTheDocument();
    expect(screen.getByText('Model Comparison')).toBeInTheDocument();
    expect(screen.getByText('Network Explorer')).toBeInTheDocument();
    expect(screen.getByText('Clustering Analysis')).toBeInTheDocument();
    expect(screen.getByText('Explainability (SHAP)')).toBeInTheDocument();
    expect(screen.getByText('Research Experiments')).toBeInTheDocument();
    expect(screen.getByText('Reports & Exports')).toBeInTheDocument();
    expect(screen.getByText('Settings & Status')).toBeInTheDocument();
  });

  it('displays research and demo mode toggle', () => {
    renderTestApp();
    const researchButtons = screen.getAllByText(/Research Mode/i);
    expect(researchButtons.length).toBeGreaterThanOrEqual(1);
    const demoButtons = screen.getAllByText(/Demo Mode/i);
    expect(demoButtons.length).toBeGreaterThanOrEqual(1);
  });
});
