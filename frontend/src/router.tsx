import React from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { DashboardPage } from './pages/Dashboard';
import { ProductCheckerPage } from './pages/ProductChecker';
import { ModelComparisonPage } from './pages/ModelComparison';
import { NetworkExplorerPage } from './pages/NetworkExplorer';
import { ClusteringPage } from './pages/Clustering';
import { ExplainabilityPage } from './pages/Explainability';
import { ExperimentsPage } from './pages/Experiments';
import { ReportsPage } from './pages/Reports';
import { SettingsPage } from './pages/Settings';

export const routes = [
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        index: true,
        element: <Navigate to="/dashboard" replace />,
      },
      {
        path: 'dashboard',
        element: <DashboardPage />,
      },
      {
        path: 'product-checker',
        element: <ProductCheckerPage />,
      },
      {
        path: 'model-comparison',
        element: <ModelComparisonPage />,
      },
      {
        path: 'network-explorer',
        element: <NetworkExplorerPage />,
      },
      {
        path: 'clustering',
        element: <ClusteringPage />,
      },
      {
        path: 'explainability',
        element: <ExplainabilityPage />,
      },
      {
        path: 'experiments',
        element: <ExperimentsPage />,
      },
      {
        path: 'reports',
        element: <ReportsPage />,
      },
      {
        path: 'settings',
        element: <SettingsPage />,
      },
      {
        path: '*',
        element: <Navigate to="/dashboard" replace />,
      },
    ],
  },
];

export const router = createBrowserRouter(routes);
