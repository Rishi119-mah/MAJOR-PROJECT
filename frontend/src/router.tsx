import React from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { DashboardPage } from './pages/Dashboard';
import { ProductCheckerPage } from './pages/ProductChecker';
import { ModelAnalysisPage } from './pages/ModelAnalysis';
import { NetworkAnalysisPage } from './pages/NetworkAnalysis';

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
        path: 'model-analysis',
        element: <ModelAnalysisPage />,
      },
      {
        path: 'network-analysis',
        element: <NetworkAnalysisPage />,
      },
      // Legacy redirects to maintain compatibility
      {
        path: 'model-comparison',
        element: <Navigate to="/model-analysis" replace />,
      },
      {
        path: 'clustering',
        element: <Navigate to="/model-analysis" replace />,
      },
      {
        path: 'explainability',
        element: <Navigate to="/model-analysis" replace />,
      },
      {
        path: 'network-explorer',
        element: <Navigate to="/network-analysis" replace />,
      },
      {
        path: 'experiments',
        element: <Navigate to="/model-analysis" replace />,
      },
      {
        path: 'reports',
        element: <Navigate to="/dashboard" replace />,
      },
      {
        path: 'settings',
        element: <Navigate to="/dashboard" replace />,
      },
      {
        path: '*',
        element: <Navigate to="/dashboard" replace />,
      },
    ],
  },
];

export const router = createBrowserRouter(routes);
