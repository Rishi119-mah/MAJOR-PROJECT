import React from 'react';
import { useLocation } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import { ShieldCheck, ExternalLink } from 'lucide-react';

export const Header: React.FC = () => {
  const location = useLocation();
  const { selectedModel } = useApp();

  const getPageTitle = (pathname: string): string => {
    switch (pathname) {
      case '/dashboard':
        return 'Overview Dashboard';
      case '/product-checker':
        return 'Product Checker';
      case '/model-analysis':
        return 'Model Analysis';
      case '/network-analysis':
        return 'Network Analysis';
      default:
        return 'Fake Review Detection System';
    }
  };

  return (
    <header className="h-16 bg-white border-b border-gray-200 px-6 flex items-center justify-between z-20 shrink-0">
      {/* Left: Page Title */}
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-bold text-gray-900 tracking-tight">
          {getPageTitle(location.pathname)}
        </h1>
        <span className="hidden md:inline-flex items-center text-xs font-medium text-gray-500 bg-gray-100 px-2.5 py-0.5 rounded border border-gray-200">
          CSE Major Project
        </span>
      </div>

      {/* Right: Active Model & API Docs */}
      <div className="flex items-center gap-4">
        {/* Active Model Indicator */}
        <div className="hidden sm:flex items-center gap-1.5 text-xs bg-gray-50 border border-gray-200 px-3 py-1 rounded-md text-gray-700">
          <ShieldCheck size={14} className="text-blue-600" />
          <span className="text-gray-500">Model:</span>
          <span className="font-mono font-semibold text-gray-900 truncate max-w-[200px]" title={selectedModel}>
            {selectedModel}
          </span>
        </div>

        {/* API Docs Link */}
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1 text-xs text-gray-600 hover:text-blue-600 font-medium px-2 py-1 rounded hover:bg-gray-50 transition-colors"
          title="Open FastAPI Swagger Interactive Documentation"
        >
          <span>FastAPI Docs</span>
          <ExternalLink size={12} />
        </a>
      </div>
    </header>
  );
};
