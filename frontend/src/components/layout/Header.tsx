import React from 'react';
import { useLocation } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import { ShieldCheck, Database, Layers, ExternalLink, HelpCircle } from 'lucide-react';

export const Header: React.FC = () => {
  const location = useLocation();
  const { t, mode, setMode, selectedModel, isOnline } = useApp();

  const getPageTitle = (pathname: string): string => {
    switch (pathname) {
      case '/dashboard':
        return t('dashboard');
      case '/product-checker':
        return t('productChecker');
      case '/model-comparison':
        return t('modelComparison');
      case '/network-explorer':
        return t('networkExplorer');
      case '/clustering':
        return t('clustering');
      case '/explainability':
        return t('explainability');
      case '/experiments':
        return t('experiments');
      case '/reports':
        return t('reports');
      case '/settings':
        return t('settings');
      default:
        return 'Fake Review Detection System';
    }
  };

  return (
    <header className="h-16 bg-white border-b border-slate-200/90 px-6 flex items-center justify-between z-20 shrink-0">
      {/* Left: Page Title & Breadcrumb */}
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-bold text-slate-900 tracking-tight">
          {getPageTitle(location.pathname)}
        </h1>
        <span className="hidden md:inline-flex items-center gap-1 text-[11px] font-medium text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
          <Layers size={12} className="text-slate-400" />
          Amazon Coordinated Fraud
        </span>
      </div>

      {/* Right: Quick Context Controls */}
      <div className="flex items-center gap-3">
        {/* Active Model Pill */}
        <div className="hidden lg:flex items-center gap-1.5 text-xs bg-slate-50 border border-slate-200 px-2.5 py-1 rounded-lg text-slate-700">
          <ShieldCheck size={14} className="text-indigo-600" />
          <span className="text-slate-500 font-medium">Model:</span>
          <span className="font-mono font-semibold truncate max-w-[170px]" title={selectedModel}>
            {selectedModel}
          </span>
        </div>

        {/* Mode Toggle */}
        <div className="flex items-center bg-slate-100 p-0.5 rounded-lg border border-slate-200 text-xs">
          <button
            onClick={() => setMode('research')}
            className={`px-2.5 py-1 rounded-md font-medium transition-all ${
              mode === 'research'
                ? 'bg-white text-slate-900 shadow-sm font-semibold'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            Research Mode
          </button>
          <button
            onClick={() => setMode('demo')}
            className={`px-2.5 py-1 rounded-md font-medium transition-all ${
              mode === 'demo'
                ? 'bg-indigo-600 text-white shadow-sm font-semibold'
                : 'text-slate-500 hover:text-slate-800'
            }`}
          >
            Demo Mode
          </button>
        </div>

        {/* API Docs link */}
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="hidden sm:flex items-center gap-1 text-xs text-slate-600 hover:text-indigo-600 font-medium px-2 py-1 rounded hover:bg-slate-50 transition-colors"
          title="Open FastAPI Swagger Interactive Documentation"
        >
          <span>FastAPI Docs</span>
          <ExternalLink size={12} />
        </a>
      </div>
    </header>
  );
};

