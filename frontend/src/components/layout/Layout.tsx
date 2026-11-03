import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useApp } from '../../context/AppContext';
import { AlertCircle, Sparkles } from 'lucide-react';

export const Layout: React.FC = () => {
  const { isOnline, mode, t } = useApp();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50">
      {/* Collapsible Left Sidebar */}
      <Sidebar />

      {/* Main App Container */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Header />

        {/* Demo Mode Notification Banner */}
        {mode === 'demo' && (
          <div className="bg-amber-50 border-b border-amber-200 px-6 py-2 flex items-center justify-between text-xs text-amber-900">
            <div className="flex items-center gap-2">
              <Sparkles size={15} className="text-amber-600 shrink-0" />
              <span>
                <strong>Demo Mode Active:</strong> Displaying illustrative sample fixtures for interactive presentation without requiring the full backend training pipeline. Switch to <strong>Research Mode</strong> in top bar for verified empirical backend data.
              </span>
            </div>
            <span className="font-mono text-[10px] bg-amber-200/70 text-amber-900 px-2 py-0.5 rounded font-bold uppercase shrink-0">
              Synthetic Fixtures
            </span>
          </div>
        )}

        {/* Offline Banner if API Disconnected in Research Mode */}
        {!isOnline && mode === 'research' && (
          <div className="bg-rose-50 border-b border-rose-200 px-6 py-2 flex items-center justify-between text-xs text-rose-800">
            <div className="flex items-center gap-2">
              <AlertCircle size={15} className="text-rose-600 shrink-0" />
              <span>
                <strong>FastAPI Backend Disconnected</strong>: Ensure server is running at{' '}
                <code className="font-mono bg-rose-100 px-1 py-0.5 rounded">uvicorn src.api.main:app --port 8000</code>.
                Switch to <strong>Demo Mode</strong> in top bar to test UI with synthetic data.
              </span>
            </div>
          </div>
        )}

        {/* Scrollable Main Content Region */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <Outlet />

          {/* Persistent Academic & Research Footnote */}
          <footer className="mt-12 pt-6 border-t border-slate-200 text-center text-xs text-slate-500">
            <p>{t('disclaimer')}</p>
            <p className="mt-1 font-mono text-[11px] text-slate-400">
              Department of Computer Science & Engineering • Final Year Major Project 2026
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
};

