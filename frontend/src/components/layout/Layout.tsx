import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useApp } from '../../context/AppContext';
import { AlertCircle } from 'lucide-react';

export const Layout: React.FC = () => {
  const { isOnline } = useApp();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-gray-50">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Container */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        <Header />

        {/* Offline Notification if API Disconnected */}
        {!isOnline && (
          <div className="bg-amber-50 border-b border-amber-200 px-6 py-2.5 flex items-center justify-between text-xs text-amber-900">
            <div className="flex items-center gap-2">
              <AlertCircle size={15} className="text-amber-600 shrink-0" />
              <span>
                <strong>FastAPI Backend Disconnected:</strong> Ensure server is running at{' '}
                <code className="font-mono bg-amber-100 px-1 py-0.5 rounded text-amber-950">
                  python3 -m uvicorn src.api.main:app --port 8000
                </code>.
              </span>
            </div>
          </div>
        )}

        {/* Scrollable Content */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <Outlet />

          {/* Academic Footer */}
          <footer className="mt-12 pt-6 border-t border-gray-200 text-center text-xs text-gray-500">
            <p className="font-medium text-gray-600">
              Fake Review Detection Using Machine Learning and Network Analysis
            </p>
            <p className="mt-1 font-mono text-[11px] text-gray-400">
              Department of Computer Science & Engineering • Major Project • Developed by Rishi Maheshwari
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
};
