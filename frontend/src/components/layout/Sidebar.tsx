import React from 'react';
import { NavLink } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import {
  LayoutDashboard,
  CheckCircle2,
  BarChart3,
  Network,
  Shield,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';

export const Sidebar: React.FC = () => {
  const { isOnline, sidebarCollapsed, toggleSidebar } = useApp();

  const navItems = [
    {
      path: '/dashboard',
      label: 'Dashboard',
      icon: LayoutDashboard,
    },
    {
      path: '/product-checker',
      label: 'Product Checker',
      icon: CheckCircle2,
    },
    {
      path: '/model-analysis',
      label: 'Model Analysis',
      icon: BarChart3,
    },
    {
      path: '/network-analysis',
      label: 'Network Analysis',
      icon: Network,
    },
  ];

  return (
    <aside
      className={`h-screen bg-slate-900 text-slate-100 flex flex-col justify-between transition-all duration-200 border-r border-slate-800 z-30 shrink-0 ${
        sidebarCollapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Top Header & Navigation */}
      <div>
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-800">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center text-white shrink-0">
              <Shield size={20} className="stroke-[2.5]" />
            </div>
            {!sidebarCollapsed && (
              <div className="flex flex-col min-w-0">
                <span className="font-bold text-white tracking-tight truncate leading-tight">FraudShield ML</span>
                <span className="text-[10px] text-slate-400 font-mono tracking-wider uppercase truncate">
                  v2.1.0 • Rishi Maheshwari
                </span>
              </div>
            )}
          </div>
          <button
            onClick={toggleSidebar}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            title={sidebarCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>

        {/* Navigation Links — Exactly 4 Pages */}
        <nav className="p-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-blue-600 text-white font-semibold'
                      : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                  } ${sidebarCollapsed ? 'justify-center px-0' : ''}`
                }
                title={sidebarCollapsed ? item.label : undefined}
              >
                <Icon size={18} className="shrink-0" />
                {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Bottom Status Section */}
      <div className="p-4 border-t border-slate-800 space-y-3">
        {/* Backend Online/Offline Status */}
        <div
          className={`flex items-center gap-2 rounded-lg bg-slate-950/50 border border-slate-800 p-2.5 text-xs ${
            sidebarCollapsed ? 'justify-center' : ''
          }`}
        >
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              isOnline ? 'bg-emerald-500' : 'bg-rose-500'
            }`}
          />
          {!sidebarCollapsed && (
            <span className="text-slate-300 text-xs font-medium">
              {isOnline ? 'FastAPI Backend Online' : 'FastAPI Backend Offline'}
            </span>
          )}
        </div>
      </div>
    </aside>
  );
};
