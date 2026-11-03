import React from 'react';
import { NavLink } from 'react-router-dom';
import { useApp } from '../../context/AppContext';
import {
  LayoutDashboard,
  ShieldAlert,
  BarChart3,
  Network,
  Boxes,
  Sparkles,
  FlaskConical,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
  Shield,
  Activity,
  Globe,
} from 'lucide-react';
import { Language } from '../../types';

export const Sidebar: React.FC = () => {
  const { sidebarCollapsed, toggleSidebar, t, language, setLanguage, isOnline, mode } = useApp();

  const navItems = [
    { to: '/dashboard', label: t('dashboard'), icon: LayoutDashboard },
    { to: '/product-checker', label: t('productChecker'), icon: ShieldAlert },
    { to: '/model-comparison', label: t('modelComparison'), icon: BarChart3 },
    { to: '/network-explorer', label: t('networkExplorer'), icon: Network },
    { to: '/clustering', label: t('clustering'), icon: Boxes },
    { to: '/explainability', label: t('explainability'), icon: Sparkles },
    { to: '/experiments', label: t('experiments'), icon: FlaskConical },
    { to: '/reports', label: t('reports'), icon: FileText },
    { to: '/settings', label: t('settings'), icon: Settings },
  ];

  return (
    <aside
      className={`bg-slate-900 border-r border-slate-800 text-slate-300 flex flex-col justify-between transition-all duration-300 select-none z-30 shrink-0 ${
        sidebarCollapsed ? 'w-20' : 'w-64'
      }`}
    >
      {/* Top Header */}
      <div>
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-800">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-md shadow-indigo-500/20 shrink-0">
              <Shield size={22} className="stroke-[2.5]" />
            </div>
            {!sidebarCollapsed && (
              <div className="flex flex-col min-w-0">
                <span className="font-bold text-white tracking-tight truncate leading-tight">FraudShield ML</span>
                <span className="text-[10px] text-slate-400 font-mono tracking-wider uppercase truncate">
                  v2.1.0 • Antigravity
                </span>
              </div>
            )}
          </div>
          <button
            onClick={toggleSidebar}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            title={sidebarCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>

        {/* Navigation Links */}
        <nav className="p-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all group ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
                      : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
                  }`
                }
                title={sidebarCollapsed ? item.label : undefined}
              >
                <Icon size={19} className="shrink-0 transition-transform group-hover:scale-105" />
                {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Bottom Controls */}
      <div className="p-4 border-t border-slate-800 space-y-3">
        {/* Language Selector */}
        {!sidebarCollapsed ? (
          <div className="flex items-center justify-between bg-slate-950/60 rounded-lg p-1.5 border border-slate-800 text-xs">
            <div className="flex items-center gap-1.5 text-slate-400 px-1">
              <Globe size={14} />
              <span>Lang</span>
            </div>
            <div className="flex gap-1">
              {(['en', 'hi', 'hinglish'] as Language[]).map((l) => (
                <button
                  key={l}
                  onClick={() => setLanguage(l)}
                  className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                    language === l
                      ? 'bg-indigo-600 text-white font-semibold'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800'
                  }`}
                >
                  {l.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <button
            onClick={() => {
              const next: Record<Language, Language> = { en: 'hi', hi: 'hinglish', hinglish: 'en' };
              setLanguage(next[language]);
            }}
            className="w-full flex items-center justify-center p-2 rounded-lg bg-slate-800/80 text-xs font-mono font-bold text-slate-300"
            title={`Language: ${language.toUpperCase()}`}
          >
            {language.substring(0, 2).toUpperCase()}
          </button>
        )}

        {/* System & Mode Status Indicator */}
        <div
          className={`flex items-center gap-2 rounded-lg bg-slate-950/40 border border-slate-800/80 p-2 text-xs ${
            sidebarCollapsed ? 'justify-center' : ''
          }`}
        >
          <div className="relative flex items-center justify-center">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isOnline ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
              }`}
            />
          </div>
          {!sidebarCollapsed && (
            <div className="flex flex-col min-w-0">
              <span className="font-semibold text-slate-200 text-[11px] leading-tight">
                {isOnline ? 'FastAPI Online' : 'API Offline'}
              </span>
              <span className="text-[10px] text-slate-400 capitalize">
                {mode === 'research' ? 'Research Mode' : 'Demo Mode'}
              </span>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};

