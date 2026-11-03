import React from 'react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  badge?: string;
  badgeColor?: 'emerald' | 'amber' | 'rose' | 'indigo' | 'slate';
  tooltip?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  badge,
  badgeColor = 'indigo',
  tooltip,
}) => {
  const badgeClasses = {
    emerald: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    amber: 'bg-amber-50 text-amber-700 border-amber-200',
    rose: 'bg-rose-50 text-rose-700 border-rose-200',
    indigo: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    slate: 'bg-slate-100 text-slate-700 border-slate-200',
  }[badgeColor];

  return (
    <div
      className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm hover:shadow transition-shadow relative overflow-hidden"
      title={tooltip}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{title}</span>
        {icon && <div className="p-2 bg-slate-50 rounded-lg text-slate-600 border border-slate-100">{icon}</div>}
      </div>

      <div className="mt-3 flex items-baseline gap-2">
        <span className="text-2xl font-bold text-slate-900 tracking-tight font-mono">{value}</span>
        {badge && (
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${badgeClasses}`}>
            {badge}
          </span>
        )}
      </div>

      {subtitle && <p className="mt-1.5 text-xs text-slate-500 line-clamp-1">{subtitle}</p>}
    </div>
  );
};

