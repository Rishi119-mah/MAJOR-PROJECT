import React from 'react';
import { RiskTier } from '../../types';
import { ShieldCheck, AlertTriangle, AlertOctagon } from 'lucide-react';

interface RiskBadgeProps {
  tier: RiskTier | string;
  score?: number;
  size?: 'sm' | 'md' | 'lg';
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({ tier, score, size = 'md' }) => {
  const isLow = tier === 'LOW_RISK';
  const isMod = tier === 'MODERATE_RISK';
  const isHigh = tier === 'HIGH_RISK';

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 font-medium',
    md: 'text-sm px-2.5 py-1 font-semibold',
    lg: 'text-base px-3.5 py-1.5 font-bold',
  }[size];

  const iconSizes = {
    sm: 12,
    md: 16,
    lg: 20,
  }[size];

  if (isLow) {
    return (
      <span
        className={`inline-flex items-center gap-1.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 ${sizeClasses}`}
      >
        <ShieldCheck size={iconSizes} className="text-emerald-600" />
        Low Risk {score !== undefined && `(${score.toFixed(1)})`}
      </span>
    );
  }

  if (isMod) {
    return (
      <span
        className={`inline-flex items-center gap-1.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 ${sizeClasses}`}
      >
        <AlertTriangle size={iconSizes} className="text-amber-600" />
        Moderate Risk {score !== undefined && `(${score.toFixed(1)})`}
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200 ${sizeClasses}`}
    >
      <AlertOctagon size={iconSizes} className="text-rose-600" />
      High Risk {score !== undefined && `(${score.toFixed(1)})`}
    </span>
  );
};

