import React from 'react';
import { RiskTier } from '../../types';

interface CircularRiskScoreProps {
  score: number;
  tier: RiskTier | string;
  size?: number;
}

export const CircularRiskScore: React.FC<CircularRiskScoreProps> = ({ score, tier, size = 180 }) => {
  const strokeWidth = 14;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clampedScore = Math.min(Math.max(score, 0), 100);
  const strokeDashoffset = circumference - (clampedScore / 100) * circumference;

  let color = '#ef4444'; // high
  let bgColor = '#fef2f2';
  let label = 'HIGH RISK';
  let textColor = 'text-rose-600';

  if (tier === 'LOW_RISK') {
    color = '#10b981';
    bgColor = '#ecfdf5';
    label = 'LOW RISK';
    textColor = 'text-emerald-600';
  } else if (tier === 'MODERATE_RISK') {
    color = '#f59e0b';
    bgColor = '#fffbeb';
    label = 'MODERATE RISK';
    textColor = 'text-amber-600';
  }

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          {/* Background Track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#e2e8f0"
            strokeWidth={strokeWidth}
            fill="transparent"
          />
          {/* Progress Arc */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            fill="transparent"
            className="transition-all duration-1000 ease-out"
          />
        </svg>

        {/* Center Content */}
        <div className="absolute flex flex-col items-center justify-center text-center">
          <span className="text-4xl font-extrabold tracking-tight text-slate-900 font-mono">
            {score.toFixed(1)}
          </span>
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider mt-0.5">
            / 100
          </span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full mt-1.5 uppercase ${textColor}`} style={{ backgroundColor: bgColor }}>
            {label}
          </span>
        </div>
      </div>
    </div>
  );
};

