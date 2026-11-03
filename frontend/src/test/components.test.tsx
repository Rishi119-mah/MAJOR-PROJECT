import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { RiskBadge } from '../components/common/RiskBadge';
import { CircularRiskScore } from '../components/common/CircularRiskScore';
import { StatCard } from '../components/common/StatCard';
import { translations } from '../i18n/translations';

describe('Frontend UI Components', () => {
  it('renders RiskBadge correctly for all three tiers', () => {
    const { rerender } = render(<RiskBadge tier="LOW_RISK" score={15.4} />);
    expect(screen.getByText(/Low Risk \(15.4\)/i)).toBeInTheDocument();

    rerender(<RiskBadge tier="MODERATE_RISK" score={45.0} />);
    expect(screen.getByText(/Moderate Risk \(45.0\)/i)).toBeInTheDocument();

    rerender(<RiskBadge tier="HIGH_RISK" score={89.2} />);
    expect(screen.getByText(/High Risk \(89.2\)/i)).toBeInTheDocument();
  });

  it('renders CircularRiskScore with score and tier label', () => {
    render(<CircularRiskScore score={78.5} tier="HIGH_RISK" />);
    expect(screen.getByText('78.5')).toBeInTheDocument();
    expect(screen.getByText('/ 100')).toBeInTheDocument();
    expect(screen.getByText(/HIGH RISK/i)).toBeInTheDocument();
  });

  it('renders StatCard with title, value, and badge', () => {
    render(
      <StatCard
        title="Total Products"
        value={3327}
        subtitle="Labeled evaluation data"
        badge="Verified"
        badgeColor="emerald"
      />
    );
    expect(screen.getByText('Total Products')).toBeInTheDocument();
    expect(screen.getByText('3327')).toBeInTheDocument();
    expect(screen.getByText('Verified')).toBeInTheDocument();
  });

  it('contains consistent translation dictionaries for EN, HI, and Hinglish', () => {
    const keys = Object.keys(translations.en);
    expect(keys.length).toBeGreaterThan(15);

    // Verify all keys exist in Hindi and Hinglish
    keys.forEach((key) => {
      expect(translations.hi[key]).toBeDefined();
      expect(translations.hinglish[key]).toBeDefined();
    });
  });
});

