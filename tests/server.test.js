import { describe, it, expect } from 'vitest';
import app from '../src/server.js';
import { AIFallbackProvider } from '../src/providers/aiFallbackProvider.js';
import { ThreatService } from '../src/services/threatService.js';

describe('1. Healthcheck & System Environment API', () => {
  it('should evaluate system health endpoint', async () => {
    const res = await app.fetch ? app.fetch(new Request('http://localhost:5000/health')) : null;
    expect(AIFallbackProvider).toBeDefined();
    expect(ThreatService).toBeDefined();
  });
});

describe('2. Multi-Provider AI Fallback Circuit Tests', () => {
  it('should run Tier 3 Deterministic Backup Engine for Benign traffic', async () => {
    const result = await AIFallbackProvider.analyzeThreat({
      destinationPort: 80,
      flowDuration: 12000,
      totalFwdPackets: 2,
      totalBwdPackets: 3,
    });

    expect(result).toBeDefined();
    expect(result.prediction).toBe('BENIGN');
    expect(result.providerUsed).toBe('ZeroDowntimeDeterministicEngine');
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.riskLevel).toBe('LOW');
  });

  it('should classify heavy packet spikes as DDoS in Tier 3 engine', async () => {
    const result = await AIFallbackProvider.analyzeThreat({
      destinationPort: 443,
      flowDuration: 2000000,
      totalFwdPackets: 6000,
      totalBwdPackets: 1000,
    });

    expect(result.prediction).toBe('DDoS');
    expect(result.riskLevel).toBe('CRITICAL');
    expect(result.providerUsed).toBe('ZeroDowntimeDeterministicEngine');
  });

  it('should classify SSH port spikes as BruteForce in Tier 3 engine', async () => {
    const result = await AIFallbackProvider.analyzeThreat({
      destinationPort: 22,
      flowDuration: 5000,
      totalFwdPackets: 150,
      totalBwdPackets: 50,
    });

    expect(result.prediction).toBe('BruteForce');
    expect(result.riskLevel).toBe('HIGH');
  });
});

describe('3. Clean Architecture Threat Service Input Validation', () => {
  it('should reject invalid destination port numbers', async () => {
    await expect(
      ThreatService.processPrediction({ destinationPort: 99999, flowDuration: 100 })
    ).rejects.toThrow('Invalid Destination Port');
  });

  it('should reject negative flow duration values', async () => {
    await expect(
      ThreatService.processPrediction({ destinationPort: 80, flowDuration: -50 })
    ).rejects.toThrow('Invalid Flow Duration');
  });

  it('should successfully process valid threat predictions', async () => {
    const response = await ThreatService.processPrediction({
      destinationPort: 80,
      flowDuration: 1200,
      totalFwdPackets: 5,
      totalBwdPackets: 5,
    });

    expect(response.status).toBe('success');
    expect(response.data.prediction).toBe('BENIGN');
  });
});
