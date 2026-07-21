import { z } from 'zod';
import { env } from '../config/env.js';

// Strict Output Schema Definition
export const ThreatAnalysisSchema = z.object({
  prediction: z.enum(['BENIGN', 'DDoS', 'PortScan', 'Bot', 'BruteForce', 'MALICIOUS']),
  confidence: z.number().min(0).max(100),
  riskLevel: z.enum(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']),
  threatVector: z.string(),
  recommendedAction: z.string(),
  providerUsed: z.enum(['OpenRouter', 'OpenAI', 'GoogleGemini', 'ZeroDowntimeDeterministicEngine']),
});

/**
 * Multi-Provider AI Fallback Circuit
 * Circuit Tier 1: OpenRouter (openrouter/auto) or OpenAI
 * Circuit Tier 2: Google Gemini (gemini-1.5-flash)
 * Circuit Tier 3: Zero-Downtime Deterministic Backup Engine
 */
export class AIFallbackProvider {
  static async analyzeThreat(features) {
    const prompt = `Analyze the following network traffic features and classify threat status: ${JSON.stringify(features)}`;

    // Circuit Tier 1: OpenRouter / OpenAI
    try {
      if (env.OPENROUTER_API_KEY && env.OPENROUTER_API_KEY !== 'mock-openrouter-key') {
        const result = await this.callOpenRouter(prompt);
        if (result) return this.validateAndFormat(result, 'OpenRouter');
      }
    } catch (err) {
      console.warn('⚠️ Primary AI Provider (OpenRouter/OpenAI) unavailable:', err.message);
    }

    // Circuit Tier 2: Google Gemini (gemini-1.5-flash)
    try {
      if (env.GEMINI_API_KEY && env.GEMINI_API_KEY !== 'mock-gemini-key') {
        const result = await this.callGemini(prompt);
        if (result) return this.validateAndFormat(result, 'GoogleGemini');
      }
    } catch (err) {
      console.warn('⚠️ Secondary AI Provider (Google Gemini) unavailable:', err.message);
    }

    // Circuit Tier 3: Zero-Downtime Deterministic Backup Engine
    console.log('🔄 Engaging Circuit Tier 3: Zero-Downtime Deterministic Backup Engine');
    const deterministicResult = this.runDeterministicEngine(features);
    return this.validateAndFormat(deterministicResult, 'ZeroDowntimeDeterministicEngine');
  }

  static async callOpenRouter(prompt) {
    // Mocked network call placeholder or fetch
    return null; // Will trigger tier fallback
  }

  static async callGemini(prompt) {
    // Mocked network call placeholder or fetch
    return null; // Will trigger tier fallback
  }

  static runDeterministicEngine(features) {
    const port = Number(features.destinationPort || 80);
    const duration = Number(features.flowDuration || 0);
    const totalPackets = (Number(features.totalFwdPackets) || 0) + (Number(features.totalBwdPackets) || 0);

    let prediction = 'BENIGN';
    let confidence = 95.0;
    let riskLevel = 'LOW';
    let threatVector = 'Normal Network Traffic';
    let recommendedAction = 'Allow traffic flow to continue normally.';

    if (totalPackets > 5000 || duration > 1000000) {
      prediction = 'DDoS';
      confidence = 98.5;
      riskLevel = 'CRITICAL';
      threatVector = 'Distributed Denial of Service (DDoS)';
      recommendedAction = 'Drop connections from source IP and enable rate limit threshold.';
    } else if (port === 22 || port === 3389 || port === 21) {
      if (totalPackets > 100) {
        prediction = 'BruteForce';
        confidence = 91.2;
        riskLevel = 'HIGH';
        threatVector = 'SSH/RDP Password Brute Force';
        recommendedAction = 'Temporary IP ban and require multi-factor authentication.';
      } else {
        prediction = 'PortScan';
        confidence = 88.0;
        riskLevel = 'MEDIUM';
        threatVector = 'Port Scanning Reconnaissance';
        recommendedAction = 'Block source IP on firewall edge filter.';
      }
    } else if (totalPackets > 500 && duration < 500) {
      prediction = 'Bot';
      confidence = 93.4;
      riskLevel = 'HIGH';
      threatVector = 'Automated Botnet Scraping/Spam';
      recommendedAction = 'Challenge request with CAPTCHA verification.';
    }

    return {
      prediction,
      confidence,
      riskLevel,
      threatVector,
      recommendedAction,
    };
  }

  static validateAndFormat(payload, providerName) {
    const enriched = {
      ...payload,
      providerUsed: providerName,
    };
    return ThreatAnalysisSchema.parse(enriched);
  }
}
