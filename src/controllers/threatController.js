import { ThreatService } from '../services/threatService.js';

export class ThreatController {
  static async predict(req, res, next) {
    try {
      const result = await ThreatService.processPrediction(req.body);
      return res.status(200).json(result);
    } catch (err) {
      next(err);
    }
  }

  static async getLogs(req, res, next) {
    try {
      const limit = Number(req.query.limit || 20);
      const result = await ThreatService.getThreatLogs(limit);
      return res.status(200).json(result);
    } catch (err) {
      next(err);
    }
  }

  static async healthCheck(req, res, next) {
    try {
      return res.status(200).json({
        status: 'UP',
        timestamp: new Date().toISOString(),
        service: 'ai-cyber-threat-detector',
        version: '1.0.0',
        aiCircuit: 'Active (OpenRouter -> Gemini -> Deterministic ONNX)',
      });
    } catch (err) {
      next(err);
    }
  }
}
