import { AIFallbackProvider } from '../providers/aiFallbackProvider.js';
import { ThreatRepository } from '../repositories/threatRepository.js';
import { AppError } from '../middleware/errorHandler.js';

export class ThreatService {
  static async processPrediction(inputData) {
    // Input validation checks
    const port = Number(inputData.destinationPort);
    if (isNaN(port) || port < 1 || port > 65535) {
      throw new AppError('Invalid Destination Port. Must be between 1 and 65535.', 400);
    }

    const duration = Number(inputData.flowDuration);
    if (isNaN(duration) || duration < 0) {
      throw new AppError('Invalid Flow Duration. Must be a non-negative number.', 400);
    }

    // Execute Multi-Provider AI Fallback Circuit
    const aiAnalysis = await AIFallbackProvider.analyzeThreat({
      destinationPort: port,
      flowDuration: duration,
      totalFwdPackets: Number(inputData.totalFwdPackets || 0),
      totalBwdPackets: Number(inputData.totalBwdPackets || 0),
    });

    const threatRecord = {
      destinationPort: port,
      flowDuration: duration,
      totalFwdPackets: Number(inputData.totalFwdPackets || 0),
      totalBwdPackets: Number(inputData.totalBwdPackets || 0),
      sourceIp: inputData.sourceIp || '127.0.0.1',
      ...aiAnalysis,
    };

    // Save to Database Ledger
    const savedRecord = await ThreatRepository.saveLog(threatRecord);

    return {
      status: 'success',
      data: savedRecord,
    };
  }

  static async getThreatLogs(limit = 20) {
    const logs = await ThreatRepository.getRecentLogs(limit);
    const stats = await ThreatRepository.getStats();
    return {
      status: 'success',
      stats,
      logs,
    };
  }
}
