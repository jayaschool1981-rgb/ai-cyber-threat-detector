import express from 'express';
import { env } from './config/env.js';
import { connectDB } from './database/db.js';
import { securityHeaders, corsMiddleware, apiRateLimiter } from './middleware/security.js';
import { errorHandler, AppError } from './middleware/errorHandler.js';
import { ThreatController } from './controllers/threatController.js';

const app = express();

// Apply Security Middleware (Phase 1)
app.use(securityHeaders);
app.use(corsMiddleware);
app.use(express.json());

// Apply IP Rate Limiter to API routes
app.use('/api', apiRateLimiter);

// API Endpoints (Phase 3 Clean Architecture Controllers)
app.get('/health', ThreatController.healthCheck);
app.get('/api/v1/health', ThreatController.healthCheck);
app.post('/api/v1/predict', ThreatController.predict);
app.get('/api/v1/logs', ThreatController.getLogs);

// 404 Route Handler
app.use('*', (req, res, next) => {
  next(new AppError(`The requested endpoint ${req.originalUrl} was not found on this server.`, 404));
});

// RFC 7807 Global Error Handler Middleware
app.use(errorHandler);

// Connect to Database & Start Server
const PORT = env.PORT || 5000;

if (process.env.NODE_ENV !== 'test') {
  connectDB().then(() => {
    app.listen(PORT, () => {
      console.log(`🚀 Clean Architecture Enterprise SaaS Server running on port ${PORT} [${env.NODE_ENV}]`);
    });
  });
}

export default app;
