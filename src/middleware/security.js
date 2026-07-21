import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';

// Security Headers Middleware
export const securityHeaders = helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", 'data:', 'https:'],
    },
  },
  crossOriginEmbedderPolicy: false,
});

// Dynamic CORS Whitelist Authorization
export const corsOptions = {
  origin: (origin, callback) => {
    // Allow non-browser requests or CLI requests (no origin)
    if (!origin) return callback(null, true);

    const isAllowed = 
      origin.includes('localhost') ||
      origin.includes('127.0.0.1') ||
      origin.endsWith('.vercel.app') ||
      origin.endsWith('.onrender.com');

    if (isAllowed) {
      callback(null, true);
    } else {
      callback(new Error(`CORS policy rejection: Origin ${origin} is not allowed`));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
};

export const corsMiddleware = cors(corsOptions);

// IP-Based Rate Limiting (100 requests per 15 minutes)
export const apiRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    type: 'https://tools.ietf.org/html/rfc7231#section-6.5.4',
    title: 'Too Many Requests',
    status: 429,
    detail: 'Rate limit exceeded. Maximum 100 requests allowed per 15 minutes window.',
    instance: '/api/v1/rate-limit-exceeded',
  },
});
