/**
 * Standardized RFC 7807 Compliant Global Error Handler Middleware
 * Spec: https://tools.ietf.org/html/rfc7807
 */

export class AppError extends Error {
  constructor(message, statusCode = 500, type = 'https://tools.ietf.org/html/rfc7231#section-6.6.1') {
    super(message);
    this.statusCode = statusCode;
    this.type = type;
    this.isOperational = true;
    Error.captureStackTrace(this, this.constructor);
  }
}

export const errorHandler = (err, req, res, next) => {
  const statusCode = err.statusCode || 500;
  
  const problemDetails = {
    type: err.type || 'https://tools.ietf.org/html/rfc7231#section-6.6.1',
    title: getTitleForStatus(statusCode),
    status: statusCode,
    detail: err.message || 'An unexpected error occurred processing your request.',
    instance: req.originalUrl || req.url,
    timestamp: new Date().toISOString(),
  };

  if (process.env.NODE_ENV === 'development' && err.stack) {
    problemDetails.stack = err.stack;
  }

  console.error(`❌ [RFC 7807 Error] ${statusCode} ${req.method} ${req.originalUrl}:`, err.message);

  res.setHeader('Content-Type', 'application/problem+json');
  return res.status(statusCode).json(problemDetails);
};

function getTitleForStatus(status) {
  switch (status) {
    case 400: return 'Bad Request';
    case 401: return 'Unauthorized';
    case 403: return 'Forbidden';
    case 404: return 'Not Found';
    case 422: return 'Unprocessable Entity';
    case 429: return 'Too Many Requests';
    default: return 'Internal Server Error';
  }
}
