/**
 * Smart Environment API URL Resolver
 * Automatically resolves the active backend API endpoint depending on environment and host window domain.
 */
export function getApiBaseUrl(): string {
  // If explicitly provided via environment variable
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }

  // Client-side dynamic host resolution
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    if (hostname.endsWith('.vercel.app')) {
      return `https://${hostname.replace('.vercel.app', '-backend.onrender.com')}/api/v1`;
    }
    if (hostname.endsWith('.onrender.com')) {
      return `${window.location.origin}/api/v1`;
    }
  }

  // Local development fallback
  return 'http://localhost:5000/api/v1';
}

export const API_BASE_URL = getApiBaseUrl();

/**
 * Universal JSON Fetch Helper with error handling
 */
export async function apiFetch(endpoint: string, options: RequestInit = {}) {
  const baseUrl = getApiBaseUrl();
  const url = endpoint.startsWith('http') ? endpoint : `${baseUrl}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;

  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail = `API Error (${response.status})`;
    try {
      const errorData = await response.json();
      errorDetail = errorData.detail || errorData.message || errorDetail;
    } catch {
      // JSON parse failed
    }
    throw new Error(errorDetail);
  }

  return response.json();
}
