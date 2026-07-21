import { z } from 'zod';

const frontendEnvSchema = z.object({
  NEXT_PUBLIC_API_URL: z.string().url().or(z.string().min(1)).optional().default('http://localhost:5000'),
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
});

const getEnv = () => {
  try {
    return frontendEnvSchema.parse({
      NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
      NODE_ENV: process.env.NODE_ENV || 'development',
    });
  } catch (err) {
    console.warn('⚠️ Frontend environment validation warning:', err);
    return {
      NEXT_PUBLIC_API_URL: 'http://localhost:5000',
      NODE_ENV: 'development' as const,
    };
  }
};

export const env = getEnv();
export default env;
