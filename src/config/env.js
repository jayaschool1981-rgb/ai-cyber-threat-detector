import { z } from 'zod';
import dotenv from 'dotenv';

dotenv.config();

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.coerce.number().default(5000),
  MONGO_URI: z.string().default('mongodb://localhost:27017/threat_detection'),
  DATABASE_URL: z.string().optional(),
  OPENROUTER_API_KEY: z.string().optional().default('mock-openrouter-key'),
  OPENAI_API_KEY: z.string().optional().default('mock-openai-key'),
  GEMINI_API_KEY: z.string().optional().default('mock-gemini-key'),
  ALLOWED_ORIGINS: z.string().optional().default('http://localhost:3000,http://localhost:5000')
});

let parsedEnv;

try {
  parsedEnv = envSchema.parse(process.env);
} catch (error) {
  if (error instanceof z.ZodError) {
    console.error('❌ Environment Variable Validation Failed:');
    console.error(JSON.stringify(error.format(), null, 2));
  } else {
    console.error('❌ Environment Variable parsing failed:', error);
  }
  // Provide safe fallback defaults so app never crashes unexpectedly in test environments
  parsedEnv = envSchema.parse({});
}

export const env = parsedEnv;
export default env;
