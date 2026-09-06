import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const adapter = env.VITE_START_ADAPTER || process.env.VITE_START_ADAPTER || 'public'
  const modelBase = env.VITE_START_MODEL_BASE || process.env.VITE_START_MODEL_BASE

  if (mode === 'production' && adapter !== 'demo') {
    if (!modelBase) {
      throw new Error(
        'Production build validation failed: VITE_START_MODEL_BASE is required for production builds. Silent Hugging Face fallback is prohibited.'
      )
    }
  }

  return {
    plugins: [react()],
    server: { port: 4173 },
    build: { sourcemap: true },
  }
})

