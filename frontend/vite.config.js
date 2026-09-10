import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api and /jobs proxy to FastAPI so the browser sees one origin in dev and
// there is no CORS dance. In production `npm run build` writes web/dist, which
// server.py mounts at / — one process, one port.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true }
    }
  },
  build: { outDir: 'dist', chunkSizeWarningLimit: 1500 }
})
