import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Single-page app. In development `/api` is proxied to the FastAPI service in
// ./server (uvicorn server.app:app --port 8000), which is the only thing that
// talks to the swmmanywhere Python package.
export default defineConfig({
  base: process.env.VITE_BASE || '/',
  plugins: [react(), tailwindcss()],
  server: {
    port: 5174,
    proxy: { '/api': 'http://localhost:8000' },
  },
})
