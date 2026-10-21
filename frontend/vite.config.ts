import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Keeps compiled asset paths relative under the SageMaker JupyterLab proxy.
  base: './',
})
