import {defineConfig} from 'vite';

export default defineConfig({
  build: {
    outDir: '../src/massive_scatter/_viewer',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
});
