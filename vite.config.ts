import { defineConfig } from 'vitest/config';

export default defineConfig({
  // Relative asset URLs so the build works under a GitHub Pages project path (/<repo>/).
  base: './',
  build: { target: 'es2022', chunkSizeWarningLimit: 4000 },
  test: { environment: 'node', testTimeout: 60_000 },
});
