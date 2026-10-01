import { defineConfig } from 'vitest/config';

export default defineConfig({
  build: { target: 'es2022', chunkSizeWarningLimit: 4000 },
  test: { environment: 'node', testTimeout: 60_000 },
});
