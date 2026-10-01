import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

/** The commit a build is from, shown in the page log. */
function buildId(): string {
  if (process.env.GITHUB_SHA) return process.env.GITHUB_SHA.slice(0, 7);
  try {
    return execSync('git rev-parse --short HEAD').toString().trim();
  } catch {
    return 'dev';
  }
}

export default defineConfig({
  define: { __BUILD__: JSON.stringify(buildId()) },
  // Relative asset URLs so the build works under a GitHub Pages project path (/<repo>/).
  base: './',
  build: {
    target: 'es2022',
    // Source maps, so stack traces copied from the page log can be mapped back to the code.
    sourcemap: true,
    chunkSizeWarningLimit: 4000,
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        bench: fileURLToPath(new URL('./bench.html', import.meta.url)),
        log: fileURLToPath(new URL('./log.html', import.meta.url)),
      },
    },
  },
  test: { environment: 'node', testTimeout: 60_000 },
});
