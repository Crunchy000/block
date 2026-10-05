import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';
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

const BUILD = buildId();

/**
 * Cache busting: each build writes version.json (the page checks it, past every cache, and
 * reloads into a newer build: src/assetUrl.ts), and the pages load public/log.js with the
 * build in its URL. (The bundled scripts already have content hashes in their names.)
 */
function cacheBusting(): Plugin {
  return {
    name: 'cache-busting',
    transformIndexHtml: (html) => html.replace(/src="\.\/log\.js"/g, `src="./log.js?v=${BUILD}"`),
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'version.json', source: JSON.stringify({ build: BUILD }) });
    },
  };
}

export default defineConfig({
  define: { __BUILD__: JSON.stringify(BUILD) },
  plugins: [cacheBusting()],
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
