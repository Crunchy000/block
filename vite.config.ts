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
 * The first thing each page runs, inline in its HTML: fetch version.json past every cache and,
 * if a newer build is deployed than this page, reload into it with ?v=<build> in the URL (a URL
 * the browser hasn't cached). Inline because GitHub Pages lets browsers keep a page's HTML for
 * 10 minutes, and a cached page may name scripts the newer deploy has removed. Once only: if
 * the URL already asks for that build, it doesn't reload again.
 */
const VERSION_CHECK = `<script>
      (function () {
        var built = ${JSON.stringify(BUILD)};
        fetch('version.json', { cache: 'no-store' })
          .then(function (r) { return r.ok ? r.json() : null; })
          .then(function (v) {
            if (!v || !v.build || v.build === built) return;
            var url = new URL(location.href);
            if (url.searchParams.get('v') === v.build) return;
            url.searchParams.set('v', v.build);
            location.replace(url.toString());
          })
          .catch(function () {});
      })();
    </script>`;

/**
 * Cache busting: each build writes version.json, which every page checks first (VERSION_CHECK),
 * and the pages load public/log.js with the build in its URL. (The bundled scripts already have
 * content hashes in their names; src/assetUrl.ts versions the other public files.)
 */
function cacheBusting(): Plugin {
  return {
    name: 'cache-busting',
    transformIndexHtml: (html) => html
      .replace('<head>', `<head>\n    ${VERSION_CHECK}`)
      .replace(/src="\.\/log\.js"/g, `src="./log.js?v=${BUILD}"`),
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
