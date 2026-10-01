// Runs test/gpu/index.html in headless Chromium with WebGPU (software Vulkan is fine)
// and reports its results. Exits 1 on any failure; reports SKIP if no WebGPU adapter.
//   npm run test:gpu                 (CHROMIUM_PATH=... to use a specific browser binary)
import { chromium } from 'playwright';
import { createServer } from 'vite';

const server = await createServer({ configFile: './vite.config.ts', server: { port: 0, host: '127.0.0.1' }, logLevel: 'error' });
await server.listen();
const url = `${server.resolvedUrls.local[0]}test/gpu/index.html`;
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || undefined,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist'],
});
let failed = false;
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => { console.error('[page error]', e.message); failed = true; });
  await page.goto(url);
  await page.waitForFunction(() => window.gpuTests !== undefined, null, { timeout: 600_000, polling: 500 });
  const { skipped, results } = await page.evaluate(() => window.gpuTests);
  if (skipped) console.log(`SKIP GPU tests: ${skipped}`);
  for (const r of results) {
    console.log(`${r.ok ? 'PASS' : 'FAIL'} ${r.name}${r.detail ? ` — ${r.detail}` : ''}`);
    if (!r.ok) failed = true;
  }
} finally {
  await browser.close();
  await server.close();
}
process.exit(failed ? 1 : 0);
