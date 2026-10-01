import * as tf from '@tensorflow/tfjs';
import { requestGpu } from '../../src/render/gpu';
import { initTensorflow } from '../../src/tf/backend';
import { blockUpdateStep } from '../../src/tf/blockUpdate';
import { blockUpdateReference } from '../../src/tf/blockUpdateReference';
import { checkFusedKernel, mulberry32, randomCells } from '../../src/tf/kernelCheck';

// Block-update rules on a real WebGPU backend, checked against the plain-JS reference.
// Run with `npm run test:gpu` (test/gpu/run.mjs serves this page in headless Chromium).

interface Result { name: string; ok: boolean; detail: string }
const results: Result[] = [];
const out = document.getElementById('out')!;
const report = (name: string, ok: boolean, detail: string) => {
  results.push({ name, ok, detail });
  out.textContent = results.map((r) => `${r.ok ? 'PASS' : 'FAIL'} ${r.name}${r.detail ? ` — ${r.detail}` : ''}`).join('\n');
};

/** The tensor-op rules (used on other backends) run on WebGPU, same random numbers as the reference. */
async function checkOpsStep(seed: number): Promise<Result> {
  const rand = mulberry32(seed);
  const N = 2, H = 9, D = 11, W = 13, size = H * D * W;
  const rates = { grassSpread: 0.25, wheatGrow: 0.125, wheatGrowWet: 0.5 };
  let states = Array.from({ length: N }, () => randomCells(size, rand));
  let mismatches = 0, cells = 0;
  for (let tick = 0; tick < 4; tick++) {
    const randoms = states.map(() => Float32Array.from({ length: size }, rand));
    const expected = states.map((s, k) => blockUpdateReference(s, H, D, W, randoms[k], rates));
    const input = tf.tensor4d(Int32Array.from(states.flatMap((s) => [...s])), [N, H, D, W], 'int32');
    const random = tf.tensor4d(Float32Array.from(randoms.flatMap((r) => [...r])), [N, H, D, W]);
    const output = blockUpdateStep(input, random, rates);
    const got = (await output.data()) as Int32Array;
    tf.dispose([input, random, output]);
    expected.forEach((e, k) => e.forEach((v, i) => { cells++; if (got[k * size + i] !== v) mismatches++; }));
    states = expected;
  }
  return { name: `tensor-op rules match the reference (seed ${seed})`, ok: mismatches === 0, detail: `${mismatches} of ${cells} cells differ` };
}

async function main(): Promise<void> {
  if (!navigator.gpu || !(await navigator.gpu.requestAdapter())) {
    (window as unknown as { gpuTests: unknown }).gpuTests = { skipped: 'no WebGPU adapter', results };
    out.textContent = 'SKIP no WebGPU adapter';
    return;
  }
  const { device, info } = await requestGpu();
  const backend = await initTensorflow(device, info);
  report('TF.js runs on WebGPU', backend === 'webgpu', `backend ${backend}`);
  for (const seed of [1, 2, 3]) {
    const r = await checkFusedKernel(seed);
    report(`fused kernel matches the reference (seed ${seed})`, r.ok, r.ok ? `${r.cells} cells` : `${r.mismatches} of ${r.cells} differ; first: ${r.detail}`);
  }
  for (const seed of [4, 5]) {
    const r = await checkOpsStep(seed);
    report(r.name, r.ok, r.detail);
  }
  const before = tf.memory().numTensors;
  await checkFusedKernel(9, 2);
  report('fused kernel leaves no tensors behind', tf.memory().numTensors === before, `${tf.memory().numTensors - before} left`);
}

main()
  .catch((e) => report('harness', false, e instanceof Error ? e.message : String(e)))
  .finally(() => {
    const w = window as unknown as { gpuTests?: unknown };
    w.gpuTests ??= { results };
  });
