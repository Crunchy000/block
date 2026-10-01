import * as tf from '@tensorflow/tfjs';
import { requestGpu } from '../../src/render/gpu';
import { initTensorflow } from '../../src/tf/backend';
import { blockUpdateStep } from '../../src/tf/blockUpdate';
import { blockUpdateReference } from '../../src/tf/blockUpdateReference';
import { Block, CHUNK_SIZE, CHUNK_VOLUME, SOURCE_LEVEL, cell } from '../../src/constants';
import { MeshPool } from '../../src/render/meshPool';
import { checkGpuStore } from '../../src/sim/check';
import { CpuStore } from '../../src/sim/cpuStore';
import { GpuStore } from '../../src/sim/gpuStore';
import { Simulation } from '../../src/sim/simulation';
import { ringSize } from '../../src/sim/store';
import { generateAll } from '../../src/world/loader';
import { World, meshSlotCount } from '../../src/world/world';
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

/**
 * The game's World and Simulation on the GPU store and on the reference store, put through
 * the same session: TF.js worldgen (straight into GPU memory for one, through the CPU for
 * the other), edits, ticks, moving away (edited chunks saved) and back (restored), meshing.
 * Everything must stay identical.
 */
async function checkWorlds(device: GPUDevice): Promise<Result> {
  const name = 'game world on the GPU matches the reference world through a session';
  const rates = { grassSpread: 0.5, wheatGrow: 0.3, wheatGrowWet: 0.6 };
  const make = async (store: GpuStore | CpuStore) => {
    const world = new World(store, 1, 2), sim = new Simulation(world, rates);
    let n = 0;
    sim.seed = () => (++n * 2654435761) >>> 0;
    return { world, sim, store };
  };
  const gpuStore = await GpuStore.create(device, ringSize(2));
  const a = await make(gpuStore), b = await make(new CpuStore(ringSize(2)));
  const both = async (f: (w: typeof a | typeof b) => Promise<unknown> | unknown) => { await f(a); await f(b); };
  let compared = 0;
  const compare = async (when: string) => {
    const keys = [...a.world.chunks.keys()];
    if (keys.join() !== [...b.world.chunks.keys()].join()) throw new Error(`${when}: different chunks in the halo`);
    for (const key of keys) {
      const ca = a.world.chunks.get(key)!, cb = b.world.chunks.get(key)!;
      if (ca.loaded !== cb.loaded || ca.version !== cb.version || ca.modified !== cb.modified) throw new Error(`${when}: chunk ${key} state differs`);
      if (!ca.loaded) continue;
      const [x, y] = await Promise.all([a.store.readChunk(ca.slot), b.store.readChunk(cb.slot)]);
      const i = x.findIndex((v, k) => v !== y[k]);
      if (i >= 0) throw new Error(`${when}: chunk ${key} cell ${i} is ${x[i]}, expected ${y[i]}`);
      compared += CHUNK_VOLUME;
    }
  };
  const ticks = async (count: number, when: string) => {
    for (let t = 0; t < count; t++) {
      await both((w) => w.sim.tick());
      for (const k of ['lastBatch', 'lastChangedChunks', 'growingChunks', 'asleep'] as const) {
        if (a.sim[k] !== b.sim[k]) throw new Error(`${when} tick ${t}: ${k} ${a.sim[k]}, expected ${b.sim[k]}`);
      }
    }
    await compare(`${when}, after ${count} ticks`);
  };
  const waitLoaded = async () => {
    for (let i = 0; i < 200 && !(a.world.haloReady() && b.world.haloReady()); i++) await new Promise((r) => setTimeout(r, 10));
  };
  const pool = new MeshPool(device, meshSlotCount(1));
  try {
    await both((w) => generateAll(w.world));
    await compare('generated');
    // Water on a platform, lava at a chunk corner, grass on dirt across four chunks, wheat by water.
    await both(({ world }) => {
      for (let z = 4; z <= 12; z++) for (let x = 4; x <= 12; x++) world.setCell(x, 45, z, cell(Block.Stone));
      world.setCell(8, 46, 8, cell(Block.Water, SOURCE_LEVEL));
      world.setCell(15, 45, 15, cell(Block.Stone));
      world.setCell(15, 46, 15, cell(Block.Lava, SOURCE_LEVEL));
      for (let z = -3; z <= 3; z++) for (let x = -3; x <= 3; x++) world.setCell(x, 50, z, cell(Block.Dirt));
      world.setCell(0, 50, 0, cell(Block.Grass));
      world.setCell(20, 45, 4, cell(Block.Dirt));
      world.setCell(20, 46, 4, cell(Block.Wheat, 0));
      world.setCell(21, 46, 4, cell(Block.Water, SOURCE_LEVEL));
    });
    await ticks(24, 'edited');
    // Away (the edited chunks leave the halo and are saved) and back (restored).
    await both(({ world }) => world.recenter(CHUNK_SIZE * 4 + 8, 8));
    await both((w) => generateAll(w.world));
    if (a.world.savedCount() !== b.world.savedCount() || a.world.savedCount() === 0) throw new Error(`saved ${a.world.savedCount()} chunks, expected ${b.world.savedCount()}`);
    await ticks(4, 'moved away');
    await both(({ world }) => world.recenter(8, 8));
    await both((w) => generateAll(w.world));
    await waitLoaded();
    await compare('moved back');
    await ticks(8, 'moved back');
    // Meshes of every active chunk.
    while (a.world.remesh(pool) > 0); // a few chunks per call
    let faces = 0;
    for (const c of a.world.activeChunks()) {
      const got = await pool.read(c.meshSlot), want = (b.store as CpuStore).meshChunk(b.world.around(b.world.getChunk(c.cx, c.cz)!));
      for (const kind of ['opaque', 'water'] as const) {
        faces += want[kind].length;
        if ([...got[kind]].sort().join() !== [...want[kind]].sort().join()) throw new Error(`mesh of chunk ${c.key}: ${kind} faces differ`);
      }
    }
    return { name, ok: true, detail: `${compared.toLocaleString()} cells, ${faces.toLocaleString()} faces compared` };
  } catch (e) {
    return { name, ok: false, detail: e instanceof Error ? e.message : String(e) };
  } finally {
    gpuStore.destroy();
    pool.destroy();
  }
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
  for (const seed of [1, 2, 3]) {
    const r = await checkGpuStore(device, seed, 4, 256);
    report(`GPU-resident world matches the reference: ticks, flags, meshes, picking (seed ${seed})`, r.ok, r.ok ? r.summary : r.detail);
  }
  const worlds = await checkWorlds(device);
  report(worlds.name, worlds.ok, worlds.detail);
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
