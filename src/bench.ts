import * as tf from '@tensorflow/tfjs';
import {
  Block, CHUNK_HEIGHT, CHUNK_VOLUME, DEFAULT_RATES, WHEAT_RIPE, cellLevel, cellType,
} from './constants';
import { requestGpu } from './render/gpu';
import { initTensorflow, warmUpKernels } from './tf/backend';
import { blockUpdateStep } from './tf/blockUpdate';
import { blockUpdateFused, fusedAvailable, randomSeed } from './tf/blockUpdateKernel';
import { checkFusedKernel, type KernelCheck } from './tf/kernelCheck';
import { randomField } from './tf/random';
import { HALO, PADDED, Simulation, packChunk } from './tf/simulation';
import { GEN_BATCH, generateChunks } from './tf/worldgen';
import type { Chunk } from './world/chunk';
import { World } from './world/world';

// Measures how many block updates per second this device computes with the game's
// code. One block update = working out one cell's next state for one tick; only real
// chunk cells count, not the ghost border cells each chunk is padded with.

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);
/** How long each measurement runs (?quick shortens it, for slow test machines). */
const BUDGET_MS = params.has('quick') ? 300 : 2000;
/** Batch sizes to try. 49 is every active chunk; bigger batches repeat chunks. */
const BATCHES = [1, 4, 16, 49, 100, 200];
/** The tensor-op rules make ~100 full-size temporaries per tick; past this they need too much GPU memory. */
const OPS_MAX_BATCH = 100;
const PADDED_VOLUME = CHUNK_HEIGHT * PADDED * PADDED;
const TICKS_PER_SECOND = 5;
/** Receives the result of the write-back comparison so the JIT can't drop it as dead code. */
let compareSink = 0;

interface Setup { backend: string; gpu: string; world: World }
let setup: Promise<Setup> | undefined;

const status = (text: string, error = false) => {
  const el = $('status');
  el.textContent = text;
  el.className = error ? 'error' : 'muted';
};
const progress = (fraction: number) => { $('bar').style.width = `${Math.round(fraction * 100)}%`; };

/** GPU, TF.js, compiled kernels and the terrain around spawn: once per page. */
function prepare(): Promise<Setup> {
  setup ??= (async () => {
    status('Starting WebGPU and TensorFlow.js…');
    const { device, info } = await requestGpu();
    const backend = await initTensorflow(device, info);
    status('Compiling GPU kernels…');
    await warmUpKernels();
    status('Generating terrain…');
    const world = await loadWorld();
    const gpu = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'unknown GPU';
    return { backend, gpu, world };
  })();
  setup.catch(() => { setup = undefined; }); // allow a retry
  return setup;
}

/** The game's view around spawn: 7x7 active chunks plus the ghost ring. */
async function loadWorld(): Promise<World> {
  const world = new World();
  world.recenter(8, 8);
  for (;;) {
    const batch = world.missingChunks(GEN_BATCH);
    if (batch.length === 0) return world;
    world.markGenerating(batch);
    const data = await generateChunks(batch);
    batch.forEach((c, i) => world.addGenerated(c, data[i]));
  }
}

interface Rate {
  /** Wall time per tick. */
  ms: number;
  /** Main-thread time per tick (packing, issuing the ops, comparing the results). */
  mainMs: number;
  updatesPerSecond: number;
}

const batchOf = (world: World, n: number): Chunk[] => {
  const active = world.activeChunks();
  return Array.from({ length: n }, (_, i) => active[i % active.length]);
};

/** Which implementation of the rules: the fused WebGPU kernel, or the tensor-op version. */
type Rules = 'fused' | 'ops';

/** One tick of the rules on a padded batch; `halo` strips the ghost border from the output. */
function step(cells: tf.Tensor4D, plants: boolean, rules: Rules, halo: number): tf.Tensor4D {
  if (rules === 'fused') return blockUpdateFused(cells, { seed: randomSeed(), plants, halo });
  const seeds = [Math.random() * 999, Math.random() * 999, Math.random() * 999] as const;
  const out = blockUpdateStep(cells, plants ? randomField(cells.shape, seeds) : undefined);
  const [n, h, d, w] = cells.shape;
  return halo ? out.slice([0, 0, halo, halo], [n, h, d - 2 * halo, w - 2 * halo]) : out;
}

/** Run `tick` repeatedly for BUDGET_MS (after one warm-up run) and work out the rate. */
async function measure(n: number, tick: () => Promise<number>): Promise<Rate> {
  await tick();
  let ticks = 0, mainMs = 0;
  const start = performance.now();
  do {
    mainMs += await tick();
    ticks++;
  } while (performance.now() - start < BUDGET_MS);
  const ms = (performance.now() - start) / ticks;
  return { ms, mainMs: mainMs / ticks, updatesPerSecond: (n * CHUNK_VOLUME * 1000) / ms };
}

/**
 * The game's tick on a fixed batch of chunks: pack each with its ghost border, run the
 * step, read the interiors back and compare them with the chunks (the world isn't changed).
 */
function gameLoop(world: World, n: number, plants: boolean, rules: Rules): Promise<Rate> {
  const chunks = batchOf(world, n);
  const cells = new Int32Array(n * PADDED_VOLUME);
  return measure(n, async () => {
    const t0 = performance.now();
    chunks.forEach((c, i) => packChunk(world, c, cells, i * PADDED_VOLUME));
    const out = tf.tidy(() => step(tf.tensor4d(cells, [n, CHUNK_HEIGHT, PADDED, PADDED], 'int32'), plants, rules, HALO));
    const t1 = performance.now();
    const next = (await out.data()) as Int32Array;
    out.dispose();
    const t2 = performance.now();
    chunks.forEach((c, k) => {
      for (let i = 0, base = k * CHUNK_VOLUME; i < CHUNK_VOLUME; i++) if (next[base + i] !== c.data[i]) compareSink++;
    });
    return t1 - t0 + (performance.now() - t2);
  });
}

/**
 * Steps chained on the GPU with no packing or reading back in between: the ceiling if
 * the world's state stayed on the GPU. Each measured round is 4 steps, then one sync.
 */
async function gpuOnly(world: World, n: number, plants: boolean, rules: Rules): Promise<Rate> {
  const cells = new Int32Array(n * PADDED_VOLUME);
  batchOf(world, n).forEach((c, i) => packChunk(world, c, cells, i * PADDED_VOLUME));
  let state = tf.tensor4d(cells, [n, CHUNK_HEIGHT, PADDED, PADDED], 'int32');
  try {
    return await measure(n * 4, async () => {
      const t0 = performance.now();
      for (let k = 0; k < 4; k++) {
        const next = tf.tidy(() => step(state, plants, rules, 0));
        state.dispose();
        state = next;
      }
      const t1 = performance.now();
      const probe = state.slice([0, 0, 0, 0], [1, 1, 1, 1]);
      await probe.data(); // wait for the GPU to finish
      probe.dispose();
      return t1 - t0;
    });
  } finally {
    state.dispose();
  }
}

const fmtRate = (v: number) =>
  v >= 1e9 ? `${(v / 1e9).toFixed(2)} billion` : v >= 1e6 ? `${(v / 1e6).toFixed(1)} million` : `${Math.round(v / 1e3)} thousand`;
const fmtMs = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1));

async function runBenchmark(): Promise<void> {
  const { backend, gpu, world } = await prepare();
  // Check the fused kernel against the reference first: only measure what computes correctly.
  let check: KernelCheck | undefined;
  if (fusedAvailable()) {
    status('Checking the fused kernel against the reference rules…');
    check = await checkFusedKernel();
  }
  const fused = check?.ok === true;
  const opsBatches = BATCHES.filter((n) => n <= OPS_MAX_BATCH);
  const steps = (fused ? BATCHES.length * 2 + 1 : 1) + opsBatches.length;
  let done = 0;
  const next = (what: string, n: number) => {
    progress(done / steps);
    status(`${what}, ${n} chunk${n === 1 ? '' : 's'} (${done + 1} of ${steps})…`);
    done++;
  };
  const rows: Row[] = BATCHES.map((n) => ({ n }));
  if (fused) {
    for (const row of rows) { next('Fused kernel, game tick loop', row.n); row.fused = await gameLoop(world, row.n, true, 'fused'); }
    for (const row of rows) { next('Fused kernel, GPU only', row.n); row.gpu = await gpuOnly(world, row.n, true, 'fused'); }
  }
  for (const row of rows.filter((r) => r.n <= OPS_MAX_BATCH)) {
    next('Tensor-op rules, game tick loop', row.n);
    row.ops = await gameLoop(world, row.n, true, 'ops');
  }
  const rules: Rules = fused ? 'fused' : 'ops';
  const best = bestRow(rows, rules);
  next('Fluid rules only', best.n);
  const fluids = await gameLoop(world, best.n, false, rules);
  progress(1);
  status(`Done. ${backend} on ${gpu}.`);
  showResult(rows, rules, fluids, check, backend, gpu);
}

interface Row { n: number; fused?: Rate; gpu?: Rate; ops?: Rate }

const rateOf = (row: Row, rules: Rules) => (rules === 'fused' ? row.fused : row.ops)?.updatesPerSecond ?? 0;
const bestRow = (rows: Row[], rules: Rules) => rows.reduce((b, r) => (rateOf(r, rules) > rateOf(b, rules) ? r : b));

function showResult(rows: Row[], rules: Rules, fluids: Rate, check: KernelCheck | undefined, backend: string, gpu: string): void {
  const best = bestRow(rows, rules), top = rateOf(best, rules);
  const chunksAt5 = Math.floor(top / (CHUNK_VOLUME * TICKS_PER_SECOND));
  const opsBest = Math.max(...rows.map((r) => r.ops?.updatesPerSecond ?? 0));
  const gpuTop = Math.max(...rows.map((r) => r.gpu?.updatesPerSecond ?? 0));
  const cellRate = (r?: Rate) => (r ? fmtRate(r.updatesPerSecond) : '—');
  const rowsHtml = rows.map((r) => {
    const main = rules === 'fused' ? r.fused : r.ops;
    return `
    <tr${r === best ? ' class="best"' : ''}>
      <td>${r.n}</td>
      <td>${main ? `${fmtMs(main.ms)} ms` : '—'}</td>
      <td>${main ? `${fmtMs(main.mainMs)} ms` : '—'}</td>
      <td>${cellRate(r.fused)}</td>
      <td>${cellRate(r.gpu)}</td>
      <td>${cellRate(r.ops)}</td>
    </tr>`;
  }).join('');
  const checkText = !check
    ? `The fused kernel needs WebGPU; this backend (${backend}) uses the tensor-op rules.`
    : check.ok
      ? `Rules check: the fused GPU kernel matched the reference rules on all ${check.cells.toLocaleString()} cells.`
      : `Rules check failed, so the tensor-op rules were measured instead: ${check.mismatches} of ${check.cells} cells differ (first: ${check.detail}).`;
  const el = $('result');
  el.hidden = false;
  el.innerHTML = `
    <div class="headline">${fmtRate(top)} <span class="unit">block updates per second</span></div>
    <p class="muted">The game's tick loop with every rule (fluids, grass, wheat) on the
      ${rules === 'fused' ? 'fused GPU kernel' : 'tensor-op rules'}: packing chunks with their ghost borders, one tick on the GPU,
      reading results back. Best with ${best.n} chunks per tick.</p>
    <div class="stats">
      <div class="stat"><b>${chunksAt5.toLocaleString()} chunks</b>could keep updating at ${TICKS_PER_SECOND} ticks per second (the view has 49)</div>
      ${rules === 'fused' && opsBest > 0 ? `<div class="stat"><b>${(top / opsBest).toFixed(1)}× faster</b>than the tensor-op rules (${fmtRate(opsBest)}/s at best)</div>` : ''}
      ${gpuTop > 0 ? `<div class="stat"><b>${fmtRate(gpuTop)}/s</b>GPU only: ticks chained with the state kept on the GPU</div>` : ''}
      <div class="stat"><b>${fmtRate(fluids.updatesPerSecond)}/s</b>fluid rules only (no grass or wheat), ${best.n} chunks</div>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>chunks per tick</th><th>tick</th><th>main thread</th><th>fused kernel</th><th>fused, GPU only</th><th>tensor-op rules</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table></div>
    <p class="muted" style="margin-top:12px">${checkText}</p>
    <p class="muted">Updates per second for each way of running the rules. One block update = one cell's next
      state for one tick, counting the 16×16×64 cells of each chunk (not its ghost border). Each figure is the
      average over ~${BUDGET_MS / 1000} s. Batches over 49 chunks repeat chunks; the tensor-op rules stop at
      ${OPS_MAX_BATCH} (they need too much GPU memory beyond that). ${backend} on ${gpu}.</p>`;
}

async function runTakeover(): Promise<void> {
  const log = $('log');
  const write = (line = '') => { log.textContent += line + '\n'; };
  log.textContent = '';
  await prepare();
  status('Generating fresh terrain for the takeover…');
  const world = await loadWorld(); // a fresh world: the takeover changes it
  const chance = (name: string, fallback: number) =>
    params.has(name) ? Math.min(1, Math.max(0, Number(params.get(name)))) : fallback;
  const rates = {
    grassSpread: chance('spread', DEFAULT_RATES.grassSpread),
    wheatGrow: chance('grow', DEFAULT_RATES.wheatGrow),
    wheatGrowWet: chance('grow', DEFAULT_RATES.wheatGrowWet),
  };
  const maxTicks = Number(params.get('ticks') ?? 600);
  write(`Chances per tick: grass ${rates.grassSpread.toFixed(4)}, wheat ${rates.wheatGrow.toFixed(4)} (wet ${rates.wheatGrowWet.toFixed(4)}). Up to ${maxTicks} ticks.`);
  write('Only awake chunks are simulated: grass still able to spread, unripe wheat, or a change next door.');
  write();
  write('  tick  game time   batch   tick ms   main thread   growing   grass   wheat ripe');
  const sim = new Simulation(world, rates);
  const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  const row = (tick: number, tickMs: string, cpuMs: string) => {
    let grass = 0, wheat = 0, ripe = 0;
    for (const c of world.activeChunks()) {
      for (const v of c.data) {
        if (cellType(v) === Block.Grass) grass++;
        else if (cellType(v) === Block.Wheat) { wheat++; if (cellLevel(v) === WHEAT_RIPE) ripe++; }
      }
    }
    const ripeText = wheat ? `${Math.round((100 * ripe) / wheat)}% of ${wheat}` : 'none';
    write(`${String(tick).padStart(6)} ${`${(tick / TICKS_PER_SECOND).toFixed(0)} s`.padStart(10)} ${String(sim.lastBatch).padStart(7)} ${tickMs.padStart(9)} ${cpuMs.padStart(13)} ${String(sim.growingChunks).padStart(9)} ${String(grass).padStart(7)}   ${ripeText}`);
  };
  const started = performance.now();
  let ticks = 0, totalMs = 0, totalCpu = 0, chunkTicks = 0, windowMs: number[] = [], windowCpu: number[] = [];
  row(0, '-', '-');
  while (ticks < maxTicks) {
    status(`Takeover: tick ${ticks + 1}…`);
    progress(ticks / maxTicks);
    if (!(await sim.tick())) break; // everything asleep: settled
    ticks++;
    totalMs += sim.lastTickMs; totalCpu += sim.lastCpuMs; chunkTicks += sim.lastBatch;
    windowMs.push(sim.lastTickMs); windowCpu.push(sim.lastCpuMs);
    if (ticks % 50 === 0) {
      row(ticks, fmtMs(median(windowMs)), fmtMs(median(windowCpu)));
      windowMs = []; windowCpu = [];
    }
  }
  if (windowMs.length) row(ticks, fmtMs(median(windowMs)), fmtMs(median(windowCpu)));
  write();
  write(`${ticks} ticks in ${((performance.now() - started) / 1000).toFixed(1)} s${sim.asleep ? ', then everything settled and went to sleep' : ''}.`);
  if (ticks) {
    write(`Average tick ${fmtMs(totalMs / ticks)} ms (main thread ${fmtMs(totalCpu / ticks)} ms), ${(chunkTicks / ticks).toFixed(1)} chunks per tick,`);
    write(`${fmtRate((chunkTicks * CHUNK_VOLUME * 1000) / totalMs)} block updates per second while ticking.`);
  }
  progress(1);
  status('Takeover done.');
}

/** Run one job at a time with both buttons disabled, and report failures in the status line. */
function wire(button: HTMLButtonElement, job: () => Promise<void>): () => Promise<void> {
  const run = async () => {
    const buttons = [$<HTMLButtonElement>('start'), $<HTMLButtonElement>('takeover')];
    buttons.forEach((b) => { b.disabled = true; });
    progress(0);
    try {
      await job();
    } catch (e) {
      console.error(e);
      status(`Failed: ${e instanceof Error ? e.message : String(e)}`, true);
    } finally {
      buttons.forEach((b) => { b.disabled = false; });
      $('start').textContent = 'Run again';
    }
  };
  button.addEventListener('click', run);
  return run;
}

const start = wire($('start'), runBenchmark);
wire($('takeover'), runTakeover);
// The game's start screen links here with ?start.
if (params.has('start')) start();
