import * as tf from '@tensorflow/tfjs';
import {
  Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, DEFAULT_RATES, WHEAT_RIPE, cellLevel, cellType,
} from './constants';
import { requestGpu } from './render/gpu';
import { initTensorflow, warmUpKernels } from './tf/backend';
import { blockUpdateStep } from './tf/blockUpdate';
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
/** Batch sizes to try. 49 is every active chunk; 100 repeats some to see if bigger batches still help. */
const BATCHES = [1, 4, 16, 49, 100];
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

const step = (cells: tf.Tensor4D, plants: boolean) => blockUpdateStep(
  cells,
  plants ? randomField(cells.shape, [Math.random() * 999, Math.random() * 999, Math.random() * 999]) : undefined,
);

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
function gameLoop(world: World, n: number, plants: boolean): Promise<Rate> {
  const chunks = batchOf(world, n);
  const cells = new Int32Array(n * PADDED_VOLUME);
  return measure(n, async () => {
    const t0 = performance.now();
    chunks.forEach((c, i) => packChunk(world, c, cells, i * PADDED_VOLUME));
    const out = tf.tidy(() => step(tf.tensor4d(cells, [n, CHUNK_HEIGHT, PADDED, PADDED], 'int32'), plants)
      .slice([0, 0, HALO, HALO], [n, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_SIZE]));
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
async function gpuOnly(world: World, n: number, plants: boolean): Promise<Rate> {
  const cells = new Int32Array(n * PADDED_VOLUME);
  batchOf(world, n).forEach((c, i) => packChunk(world, c, cells, i * PADDED_VOLUME));
  let state = tf.tensor4d(cells, [n, CHUNK_HEIGHT, PADDED, PADDED], 'int32');
  try {
    return await measure(n * 4, async () => {
      const t0 = performance.now();
      for (let k = 0; k < 4; k++) {
        const next = tf.tidy(() => step(state, plants));
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
  const steps = BATCHES.length * 2 + 1;
  let done = 0;
  const next = (what: string) => {
    progress(done / steps);
    status(`${what} (${done + 1} of ${steps})…`);
    done++;
  };
  const game: Rate[] = [], gpuRates: Rate[] = [];
  for (const n of BATCHES) {
    next(`Game tick loop, ${n} chunk${n === 1 ? '' : 's'}`);
    game.push(await gameLoop(world, n, true));
  }
  for (const n of BATCHES) {
    next(`GPU only, ${n} chunk${n === 1 ? '' : 's'}`);
    gpuRates.push(await gpuOnly(world, n, true));
  }
  const bestIndex = game.reduce((b, r, i) => (r.updatesPerSecond > game[b].updatesPerSecond ? i : b), 0);
  next(`Fluid rules only, ${BATCHES[bestIndex]} chunks`);
  const fluids = await gameLoop(world, BATCHES[bestIndex], false);
  progress(1);
  status(`Done. ${backend} on ${gpu}.`);
  showResult(game, gpuRates, bestIndex, fluids, backend, gpu);
}

function showResult(game: Rate[], gpuRates: Rate[], best: number, fluids: Rate, backend: string, gpu: string): void {
  const top = game[best], gpuTop = Math.max(...gpuRates.map((r) => r.updatesPerSecond));
  const chunksAt5 = Math.floor(top.updatesPerSecond / (CHUNK_VOLUME * TICKS_PER_SECOND));
  const rows = BATCHES.map((n, i) => `
    <tr${i === best ? ' class="best"' : ''}>
      <td>${n}</td>
      <td>${fmtMs(game[i].ms)} ms</td>
      <td>${fmtMs(game[i].mainMs)} ms</td>
      <td>${fmtRate(game[i].updatesPerSecond)}</td>
      <td>${fmtRate(gpuRates[i].updatesPerSecond)}</td>
    </tr>`).join('');
  const el = $('result');
  el.hidden = false;
  el.innerHTML = `
    <div class="headline">${fmtRate(top.updatesPerSecond)} <span class="unit">block updates per second</span></div>
    <p class="muted">The game's tick loop with every rule (fluids, grass, wheat): packing chunks with their ghost
      borders, the TF.js step on the GPU, reading results back. Best with ${BATCHES[best]} chunks per tick.</p>
    <div class="stats">
      <div class="stat"><b>${chunksAt5.toLocaleString()} chunks</b>could keep updating at ${TICKS_PER_SECOND} ticks per second (the view has 49)</div>
      <div class="stat"><b>${fmtRate(gpuTop)}/s</b>GPU only: steps chained with the state kept on the GPU</div>
      <div class="stat"><b>${fmtRate(fluids.updatesPerSecond)}/s</b>fluid rules only (no grass or wheat), ${BATCHES[best]} chunks</div>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>chunks per tick</th><th>tick</th><th>main thread</th><th>updates/s (game loop)</th><th>updates/s (GPU only)</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="muted" style="margin-top:12px">One block update = one cell's next state for one tick, counting the
      16×16×64 cells of each chunk (not its ghost border). Each figure is the average over ~${BUDGET_MS / 1000} s.
      Batches over 49 chunks repeat chunks. ${backend} on ${gpu}.</p>`;
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
