import * as tf from '@tensorflow/tfjs';
import {
  Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, DEFAULT_RATES, WHEAT_RIPE, blockIndex, cellLevel, cellType,
} from './constants';
import { log, logError } from './log';
import { requestGpu } from './render/gpu';
import { checkGpuStore, type StoreCheck } from './sim/check';
import { GpuStore } from './sim/gpuStore';
import { randomSeed } from './sim/rules';
import { Simulation } from './sim/simulation';
import { ringSize } from './sim/store';
import { initTensorflow, warmUpKernels } from './tf/backend';
import { blockUpdateFused, fusedAvailable } from './tf/blockUpdateKernel';
import { generateAll } from './world/loader';
import { World, tickJobs } from './world/world';

// Measures how many block updates per second this device computes with the game's code.
// One block update = working out one cell's next state for one tick, counting the
// 16x16x64 cells of each chunk simulated.

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);
/** How long each measurement runs (?quick shortens it, for slow test machines). */
const BUDGET_MS = params.has('quick') ? 300 : 2000;
/** Chunks per tick to try (all different chunks; the game's view has 49). */
const BATCHES = [1, 4, 16, 49, 100, 200];
/** The benchmark world: 15x15 active chunks (enough for the biggest batch) plus the ghost ring. */
const ACTIVE_RADIUS = 7;
const TICKS_PER_SECOND = 5;
/** Receives the result of the write-back comparison so the JIT can't drop it as dead code. */
let compareSink = 0;

interface Setup {
  device: GPUDevice;
  backend: string;
  gpu: string;
  store: GpuStore;
  world: World;
  /** Every chunk's cells at the start, on the CPU, for the previous design's tick loop. */
  copies: Map<string, Uint8Array>;
}
let setup: Promise<Setup> | undefined;

const status = (text: string, error = false) => {
  const el = $('status');
  el.textContent = text;
  el.className = error ? 'error' : 'muted';
  if (!error) log.info(text);
};
const progress = (fraction: number) => { $('bar').style.width = `${Math.round(fraction * 100)}%`; };

/** GPU, TF.js, compiled kernels and the benchmark world: once per page. */
function prepare(): Promise<Setup> {
  setup ??= (async () => {
    log.info(`Block build ${__BUILD__}`);
    status('Starting WebGPU and TensorFlow.js…');
    const { device, info } = await requestGpu();
    const backend = await initTensorflow(device, info);
    log.info(`TensorFlow.js backend: ${backend}`);
    device.lost.then((lost) => status(`The GPU stopped working (WebGPU device lost, ${lost.reason}): ${(lost.message || 'no details').replace(/\.$/, '')}. `
      + 'Reload to try again; if WebGPU is then unavailable, fully close and reopen the browser.', true));
    status('Compiling GPU kernels…');
    await warmUpKernels();
    const store = await GpuStore.create(device, ringSize(ACTIVE_RADIUS + 1));
    status('Generating terrain…');
    const world = new World(store, ACTIVE_RADIUS, ACTIVE_RADIUS + 1);
    await generateAll(world);
    const copies = new Map<string, Uint8Array>();
    for (const c of world.chunks.values()) copies.set(c.key, Uint8Array.from(await store.readChunk(c.slot)));
    const gpu = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'unknown GPU';
    return { device, backend, gpu, store, world, copies };
  })();
  setup.catch(() => { setup = undefined; }); // allow a retry
  return setup;
}

interface Rate {
  /** Wall time per tick. */
  ms: number;
  /** Main-thread time per tick. */
  mainMs: number;
  updatesPerSecond: number;
}

/** Run `tick` repeatedly for BUDGET_MS (after one warm-up run) and work out the rate. `tick` returns its main-thread ms. */
async function measure(chunksPerTick: number, tick: () => Promise<number>): Promise<Rate> {
  await tick();
  let ticks = 0, mainMs = 0;
  const start = performance.now();
  do {
    mainMs += await tick();
    ticks++;
  } while (performance.now() - start < BUDGET_MS);
  const ms = (performance.now() - start) / ticks;
  return { ms, mainMs: mainMs / ticks, updatesPerSecond: (chunksPerTick * CHUNK_VOLUME * 1000) / ms };
}

/** The n active chunks nearest the centre. */
const nearest = (world: World, n: number) =>
  world.activeChunks().sort((a, b) => Math.hypot(a.cx, a.cz) - Math.hypot(b.cx, b.cz)).slice(0, n);

/**
 * The game's tick loop: the chunks are updated where they live in GPU memory, and the
 * tick waits for their flags (4 bytes per chunk) to come back, as the game does.
 */
function residentLoop({ store, world }: Setup, n: number): Promise<Rate> {
  const jobs = tickJobs(world, nearest(world, n));
  return measure(n, async () => {
    const t0 = performance.now();
    const flags = store.tick(jobs, randomSeed(), DEFAULT_RATES);
    const mainMs = performance.now() - t0;
    compareSink += (await flags)[0];
    return mainMs;
  });
}

/** Ticks issued back to back without reading anything back, with one wait per 4 ticks: the GPU's own speed. */
function gpuOnly({ store, world, device }: Setup, n: number, plants = true): Promise<Rate> {
  const jobs = tickJobs(world, nearest(world, n));
  return measure(n * 4, async () => {
    const t0 = performance.now();
    for (let k = 0; k < 4; k++) store.step(jobs, randomSeed(), DEFAULT_RATES, false, plants);
    const mainMs = performance.now() - t0;
    await device.queue.onSubmittedWorkDone();
    return mainMs;
  });
}

const P = CHUNK_SIZE + 2;
const PADDED_VOLUME = CHUNK_HEIGHT * P * P;

/**
 * The previous design's tick loop, for comparison: the world on the CPU, each tick
 * packing the chunks with a one-cell border from their neighbours, uploading them, one
 * step of the same rules (as a TF.js kernel), reading the results back and comparing them
 * with the chunks to find what changed (here nothing is written back).
 */
function previousLoop({ world, copies }: Setup, n: number): Promise<Rate> {
  const chunks = nearest(world, n);
  const cells = new Int32Array(n * PADDED_VOLUME);
  const data = (cx: number, cz: number) => copies.get(`${cx},${cz}`)!;
  return measure(n, async () => {
    const t0 = performance.now();
    chunks.forEach((c, i) => packChunk(data, c.cx, c.cz, cells, i * PADDED_VOLUME));
    const out = tf.tidy(() => blockUpdateFused(tf.tensor4d(cells, [n, CHUNK_HEIGHT, P, P], 'int32'), { seed: randomSeed(), plants: true, halo: 1 }));
    const t1 = performance.now();
    const next = (await out.data()) as Int32Array;
    out.dispose();
    const t2 = performance.now();
    chunks.forEach((c, k) => {
      const own = data(c.cx, c.cz);
      for (let i = 0, base = k * CHUNK_VOLUME; i < CHUNK_VOLUME; i++) if (next[base + i] !== own[i]) compareSink++;
    });
    return t1 - t0 + (performance.now() - t2);
  });
}

/** Write chunk (cx, cz) with a one-cell border from its eight neighbours into `out` at `base`, as [H][18][18]. */
function packChunk(data: (cx: number, cz: number) => Uint8Array, cx: number, cz: number, out: Int32Array, base: number): void {
  const S = CHUNK_SIZE, last = S - 1;
  const at = (dx: number, dz: number) => data(cx + dx, cz + dz);
  const self = at(0, 0), n = at(0, -1), s = at(0, 1), w = at(-1, 0), e = at(1, 0);
  const nw = at(-1, -1), ne = at(1, -1), sw = at(-1, 1), se = at(1, 1);
  for (let y = 0; y < CHUNK_HEIGHT; y++) {
    const row = (pz: number) => base + (y * P + pz) * P;
    for (let z = 0; z < S; z++) {
      const dst = row(z + 1), src = blockIndex(0, y, z);
      out.set(self.subarray(src, src + S), dst + 1);
      out[dst] = w[blockIndex(last, y, z)];
      out[dst + P - 1] = e[blockIndex(0, y, z)];
    }
    const top = row(0), bottom = row(P - 1);
    const nRow = blockIndex(0, y, last), sRow = blockIndex(0, y, 0);
    out.set(n.subarray(nRow, nRow + S), top + 1);
    out[top] = nw[blockIndex(last, y, last)];
    out[top + P - 1] = ne[blockIndex(0, y, last)];
    out.set(s.subarray(sRow, sRow + S), bottom + 1);
    out[bottom] = sw[blockIndex(last, y, 0)];
    out[bottom + P - 1] = se[blockIndex(0, y, 0)];
  }
}

const fmtRate = (v: number) =>
  v >= 1e9 ? `${(v / 1e9).toFixed(2)} billion` : v >= 1e6 ? `${(v / 1e6).toFixed(1)} million` : `${Math.round(v / 1e3)} thousand`;
const fmtMs = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2));

interface Row { n: number; resident?: Rate; gpu?: Rate; previous?: Rate }

async function runBenchmark(): Promise<void> {
  const s = await prepare();
  status('Checking the GPU world code against the reference rules…');
  const check = await checkGpuStore(s.device);
  if (!check.ok) throw new Error(`the GPU world code disagrees with the reference on this GPU (${check.detail})`);
  log.info('GPU world check passed', check.summary);
  const previous = fusedAvailable();
  const steps = BATCHES.length * (previous ? 3 : 2) + 1;
  let done = 0;
  const next = (what: string, n: number) => {
    progress(done / steps);
    status(`${what}, ${n} chunk${n === 1 ? '' : 's'} per tick (${done + 1} of ${steps})…`);
    done++;
  };
  const rows: Row[] = BATCHES.map((n) => ({ n }));
  // The previous design first: it reads the CPU copies and changes nothing.
  if (previous) for (const row of rows) { next('Previous design (pack, upload, read back)', row.n); row.previous = await previousLoop(s, row.n); }
  for (const row of rows) { next('Game tick loop, world on the GPU', row.n); row.resident = await residentLoop(s, row.n); }
  for (const row of rows) { next('GPU only', row.n); row.gpu = await gpuOnly(s, row.n); }
  const best = bestRow(rows);
  next('Fluid rules only', best.n);
  const fluids = await gpuOnly(s, best.n, false);
  progress(1);
  status(`Done. ${s.gpu}.`);
  showResult(rows, fluids, check, s.gpu);
  log.info('Benchmark result', $('result').innerText);
}

const rate = (r?: Rate) => r?.updatesPerSecond ?? 0;
const bestRow = (rows: Row[]) => rows.reduce((b, r) => (rate(r.resident) > rate(b.resident) ? r : b));

function showResult(rows: Row[], fluids: Rate, check: StoreCheck, gpu: string): void {
  const best = bestRow(rows), top = rate(best.resident);
  const chunksAt5 = Math.floor(top / (CHUNK_VOLUME * TICKS_PER_SECOND));
  const previousBest = Math.max(...rows.map((r) => rate(r.previous)));
  const gpuBest = Math.max(...rows.map((r) => rate(r.gpu)));
  const cellRate = (r?: Rate) => (r ? fmtRate(r.updatesPerSecond) : '—');
  const ms = (r?: Rate, key: 'ms' | 'mainMs' = 'ms') => (r ? `${fmtMs(r[key])} ms` : '—');
  const rowsHtml = rows.map((r) => `
    <tr${r === best ? ' class="best"' : ''}>
      <td>${r.n}</td>
      <td>${ms(r.resident)}</td>
      <td>${ms(r.resident, 'mainMs')}</td>
      <td>${cellRate(r.resident)}</td>
      <td>${cellRate(r.gpu)}</td>
      <td>${cellRate(r.previous)}</td>
      <td>${ms(r.previous, 'mainMs')}</td>
    </tr>`).join('');
  const el = $('result');
  el.hidden = false;
  el.innerHTML = `
    <div class="headline">${fmtRate(top)} <span class="unit">block updates per second</span></div>
    <p class="muted">The game's tick loop with every rule (fluids, grass, wheat): the world stays in GPU memory,
      each tick updates the chunks where they are and reads back only a 4-byte flags word per chunk.
      Best with ${best.n} chunks per tick.</p>
    <div class="stats">
      <div class="stat"><b>${chunksAt5.toLocaleString()} chunks</b>could keep updating at ${TICKS_PER_SECOND} ticks per second (the view has 49)</div>
      ${previousBest > 0 ? `<div class="stat"><b>${(top / previousBest).toFixed(1)}× faster</b>than the previous design, which moved the chunks to and from the CPU every tick (${fmtRate(previousBest)}/s at best)</div>` : ''}
      <div class="stat"><b>${fmtRate(gpuBest)}/s</b>GPU only: ticks issued back to back without waiting for their flags</div>
      <div class="stat"><b>${fmtRate(fluids.updatesPerSecond)}/s</b>GPU only with fluid rules only (no grass or wheat), ${best.n} chunks</div>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>chunks<br>per tick</th><th>tick</th><th>main<br>thread</th><th>game<br>loop</th><th>GPU<br>only</th><th>previous<br>design</th><th>previous<br>main thread</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table></div>
    <p class="muted" style="margin-top:12px">Checked first: on this GPU the world code matched the reference rules exactly
      (${check.summary}).</p>
    <p class="muted">One block update = one cell's next state for one tick, counting the 16×16×64 cells of each chunk;
      every chunk in a batch is a different one (from a 15×15-chunk world). Each figure is the average over
      ~${BUDGET_MS / 1000} s; "main thread" is the CPU time a tick takes on the page's thread. The previous design kept
      the world on the CPU: each tick it packed the chunks with a one-cell border from their neighbours, ran the same
      rules as a TF.js kernel, and read every cell back to see what changed. ${gpu}.</p>`;
}

async function runTakeover(): Promise<void> {
  const log = $('log');
  const write = (line = '') => { log.textContent += line + '\n'; };
  log.textContent = '';
  const { device } = await prepare();
  status('Generating fresh terrain for the takeover…');
  // A fresh world: the game's view around spawn. The takeover changes it.
  const store = await GpuStore.create(device, ringSize(4));
  try {
    const world = new World(store, 3, 4);
    await generateAll(world);
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
    write('(Counting the plants reads the chunks back from the GPU; the game never needs to.)');
    write();
    write('  tick  game time   batch   tick ms   main thread   growing   grass   wheat ripe');
    const sim = new Simulation(world, rates);
    const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
    const row = async (tick: number, tickMs: string, cpuMs: string) => {
      let grass = 0, wheat = 0, ripe = 0;
      for (const c of world.activeChunks()) {
        for (const v of await store.readChunk(c.slot)) {
          if (cellType(v) === Block.Grass) grass++;
          else if (cellType(v) === Block.Wheat) { wheat++; if (cellLevel(v) === WHEAT_RIPE) ripe++; }
        }
      }
      const ripeText = wheat ? `${Math.round((100 * ripe) / wheat)}% of ${wheat}` : 'none';
      write(`${String(tick).padStart(6)} ${`${(tick / TICKS_PER_SECOND).toFixed(0)} s`.padStart(10)} ${String(sim.lastBatch).padStart(7)} ${tickMs.padStart(9)} ${cpuMs.padStart(13)} ${String(sim.growingChunks).padStart(9)} ${String(grass).padStart(7)}   ${ripeText}`);
    };
    const started = performance.now();
    let ticks = 0, totalMs = 0, totalCpu = 0, chunkTicks = 0, windowMs: number[] = [], windowCpu: number[] = [];
    await row(0, '-', '-');
    while (ticks < maxTicks) {
      status(`Takeover: tick ${ticks + 1}…`);
      progress(ticks / maxTicks);
      if (!(await sim.tick())) break; // everything asleep: settled
      ticks++;
      totalMs += sim.lastTickMs; totalCpu += sim.lastCpuMs; chunkTicks += sim.lastBatch;
      windowMs.push(sim.lastTickMs); windowCpu.push(sim.lastCpuMs);
      if (ticks % 50 === 0) {
        await row(ticks, fmtMs(median(windowMs)), fmtMs(median(windowCpu)));
        windowMs = []; windowCpu = [];
      }
    }
    if (windowMs.length) await row(ticks, fmtMs(median(windowMs)), fmtMs(median(windowCpu)));
    write();
    write(`${ticks} ticks in ${((performance.now() - started) / 1000).toFixed(1)} s${sim.asleep ? ', then everything settled and went to sleep' : ''}.`);
    if (ticks) {
      write(`Average tick ${fmtMs(totalMs / ticks)} ms (main thread ${fmtMs(totalCpu / ticks)} ms), ${(chunkTicks / ticks).toFixed(1)} chunks per tick,`);
      write(`${fmtRate((chunkTicks * CHUNK_VOLUME * 1000) / totalMs)} block updates per second while ticking.`);
    }
  } finally {
    store.destroy();
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
      logError('Benchmark failed', e);
      status(`Failed: ${e instanceof Error ? e.message : String(e)}`, true);
      log.open();
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
