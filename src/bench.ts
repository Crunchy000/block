import * as tf from '@tensorflow/tfjs';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, DEFAULT_RATES, WHEAT_RIPE, cellLevel, cellType } from './constants';
import { requestGpu } from './render/gpu';
import { initTensorflow, warmUpKernels } from './tf/backend';
import { blockUpdateStep } from './tf/blockUpdate';
import { randomField } from './tf/random';
import { HALO, PADDED, Simulation, packChunk } from './tf/simulation';
import { GEN_BATCH, generateChunks } from './tf/worldgen';
import { World } from './world/world';

// Benchmarks block updates on this machine through the same code paths as the game.

const out = document.getElementById('out')!;
const log = (line = '') => { out.textContent += line + '\n'; };
const params = new URLSearchParams(location.search);
const chance = (name: string, fallback: number) =>
  params.has(name) ? Math.min(1, Math.max(0, Number(params.get(name)))) : fallback;
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const ms = (v: number, width = 9) => `${v.toFixed(1)} ms`.padStart(width);

async function main(): Promise<void> {
  const { device, info } = await requestGpu();
  const backend = await initTensorflow(device, info);
  log(`TF.js backend: ${backend}   GPU: ${[info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'unknown'}`);
  let t = performance.now();
  await warmUpKernels();
  log(`Kernel warm-up (compile): ${ms(performance.now() - t, 0)}`);

  t = performance.now();
  const world = await loadWorld();
  log(`World generation: ${world.chunks.size} chunks in ${ms(performance.now() - t, 0)}`);
  log();
  await batchCosts(world);
  log();
  await takeover(world);
  log();
  log('Done.');
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

async function batchCosts(world: World): Promise<void> {
  log('1. One tick by batch size: GPU work plus reading the result back (median of 10).');
  log('   Each chunk is 16x16x64 cells plus a one-cell ghost border.');
  log('   "dispatch" is the main-thread time to issue the TF.js ops.');
  log();
  log('   chunks      cells   fluids only  + plant rules   per chunk   dispatch');
  const chunks = world.activeChunks();
  const volume = CHUNK_HEIGHT * PADDED * PADDED;
  for (const n of [1, 4, 9, 16, 25, 49]) {
    const cells = new Int32Array(n * volume);
    chunks.slice(0, n).forEach((c, i) => packChunk(world, c, cells, i * volume));
    const shape: [number, number, number, number] = [n, CHUNK_HEIGHT, PADDED, PADDED];
    const step = (plants: boolean) => tf.tidy(() => {
      const input = tf.tensor4d(cells, shape, 'int32');
      const random = plants ? randomField(shape, [Math.random() * 999, 17, 31]) : undefined;
      return blockUpdateStep(input, random).slice([0, 0, HALO, HALO], [n, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_SIZE]);
    });
    const measure = async (plants: boolean) => {
      const total: number[] = [], dispatch: number[] = [];
      for (let i = 0; i < 12; i++) {
        const t0 = performance.now();
        const result = step(plants);
        const t1 = performance.now();
        await result.data();
        result.dispose();
        if (i >= 2) { // the first runs settle buffer pools
          total.push(performance.now() - t0);
          dispatch.push(t1 - t0);
        }
      }
      return { total: median(total), dispatch: median(dispatch) };
    };
    const fluids = await measure(false), plants = await measure(true);
    log(`   ${String(n).padStart(6)} ${String(n * volume).padStart(10)}  ${ms(fluids.total, 12)}  ${ms(plants.total, 13)}  ${ms(plants.total / n, 10)}  ${ms(plants.dispatch, 9)}`);
  }

  // The rest of a tick's main-thread work: copying chunks (and borders) in, comparing results out.
  const cells = new Int32Array(chunks.length * volume);
  const t0 = performance.now();
  for (let r = 0; r < 5; r++) chunks.forEach((c, i) => packChunk(world, c, cells, i * volume));
  log();
  log(`   Packing chunks with their ghost borders: ${((performance.now() - t0) / (5 * chunks.length)).toFixed(3)} ms per chunk`);
}

async function takeover(world: World): Promise<void> {
  const rates = {
    grassSpread: chance('spread', DEFAULT_RATES.grassSpread),
    wheatGrow: chance('grow', DEFAULT_RATES.wheatGrow),
    wheatGrowWet: chance('grow', DEFAULT_RATES.wheatGrowWet),
  };
  const maxTicks = Number(params.get('ticks') ?? 600);
  log(`2. Grass spreading and wheat growing around spawn, ticks back to back (the game runs 5 per second).`);
  log(`   Chances per tick: grass ${rates.grassSpread.toFixed(4)}, wheat ${rates.wheatGrow.toFixed(4)} (wet ${rates.wheatGrowWet.toFixed(4)}). Up to ${maxTicks} ticks.`);
  log(`   Only awake chunks are simulated: ones with grass still able to spread, unripe wheat, or a change next door.`);
  log();
  log('    tick  game time   batch   tick ms   main thread   growing   grass   wheat ripe');
  const sim = new Simulation(world, rates);
  const census = () => {
    let grass = 0, wheat = 0, ripe = 0;
    for (const c of world.activeChunks()) {
      for (const v of c.data) {
        if (cellType(v) === Block.Grass) grass++;
        else if (cellType(v) === Block.Wheat) { wheat++; if (cellLevel(v) === WHEAT_RIPE) ripe++; }
      }
    }
    return { grass, ripe: wheat ? `${Math.round((100 * ripe) / wheat)}% of ${wheat}` : 'none' };
  };
  const row = (tick: number, tickMs: string, cpuMs: string) => {
    const c = census();
    log(`   ${String(tick).padStart(5)} ${`${(tick / 5).toFixed(0)} s`.padStart(10)} ${String(sim.lastBatch).padStart(7)} ${tickMs.padStart(9)} ${cpuMs.padStart(13)} ${String(sim.growingChunks).padStart(9)} ${String(c.grass).padStart(7)}   ${c.ripe}`);
  };
  const started = performance.now();
  let ticks = 0, totalMs = 0, totalCpu = 0, chunkTicks = 0, peak = 0, windowMs: number[] = [], windowCpu: number[] = [];
  row(0, '-', '-');
  while (ticks < maxTicks) {
    if (!(await sim.tick())) break; // everything asleep: settled
    ticks++;
    totalMs += sim.lastTickMs; totalCpu += sim.lastCpuMs; chunkTicks += sim.lastBatch;
    peak = Math.max(peak, sim.lastBatch);
    windowMs.push(sim.lastTickMs); windowCpu.push(sim.lastCpuMs);
    if (ticks % 50 === 0) {
      row(ticks, `${median(windowMs).toFixed(1)}`, `${median(windowCpu).toFixed(1)}`);
      windowMs = []; windowCpu = [];
    }
  }
  if (windowMs.length) row(ticks, `${median(windowMs).toFixed(1)}`, `${median(windowCpu).toFixed(1)}`);
  log();
  log(`   ${ticks} ticks in ${ms(performance.now() - started, 0)} wall time${sim.asleep ? ', then everything settled and went to sleep' : ''}.`);
  if (ticks) {
    log(`   Average tick: ${ms(totalMs / ticks, 0)} (main thread ${ms(totalCpu / ticks, 0)}), batch ${(chunkTicks / ticks).toFixed(1)} chunks (peak ${peak}).`);
    log(`   Cost per chunk per tick: ${ms(totalMs / chunkTicks, 0)}. At 5 ticks/s the budget is 200 ms per tick.`);
  }
}

main().catch((e) => {
  console.error(e);
  log(`Failed: ${e instanceof Error ? e.message : String(e)}`);
});
