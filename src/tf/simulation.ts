import * as tf from '@tensorflow/tfjs';
import {
  Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, DEFAULT_RATES, PRIMED_DIRT, blockIndex, isGrowing, isPlant,
  type PlantRates,
} from '../constants';
import type { Chunk } from '../world/chunk';
import { Border, type World } from '../world/world';
import { blockUpdateStep } from './blockUpdate';
import { blockUpdateFused, fusedAvailable, randomSeed } from './blockUpdateKernel';
import { randomField } from './random';

/** Ghost cells around each simulated chunk: one tick only reads direct neighbours (diagonals included). */
export const HALO = 1;
/** Width of a chunk plus its ghost border. */
export const PADDED = CHUNK_SIZE + 2 * HALO;
const PADDED_VOLUME = CHUNK_HEIGHT * PADDED * PADDED;

/**
 * Runs block updates for the active area.
 *
 * Only chunks that can change are simulated: the world keeps an "awake" set
 * (chunks that were edited, freshly loaded, next to a change, or with plants
 * still growing). Each tick packs the awake chunks into one batch
 * [N, H, 18, 18]: every chunk with a one-cell ghost border copied from its
 * eight neighbours (active or ghost chunks). The rules only look one cell
 * sideways, so that border is all a chunk needs; the step runs on the whole
 * batch at once, and only the 16x16 interiors are read back and written.
 * On WebGPU the step is the fused kernel (one GPU pass); elsewhere the tensor-op rules.
 * Cost scales with the number of awake chunks, wherever they are.
 * When nothing is awake the tick is skipped entirely.
 */
export class Simulation {
  ticks = 0;
  /** Wall time of the last tick, including waiting for the GPU. */
  lastTickMs = 0;
  /** Main-thread time of the last tick (packing, dispatch, unpacking). */
  lastCpuMs = 0;
  lastChangedChunks = 0;
  /** Chunks simulated in the last tick. */
  lastBatch = 0;
  /** Whether the last tick ran the plant rules (grass, wheat). */
  lastPlants = false;
  /** Chunks where plants could still change by chance after the last tick. */
  growingChunks = 0;
  asleep = false;
  /** Use the fused WebGPU kernel when the backend has it (the startup check can turn it off). */
  useFused = true;
  /** Whether the last tick ran on the fused kernel. */
  lastFused = false;
  private buffer = new Int32Array(0);

  constructor(private readonly world: World, public rates: PlantRates = DEFAULT_RATES) {}

  /** Returns true if a tick ran (false when not ready, busy, or nothing is awake). */
  async tick(): Promise<boolean> {
    const world = this.world;
    if (world.locked || !world.haloReady()) return false;
    const awake = world.takeAwake();
    this.asleep = awake.length === 0;
    if (this.asleep) return false;

    const t0 = performance.now();
    world.locked = true;
    let cpuMs = 0;
    try {
      const N = awake.length, H = CHUNK_HEIGHT, S = CHUNK_SIZE, P = PADDED;
      if (this.buffer.length < N * PADDED_VOLUME) this.buffer = new Int32Array(N * PADDED_VOLUME);
      const cells = this.buffer.subarray(0, N * PADDED_VOLUME);
      awake.forEach((chunk, n) => packChunk(world, chunk, cells, n * PADDED_VOLUME));
      // The plant rules only run when plants are in a batch chunk or its border.
      const plants = awake.some((c) => aroundChunk(world, c).some((n) => n.plants));
      this.lastBatch = N;
      this.lastPlants = plants;

      const fused = this.useFused && fusedAvailable();
      this.lastFused = fused;
      const interior = tf.tidy(() => {
        const input = tf.tensor4d(cells, [N, H, P, P], 'int32');
        if (fused) return blockUpdateFused(input, { seed: randomSeed(), plants, rates: this.rates, halo: HALO });
        const seeds = [Math.random() * 1024, Math.random() * 1024, Math.random() * 1024] as const;
        const random = plants ? randomField([N, H, P, P], seeds) : undefined;
        return blockUpdateStep(input, random, this.rates).slice([0, 0, HALO, HALO], [N, H, S, S]);
      });
      cpuMs += performance.now() - t0;
      let next: Int32Array;
      try {
        next = (await interior.data()) as Int32Array;
      } finally {
        interior.dispose();
      }

      const t1 = performance.now();
      this.lastChangedChunks = this.writeBack(next, awake);
      cpuMs += performance.now() - t1;
      this.ticks++;
      return true;
    } catch (e) {
      world.wake(awake); // retry these next time (e.g. after a backend fallback)
      throw e;
    } finally {
      world.locked = false;
      world.flushPendingEdits();
      this.lastTickMs = performance.now() - t0;
      this.lastCpuMs = cpuMs;
    }
  }

  /**
   * Copy each chunk's stepped interior back. The interior layout [H][16][16] is the
   * chunk's own layout, so chunk n's cells start at n * CHUNK_VOLUME.
   * Returns how many chunks changed in a way that matters (anything but priming).
   */
  private writeBack(next: Int32Array, chunks: Chunk[]): number {
    let changed = 0, growingChunks = 0;
    chunks.forEach((chunk, n) => {
      const data = chunk.data, base = n * CHUNK_VOLUME;
      let relevant = false, flags = 0, borders = Border.None;
      for (let i = 0; i < CHUNK_VOLUME; i++) {
        const v = next[base + i], old = data[i];
        flags |= PLANT_FLAGS[v];
        if (v === old) continue;
        data[i] = v;
        // Priming is bookkeeping: it changes neither how the block looks nor how neighbours behave.
        if (unprimed(v) === unprimed(old)) continue;
        relevant = true;
        const x = i & 15, z = (i >> 4) & 15;
        if (x === 0) borders |= Border.West;
        else if (x === 15) borders |= Border.East;
        if (z === 0) borders |= Border.North;
        else if (z === 15) borders |= Border.South;
        if ((x === 0 || x === 15) && (z === 0 || z === 15)) {
          borders |= x === 0 ? (z === 0 ? Border.NorthWest : Border.SouthWest) : (z === 0 ? Border.NorthEast : Border.SouthEast);
        }
      }
      chunk.plants = (flags & PLANT) !== 0;
      if (relevant) {
        changed++;
        this.world.markChanged(chunk, borders);
      }
      // Primed dirt and unripe wheat change by chance, so this chunk isn't settled even if
      // nothing changed this tick.
      if (flags & GROWING) {
        growingChunks++;
        this.world.wake([chunk]);
      }
    });
    this.growingChunks = growingChunks;
    return changed;
  }
}

const unprimed = (c: number) => (c === PRIMED_DIRT ? Block.Dirt : c);

// Per-cell-value flags, so the write-back loop does one table lookup per cell.
const PLANT = 1, GROWING = 2;
const PLANT_FLAGS = Uint8Array.from({ length: 256 }, (_, c) => (isPlant(c) ? PLANT : 0) | (isGrowing(c) ? GROWING : 0));

/** The chunk and its eight neighbours (all loaded while the halo is ready). */
function aroundChunk(world: World, chunk: Chunk): Chunk[] {
  const out: Chunk[] = [];
  for (let dz = -1; dz <= 1; dz++)
    for (let dx = -1; dx <= 1; dx++) out.push(world.getChunk(chunk.cx + dx, chunk.cz + dz)!);
  return out;
}

/**
 * Write `chunk` with a one-cell ghost border into `out` at `base`, as [H][18][18]:
 * the interior from the chunk itself, edges and corners from its neighbours.
 */
export function packChunk(world: World, chunk: Chunk, out: Int32Array, base: number): void {
  const S = CHUNK_SIZE, P = PADDED, last = S - 1;
  const at = (dx: number, dz: number) => world.getChunk(chunk.cx + dx, chunk.cz + dz)!.data;
  const self = chunk.data, n = at(0, -1), s = at(0, 1), w = at(-1, 0), e = at(1, 0);
  const nw = at(-1, -1), ne = at(1, -1), sw = at(-1, 1), se = at(1, 1);
  for (let y = 0; y < CHUNK_HEIGHT; y++) {
    const row = (pz: number) => base + (y * P + pz) * P; // padded row start
    for (let z = 0; z < S; z++) {
      const dst = row(z + HALO), src = blockIndex(0, y, z);
      out.set(self.subarray(src, src + S), dst + HALO);
      out[dst] = w[blockIndex(last, y, z)];
      out[dst + P - 1] = e[blockIndex(0, y, z)];
    }
    const top = row(0), bottom = row(P - 1);
    const nRow = blockIndex(0, y, last), sRow = blockIndex(0, y, 0);
    out.set(n.subarray(nRow, nRow + S), top + HALO);
    out[top] = nw[blockIndex(last, y, last)];
    out[top + P - 1] = ne[blockIndex(0, y, last)];
    out.set(s.subarray(sRow, sRow + S), bottom + HALO);
    out[bottom] = sw[blockIndex(last, y, 0)];
    out[bottom + P - 1] = se[blockIndex(0, y, 0)];
  }
}
