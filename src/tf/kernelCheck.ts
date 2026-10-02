import * as tf from '@tensorflow/tfjs';
import { Block, PRIMED_DIRT, SOURCE_LEVEL, cell } from '../constants';
import { blockUpdateFused } from './blockUpdateKernel';
import { blockUpdateReference, cellRandom } from './blockUpdateReference';

export interface KernelCheck {
  ok: boolean;
  /** Cells compared. */
  cells: number;
  mismatches: number;
  /** The first mismatch, if any. */
  detail: string;
}

/**
 * Run the fused GPU kernel and the plain-JS reference on the same random cells and the
 * same random numbers, for several chained ticks, and compare every cell. Covers the plant
 * rules on and off, and output with and without the ghost border.
 */
export async function checkFusedKernel(seed = 1234, ticks = 6): Promise<KernelCheck> {
  const rand = mulberry32(seed);
  const N = 3, H = 10, D = 12, W = 14, size = H * D * W;
  const rates = { grassSpread: 0.25, wheatGrow: 0.125, wheatGrowWet: 0.5 };
  let states = Array.from({ length: N }, () => randomCells(size, rand));
  let cells = 0, mismatches = 0, detail = '';
  for (let tick = 0; tick < ticks; tick++) {
    const plants = tick % 3 !== 2;
    const halo = tick % 2;
    const tickSeed = (rand() * 2 ** 32) >>> 0;
    const expected = states.map((s, k) => blockUpdateReference(s, H, D, W,
      plants ? Float32Array.from({ length: size }, (_, i) => cellRandom(k * size + i, tickSeed)) : undefined, rates));
    const input = tf.tensor4d(Int32Array.from(states.flatMap((s) => [...s])), [N, H, D, W], 'int32');
    const output = blockUpdateFused(input, { seed: tickSeed, plants, rates, halo });
    const got = (await output.data()) as Int32Array;
    tf.dispose([input, output]);
    const oD = D - 2 * halo, oW = W - 2 * halo;
    for (let k = 0; k < N; k++) {
      for (let y = 0; y < H; y++) {
        for (let z = 0; z < oD; z++) {
          for (let x = 0; x < oW; x++) {
            const g = got[((k * H + y) * oD + z) * oW + x];
            const before = states[k][(y * D + z + halo) * W + x + halo];
            const e = expected[k][(y * D + z + halo) * W + x + halo];
            cells++;
            if (g === e) continue;
            mismatches++;
            detail ||= `tick ${tick} (plants ${plants}, halo ${halo}) item ${k} at y${y} z${z + halo} x${x + halo}: ` +
              `cell ${before} became ${g}, expected ${e}`;
          }
        }
      }
    }
    states = expected;
  }
  return { ok: mismatches === 0, cells, mismatches, detail };
}

/** Random cells of every kind, including primed dirt and wheat at every stage. */
export function randomCells(size: number, rand: () => number): Int32Array {
  return Int32Array.from({ length: size }, () => {
    const r = rand(), level = 1 + Math.floor(rand() * SOURCE_LEVEL);
    return r < 0.45 ? cell(Block.Air) : r < 0.53 ? cell(Block.Stone) : r < 0.55 ? cell(Block.Diamond) : r < 0.67 ? cell(Block.Dirt)
      : r < 0.7 ? PRIMED_DIRT : r < 0.76 ? cell(Block.Grass) : r < 0.82 ? cell(Block.Wheat, level - 1)
        : r < 0.91 ? cell(Block.Water, level) : cell(Block.Lava, level);
  });
}

/** Small deterministic PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
