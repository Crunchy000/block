import * as tf from '@tensorflow/tfjs';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, SEA_LEVEL, SOURCE_LEVEL, cell } from '../constants';
import { fbm2, valueNoise3 } from './noise';

export interface ChunkCoord { cx: number; cz: number }

export const DEFAULT_SEED = 1337;

/**
 * Chunks per generation batch. Batches are always padded to this size so every
 * batch has the same shapes and runs the same (already compiled) GPU kernels.
 */
export const GEN_BATCH = 16;

/**
 * Generate a batch of chunks in one TF graph.
 * Returns the cell tensor of shape [N, H, Z, X] (int32). Caller disposes.
 *
 * Layers:
 *   - terrain height from 2D fBm; dirt on the top 3 blocks, stone below
 *   - water sources fill everything between the terrain and SEA_LEVEL
 *   - 3D-noise caves carved under the surface; deep cave cells (y <= 10) become lava lakes
 *   - y = 0 is always stone
 */
export function generateChunksTensor(coords: ChunkCoord[], seed = DEFAULT_SEED): tf.Tensor4D {
  return tf.tidy(() => {
    const n = coords.length, S = CHUNK_SIZE, H = CHUNK_HEIGHT;
    const ox = tf.tensor1d(coords.map((c) => c.cx * S)).reshape([n, 1, 1, 1]);
    const oz = tf.tensor1d(coords.map((c) => c.cz * S)).reshape([n, 1, 1, 1]);
    const lx = tf.range(0, S).reshape([1, 1, 1, S]);
    const lz = tf.range(0, S).reshape([1, 1, S, 1]);

    // Everything below is built from [N, 1, Z, X] column tensors and a [1, H, 1, 1]
    // height axis that broadcast inside each op. Never broadcastTo/where-with-a-scalar
    // here: TF.js expands small tensors to full size on the CPU, on the main thread.
    const wx = ox.add(lx).add(tf.zeros([1, 1, S, 1]));
    const wz = oz.add(lz).add(tf.zeros([1, 1, 1, S]));
    const y = tf.range(0, H).reshape([1, H, 1, 1]);

    // Height: broad continents + rolling hills.
    const continents = fbm2(wx, wz, seed, 128, 3);
    const hills = fbm2(wx, wz, seed + 17, 32, 4);
    const height = continents.sub(0.5).mul(36).add(hills.sub(0.5).mul(14)).add(SEA_LEVEL + 2).floor();

    const ground = y.lessEqual(height);
    const dirt = ground.logicalAnd(y.greater(height.sub(3)));
    const water = y.greater(height).logicalAnd(y.lessEqual(SEA_LEVEL));

    // Caves: thresholded 3D noise, kept a few blocks below the surface so seas don't drain.
    const caveNoise = valueNoise3(wx.div(14), y.div(9), wz.div(14), seed + 999)
      .add(valueNoise3(wx.div(6), y.div(6), wz.div(6), seed + 555).mul(0.35));
    const cave = caveNoise.greater(0.98)
      .logicalAnd(y.less(height.sub(4)))
      .logicalAnd(y.greater(0));
    const lava = cave.logicalAnd(y.lessEqual(10));

    // Combine mutually exclusive masks: y = 0 is bedrock stone; caves are air (lava when deep).
    const bedrock = y.equal(0), aboveBedrock = y.greater(0);
    const solid = ground.logicalAnd(cave.logicalNot());
    const stone = solid.logicalAnd(dirt.logicalNot()).logicalOr(bedrock);
    const term = (mask: tf.Tensor, value: number) => mask.cast('int32').mul(tf.scalar(value, 'int32'));
    return tf.addN([
      term(stone, cell(Block.Stone)),
      term(solid.logicalAnd(dirt).logicalAnd(aboveBedrock), cell(Block.Dirt)),
      term(water.logicalAnd(aboveBedrock), cell(Block.Water, SOURCE_LEVEL)),
      term(lava, cell(Block.Lava, SOURCE_LEVEL)),
    ]) as tf.Tensor4D;
  });
}

/** Generate up to GEN_BATCH chunks and read them back as per-chunk byte arrays. */
export async function generateChunks(coords: ChunkCoord[], seed = DEFAULT_SEED): Promise<Uint8Array[]> {
  if (coords.length === 0) return [];
  if (coords.length > GEN_BATCH) throw new Error(`generateChunks: at most ${GEN_BATCH} chunks per batch`);
  const padded = [...coords];
  while (padded.length < GEN_BATCH) padded.push(coords[0]);
  const t = generateChunksTensor(padded, seed);
  try {
    const flat = (await t.data()) as Int32Array;
    return coords.map((_, k) => Uint8Array.from(flat.subarray(k * CHUNK_VOLUME, (k + 1) * CHUNK_VOLUME)));
  } finally {
    t.dispose();
  }
}
