import * as tf from '@tensorflow/tfjs';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, SEA_LEVEL, SOURCE_LEVEL, cell } from '../constants';
import { fbm2, valueNoise3 } from './noise';

export interface ChunkCoord { cx: number; cz: number }

export const DEFAULT_SEED = 1337;

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
    const y = tf.range(0, H).reshape([1, H, 1, 1]);

    // Column coordinates [N, 1, Z, X].
    const wx = ox.add(lx).add(tf.zeros([1, 1, S, 1]));
    const wz = oz.add(lz).add(tf.zeros([1, 1, 1, S]));

    // Height: broad continents + rolling hills.
    const continents = fbm2(wx, wz, seed, 128, 3);
    const hills = fbm2(wx, wz, seed + 17, 32, 4);
    const height = continents.sub(0.5).mul(36).add(hills.sub(0.5).mul(14)).add(SEA_LEVEL + 2).floor();

    // Broadcast to [N, H, Z, X].
    const full = [n, H, S, S];
    const yb = y.broadcastTo(full);
    const hb = height.broadcastTo(full);
    const wx3 = wx.broadcastTo(full), wz3 = wz.broadcastTo(full);

    const ground = yb.lessEqual(hb);
    const dirt = ground.logicalAnd(yb.greater(hb.sub(3)));
    const water = yb.greater(hb).logicalAnd(yb.lessEqual(SEA_LEVEL));

    // Caves: thresholded 3D noise, kept a few blocks below the surface so seas don't drain.
    const caveNoise = valueNoise3(wx3.div(14), yb.div(9), wz3.div(14), seed + 999)
      .add(valueNoise3(wx3.div(6), yb.div(6), wz3.div(6), seed + 555).mul(0.35));
    const cave = caveNoise.greater(0.98)
      .logicalAnd(yb.less(hb.sub(4)))
      .logicalAnd(yb.greater(0));
    const lava = cave.logicalAnd(yb.lessEqual(10));

    const i = (v: number) => tf.scalar(v, 'int32');
    let out: tf.Tensor = tf.zeros(full, 'int32');
    out = tf.where(water, i(cell(Block.Water, SOURCE_LEVEL)), out);
    out = tf.where(ground, i(cell(Block.Stone)), out);
    out = tf.where(dirt, i(cell(Block.Dirt)), out);
    out = tf.where(cave, i(cell(Block.Air)), out);
    out = tf.where(lava, i(cell(Block.Lava, SOURCE_LEVEL)), out);
    out = tf.where(yb.equal(0), i(cell(Block.Stone)), out);
    return out as tf.Tensor4D;
  });
}

/** Generate a batch of chunks and read them back as per-chunk byte arrays. */
export async function generateChunks(coords: ChunkCoord[], seed = DEFAULT_SEED): Promise<Uint8Array[]> {
  if (coords.length === 0) return [];
  const t = generateChunksTensor(coords, seed);
  const flat = (await t.data()) as Int32Array;
  t.dispose();
  return coords.map((_, k) => Uint8Array.from(flat.subarray(k * CHUNK_VOLUME, (k + 1) * CHUNK_VOLUME)));
}
