import * as tf from '@tensorflow/tfjs';
import {
  Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, GRAVEL, SAND, SEA_LEVEL, SOURCE_LEVEL, WHEAT_RIPE, cell,
} from '../constants';
import { fbm2, hash12, valueNoise3 } from './noise';

// The world's settings, shared with the plain-JS generator (world/jsWorldgen.ts) without TF.js.
export {
  DEFAULT_SEED, DIAMOND_MAX_Y, DIAMOND_THRESHOLD, SAND_ABOVE, SAND_BELOW, WHEAT_PATCH_CHANCE, WILD_WHEAT, type ChunkCoord,
} from '../world/worldgenParams';
import { DEFAULT_SEED, DIAMOND_MAX_Y, DIAMOND_THRESHOLD, SAND_ABOVE, SAND_BELOW, WHEAT_PATCH_CHANCE, WILD_WHEAT, type ChunkCoord } from '../world/worldgenParams';

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
 *   - terrain height from 2D fBm; dirt on the top 3 blocks (sand on beaches and in shallow
 *     water, gravel on the deeper sea floor), stone below
 *   - water sources fill everything between the terrain and SEA_LEVEL
 *   - 3D-noise caves carved under the surface; deep cave cells (y <= 10) become lava lakes
 *   - diamond ore in small 3D-noise blobs through the deep stone (y <= DIAMOND_MAX_Y)
 *   - grass on top of dry land but beaches, and (if WILD_WHEAT) a few patches of ripe wild wheat
 *   (Generated terrain is settled: nothing in it flows, spreads or grows, so freshly
 *   loaded chunks go to sleep after one block-update tick. Water lies on the surface and
 *   lava only in deep caves sealed under it, so the two never meet.)
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

    const height = terrainHeight(wx, wz, seed);

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
    // Diamond ore: small blobs through the deep stone.
    const ore = valueNoise3(wx.div(3), y.div(3), wz.div(3), seed + 333).greater(DIAMOND_THRESHOLD)
      .logicalAnd(y.lessEqual(DIAMOND_MAX_Y));

    const features = surfaceFeatures(wx, wz, height, seed);
    const grass = y.equal(height).logicalAnd(features.grass);
    const wheat = y.equal(height.add(1)).logicalAnd(features.wheat).logicalAnd(WILD_WHEAT);

    // Combine mutually exclusive masks: y = 0 is bedrock stone; caves are air (lava when deep).
    const bedrock = y.equal(0), aboveBedrock = y.greater(0);
    const solid = ground.logicalAnd(cave.logicalNot());
    const stone = solid.logicalAnd(dirt.logicalNot()).logicalOr(bedrock);
    const diamond = stone.logicalAnd(ore).logicalAnd(aboveBedrock);
    const term = (mask: tf.Tensor, value: number) => mask.cast('int32').mul(tf.scalar(value, 'int32'));
    // (One term for the three, per column: addN takes at most 7 inputs on WebGPU's storage-buffer limit.)
    const topLayer = tf.addN([term(features.bare, cell(Block.Dirt)), term(features.sand, SAND), term(features.gravel, GRAVEL)]);
    return tf.addN([
      term(stone.logicalAnd(diamond.logicalNot()), cell(Block.Stone)),
      term(diamond, cell(Block.Diamond)),
      // The top 3 blocks (but grass): dirt, sand or gravel by column.
      solid.logicalAnd(dirt).logicalAnd(aboveBedrock).logicalAnd(grass.logicalNot()).cast('int32').mul(topLayer),
      term(grass, cell(Block.Grass)),
      term(water.logicalAnd(aboveBedrock), cell(Block.Water, SOURCE_LEVEL)),
      term(lava, cell(Block.Lava, SOURCE_LEVEL)),
      term(wheat, cell(Block.Wheat, WHEAT_RIPE)),
    ]) as tf.Tensor4D;
  });
}

/**
 * Per-column decorations on dry land (columns whose ground is at or above sea level,
 * so no water on top), from [N, 1, Z, X] column coordinates and ground height:
 * - grass: the ground block is grass (all dry land but beaches, so there's no bare dirt for it to spread onto)
 * - wheat: ripe wild wheat stands on the ground (grass)
 * - sand, gravel: the top 3 blocks are sand (beaches, shallow water) or gravel (deeper sea floor),
 *   and bare: neither (dirt, under grass or water)
 */
export function surfaceFeatures(wx: tf.Tensor, wz: tf.Tensor, height: tf.Tensor, seed: number) {
  // Two independent rolls per decision: one hash can't resolve such small chances in float32.
  const roll = (x: tf.Tensor, z: tf.Tensor, chance: number, s1: number, s2: number) =>
    hash12(x, z, s1).less(Math.sqrt(chance)).logicalAnd(hash12(x, z, s2).less(Math.sqrt(chance)));
  const dry = height.greaterEqual(SEA_LEVEL);
  const sand = height.greaterEqual(SEA_LEVEL - SAND_BELOW).logicalAnd(height.lessEqual(SEA_LEVEL + SAND_ABOVE));
  const gravel = height.less(SEA_LEVEL - SAND_BELOW);
  const bare = sand.logicalOr(gravel).logicalNot();
  const grass = dry.logicalAnd(sand.logicalNot());
  // Wild wheat: patches picked per 4x4 area, about 60% of a patch's columns planted.
  const ax = wx.div(4).floor(), az = wz.div(4).floor();
  const wheat = roll(ax, az, WHEAT_PATCH_CHANCE, (seed * 3) % 983, (seed * 11) % 977 + 0.25)
    .logicalAnd(hash12(wx, wz, (seed * 13) % 971 + 0.75).less(0.6))
    .logicalAnd(grass);
  return { grass, wheat, sand, gravel, bare };
}

/**
 * The height of the ground (its top block's y) at world columns (wx, wz): broad continents
 * plus rolling hills. Float tensors of any (equal or broadcastable) shapes. The far terrain
 * (world/farTerrain.ts) samples it too, so it lines up with generated chunks.
 */
export function terrainHeight(wx: tf.Tensor, wz: tf.Tensor, seed = DEFAULT_SEED): tf.Tensor {
  return tf.tidy(() => {
    const continents = fbm2(wx, wz, seed, 128, 3);
    const hills = fbm2(wx, wz, seed + 17, 32, 4);
    return continents.sub(0.5).mul(36).add(hills.sub(0.5).mul(14)).add(SEA_LEVEL + 2).floor();
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
