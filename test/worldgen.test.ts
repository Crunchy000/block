import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, SEA_LEVEL, blockIndex, cellType } from '../src/constants';
import {
  GEN_BATCH, GRASS_SEED_CHANCE, WHEAT_PATCH_CHANCE, generateChunks, surfaceFeatures, type ChunkCoord,
} from '../src/tf/worldgen';

beforeAll(async () => {
  await tf.setBackend('cpu');
});

describe('generateChunks', () => {
  it('produces the materials, with plants placed on the ground', async () => {
    const coords: ChunkCoord[] = [];
    for (let cz = -3; cz <= 3; cz++) for (let cx = -3; cx <= 3; cx++) coords.push({ cx, cz });
    const chunks: Uint8Array[] = [];
    for (let i = 0; i < coords.length; i += GEN_BATCH) chunks.push(...await generateChunks(coords.slice(i, i + GEN_BATCH)));
    expect(chunks).toHaveLength(coords.length);
    const counts = new Array(7).fill(0);
    for (const c of chunks) {
      expect(c.length).toBe(CHUNK_VOLUME);
      for (const v of c) counts[cellType(v)]++;
    }
    console.log('block counts [air, stone, dirt, water, lava, grass, wheat]', counts);
    for (const b of [Block.Air, Block.Stone, Block.Dirt, Block.Water, Block.Lava]) expect(counts[b]).toBeGreaterThan(0);
    // Wild wheat stands on dirt or grass.
    chunks.forEach((c) => c.forEach((v, i) => {
      if (cellType(v) === Block.Wheat) expect([Block.Dirt, Block.Grass]).toContain(cellType(c[i - CHUNK_SIZE * CHUNK_SIZE]));
    }));
    // Grass is just a few seeds, each on top of the ground with air above it.
    expect(counts[Block.Grass]).toBeLessThan(40);
    chunks.forEach((c) => c.forEach((v, i) => {
      if (cellType(v) !== Block.Grass) return;
      expect(cellType(c[i + CHUNK_SIZE * CHUNK_SIZE])).toBe(Block.Air);
      expect(Math.floor(i / (CHUNK_SIZE * CHUNK_SIZE))).toBeGreaterThanOrEqual(SEA_LEVEL);
    }));
  });

  it('is deterministic and seamless across chunk borders', async () => {
    const [a, b] = await generateChunks([{ cx: 0, cz: 0 }, { cx: 1, cz: 0 }]);
    const [a2] = await generateChunks([{ cx: 0, cz: 0 }]);
    expect(a2).toEqual(a);
    // Surface heights on either side of the border differ by a small step, not a cliff.
    const surface = (c: Uint8Array, x: number, z: number) => {
      for (let y = CHUNK_HEIGHT - 1; y >= 0; y--) {
        const t = cellType(c[blockIndex(x, y, z)]);
        if (t === Block.Stone || t === Block.Dirt) return y;
      }
      return 0;
    };
    for (let z = 0; z < CHUNK_SIZE; z++) {
      expect(Math.abs(surface(a, CHUNK_SIZE - 1, z) - surface(b, 0, z))).toBeLessThanOrEqual(3);
    }
  });
});

describe('surfaceFeatures', () => {
  // A 256x256 area of land, all dry (or all under water).
  const area = (groundY: number) => {
    const n = 256;
    const wx = tf.range(-128, 128).reshape([1, 1, 1, n]).add(tf.zeros([1, 1, n, 1]));
    const wz = tf.range(-128, 128).reshape([1, 1, n, 1]).add(tf.zeros([1, 1, 1, n]));
    const f = surfaceFeatures(wx, wz, tf.fill([1, 1, n, n], groundY), 1337);
    return {
      grass: Array.from(f.grassSeed.dataSync()), wheat: Array.from(f.wheat.dataSync()),
      value: Array.from(f.wheatValue.dataSync()), n,
    };
  };

  it('seeds grass and wheat patches at about the intended rates on dry land', () => {
    const { grass, wheat, value, n } = area(SEA_LEVEL + 3);
    const cols = n * n;
    const grassCount = grass.reduce((a, b) => a + b, 0), wheatCount = wheat.reduce((a, b) => a + b, 0);
    expect(grassCount / (cols * GRASS_SEED_CHANCE)).toBeGreaterThan(0.5);
    expect(grassCount / (cols * GRASS_SEED_CHANCE)).toBeLessThan(1.6);
    expect(wheatCount / (cols * WHEAT_PATCH_CHANCE * 0.6)).toBeGreaterThan(0.5);
    expect(wheatCount / (cols * WHEAT_PATCH_CHANCE * 0.6)).toBeLessThan(1.6);
    // Wheat comes in patches: most plants have another plant in their 4x4 area, and every
    // stage from seedling to ripe shows up.
    const stages = new Set<number>();
    let clustered = 0;
    for (let i = 0; i < cols; i++) {
      if (!wheat[i]) continue;
      expect(cellType(value[i])).toBe(Block.Wheat);
      stages.add(value[i] >> 3);
      const x = i % n, z = Math.floor(i / n), x0 = x - (x % 4), z0 = z - (z % 4);
      let others = 0;
      for (let dz = 0; dz < 4; dz++) for (let dx = 0; dx < 4; dx++) if (wheat[(z0 + dz) * n + x0 + dx]) others++;
      if (others > 1) clustered++;
    }
    expect(clustered / wheatCount).toBeGreaterThan(0.8);
    expect(stages.size).toBe(8);
  });

  it('puts no plants on land under water', () => {
    const { grass, wheat } = area(SEA_LEVEL - 1);
    expect(grass.some((v) => v)).toBe(false);
    expect(wheat.some((v) => v)).toBe(false);
  });
});

describe('GPU-friendliness', () => {
  // tf.where with a scalar, or broadcastTo, runs TF.js's Tile kernel, which expands small
  // tensors on the CPU (main thread) even on GPU backends. Keep it out of the hot paths.
  it('worldgen and block updates never run the Tile kernel', async () => {
    const { blockUpdateStep } = await import('../src/tf/blockUpdate');
    const { generateChunksTensor } = await import('../src/tf/worldgen');
    const gen = await tf.profile(() => generateChunksTensor([{ cx: 0, cz: 0 }, { cx: 1, cz: 0 }]));
    expect(gen.kernelNames).not.toContain('Tile');
    (gen.result as tf.Tensor).dispose();
    const { randomField } = await import('../src/tf/random');
    const cells = tf.zeros([2, 8, 18, 18], 'int32') as tf.Tensor4D;
    const step = await tf.profile(() => blockUpdateStep(cells));
    expect(step.kernelNames).not.toContain('Tile');
    (step.result as tf.Tensor).dispose();
    const grassStep = await tf.profile(() => blockUpdateStep(cells, randomField([2, 8, 18, 18], [1, 2, 3])));
    expect(grassStep.kernelNames).not.toContain('Tile');
    expect(grassStep.kernelNames).toContain('MaxPool3D');
    (grassStep.result as tf.Tensor).dispose();
    cells.dispose();
  });
});
