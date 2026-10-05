import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, GRAVEL, SAND, SEA_LEVEL, WHEAT_RIPE, blockId, blockIndex, cell, cellType,
} from '../src/constants';
import {
  DIAMOND_MAX_Y, GEN_BATCH, SAND_ABOVE, SAND_BELOW, WHEAT_PATCH_CHANCE, WILD_WHEAT, generateChunks, surfaceFeatures, type ChunkCoord,
} from '../src/tf/worldgen';

beforeAll(async () => {
  await tf.setBackend('cpu');
});

describe('generateChunks', () => {
  it('scatters lots of diamond ore through the deep stone', async () => {
    const coords: ChunkCoord[] = Array.from({ length: GEN_BATCH }, (_, i) => ({ cx: i % 4 - 2, cz: Math.floor(i / 4) - 2 }));
    const chunks = await generateChunks(coords);
    let stone = 0, diamond = 0;
    for (const c of chunks) {
      c.forEach((v, i) => {
        const t = cellType(v), y = Math.floor(i / (CHUNK_SIZE * CHUNK_SIZE));
        if (t === Block.Diamond) {
          diamond++;
          expect(y).toBeGreaterThan(0);
          expect(y).toBeLessThanOrEqual(DIAMOND_MAX_Y);
        } else if (t === Block.Stone && y > 0 && y <= DIAMOND_MAX_Y) stone++;
      });
    }
    // A few percent of the deep stone: well over a hundred a chunk.
    expect(diamond / (diamond + stone)).toBeGreaterThan(0.02);
    expect(diamond / (diamond + stone)).toBeLessThan(0.06);
    expect(diamond / chunks.length).toBeGreaterThan(100);
  });

  it('produces the materials, with plants placed on the ground', async () => {
    const coords: ChunkCoord[] = [];
    // (Spread out, so the sample has sea and land whatever the terrain near the origin is.)
    for (let cz = -3; cz <= 3; cz++) for (let cx = -3; cx <= 3; cx++) coords.push({ cx: cx * 6, cz: cz * 6 });
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
    // Grass tops dry land: under air or wheat, never under water, never below sea level;
    // all wheat is ripe; lava never touches water.
    const layer = CHUNK_SIZE * CHUNK_SIZE;
    chunks.forEach((c) => c.forEach((v, i) => {
      const t = cellType(v), above = cellType(c[i + layer] ?? 0);
      if (t === Block.Grass) {
        expect([Block.Air, Block.Wheat]).toContain(above);
        expect(Math.floor(i / layer)).toBeGreaterThanOrEqual(SEA_LEVEL);
      }
      if (t === Block.Dirt && Math.floor(i / layer) >= SEA_LEVEL) expect(above).not.toBe(Block.Air); // no bare dirt on dry land
      if (t === Block.Wheat) expect(v).toBe(cell(Block.Wheat, WHEAT_RIPE));
      if (t === Block.Lava) {
        const x = i % CHUNK_SIZE, z = Math.floor(i / CHUNK_SIZE) % CHUNK_SIZE, y = Math.floor(i / layer);
        for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
          const xx = x + dx, yy = y + dy, zz = z + dz;
          if (xx < 0 || xx >= CHUNK_SIZE || zz < 0 || zz >= CHUNK_SIZE || yy < 0 || yy >= CHUNK_HEIGHT) continue;
          expect(cellType(c[i + dx + dz * CHUNK_SIZE + dy * layer])).not.toBe(Block.Water);
        }
      }
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
      grass: Array.from(f.grass.dataSync()), wheat: Array.from(f.wheat.dataSync()), n,
    };
  };

  it('covers dry land in grass, with wheat patches at about the intended rate', () => {
    const { grass, wheat, n } = area(SEA_LEVEL + 3);
    const cols = n * n;
    const wheatCount = wheat.reduce((a, b) => a + b, 0);
    expect(grass.every((v) => v)).toBe(true);
    expect(wheatCount / (cols * WHEAT_PATCH_CHANCE * 0.6)).toBeGreaterThan(0.5);
    expect(wheatCount / (cols * WHEAT_PATCH_CHANCE * 0.6)).toBeLessThan(1.6);
    // Wheat comes in patches: most plants have another plant in their 4x4 area.
    let clustered = 0;
    for (let i = 0; i < cols; i++) {
      if (!wheat[i]) continue;
      const x = i % n, z = Math.floor(i / n), x0 = x - (x % 4), z0 = z - (z % 4);
      let others = 0;
      for (let dz = 0; dz < 4; dz++) for (let dx = 0; dx < 4; dx++) if (wheat[(z0 + dz) * n + x0 + dx]) others++;
      if (others > 1) clustered++;
    }
    expect(clustered / wheatCount).toBeGreaterThan(0.8);
  });

  it('puts no plants on land under water', () => {
    const { grass, wheat } = area(SEA_LEVEL - 1);
    expect(grass.some((v) => v)).toBe(false);
    expect(wheat.some((v) => v)).toBe(false);
  });
});

describe('sand and gravel', () => {
  const column = (groundY: number) => {
    const f = surfaceFeatures(tf.zeros([1, 1, 1, 1]), tf.zeros([1, 1, 1, 1]), tf.fill([1, 1, 1, 1], groundY), 1337);
    return { grass: f.grass.dataSync()[0], sand: f.sand.dataSync()[0], gravel: f.gravel.dataSync()[0], bare: f.bare.dataSync()[0] };
  };

  it('makes beaches and shallow sea floors sand, deeper sea floors gravel, the rest dirt', () => {
    expect(column(SEA_LEVEL + SAND_ABOVE + 1)).toEqual({ grass: 1, sand: 0, gravel: 0, bare: 1 });
    expect(column(SEA_LEVEL + SAND_ABOVE)).toEqual({ grass: 0, sand: 1, gravel: 0, bare: 0 });
    expect(column(SEA_LEVEL - SAND_BELOW)).toEqual({ grass: 0, sand: 1, gravel: 0, bare: 0 });
    expect(column(SEA_LEVEL - SAND_BELOW - 1)).toEqual({ grass: 0, sand: 0, gravel: 1, bare: 0 });
  });

  it('generates them in place of the top dirt, and no wild wheat while it is off', async () => {
    const chunks = await generateChunks([{ cx: 0, cz: 0 }, { cx: 3, cz: -2 }, { cx: -5, cz: 7 }, { cx: 9, cz: 9 }]);
    const ids = new Set(chunks.flatMap((c) => Array.from(c, blockId)));
    expect(ids.has(Block.Sand) || ids.has(Block.Gravel)).toBe(true);
    if (!WILD_WHEAT) expect(ids.has(Block.Wheat)).toBe(false);
  });

  it('reads as their own blocks, though stored as stone', () => {
    expect(cellType(SAND)).toBe(Block.Stone);
    expect(blockId(SAND)).toBe(Block.Sand);
    expect(blockId(GRAVEL)).toBe(Block.Gravel);
    expect(blockId(cell(Block.Stone))).toBe(Block.Stone);
    expect(blockId(cell(Block.Water, 8))).toBe(Block.Water);
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
    const cells = tf.zeros([2, 8, 18, 18], 'int32') as tf.Tensor4D;
    const random = tf.fill([2, 8, 18, 18], 0.5) as tf.Tensor4D;
    const step = await tf.profile(() => blockUpdateStep(cells));
    expect(step.kernelNames).not.toContain('Tile');
    (step.result as tf.Tensor).dispose();
    const grassStep = await tf.profile(() => blockUpdateStep(cells, random));
    expect(grassStep.kernelNames).not.toContain('Tile');
    expect(grassStep.kernelNames).toContain('MaxPool3D');
    (grassStep.result as tf.Tensor).dispose();
    tf.dispose([cells, random]);
  });
});
