import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, blockIndex, cellType } from '../src/constants';
import { GEN_BATCH, generateChunks, type ChunkCoord } from '../src/tf/worldgen';

beforeAll(async () => {
  await tf.setBackend('cpu');
});

describe('generateChunks', () => {
  it('produces all four materials across a patch of chunks', async () => {
    const coords: ChunkCoord[] = [];
    for (let cz = -3; cz <= 3; cz++) for (let cx = -3; cx <= 3; cx++) coords.push({ cx, cz });
    const chunks: Uint8Array[] = [];
    for (let i = 0; i < coords.length; i += GEN_BATCH) chunks.push(...await generateChunks(coords.slice(i, i + GEN_BATCH)));
    expect(chunks).toHaveLength(coords.length);
    const counts = new Array(5).fill(0);
    for (const c of chunks) {
      expect(c.length).toBe(CHUNK_VOLUME);
      for (const v of c) counts[cellType(v)]++;
    }
    console.log('block counts [air, stone, dirt, water, lava]', counts);
    for (const b of [Block.Air, Block.Stone, Block.Dirt, Block.Water, Block.Lava]) expect(counts[b]).toBeGreaterThan(0);
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

describe('GPU-friendliness', () => {
  // tf.where with a scalar, or broadcastTo, runs TF.js's Tile kernel, which expands small
  // tensors on the CPU (main thread) even on GPU backends. Keep it out of the hot paths.
  it('worldgen and block updates never run the Tile kernel', async () => {
    const { blockUpdateStep } = await import('../src/tf/blockUpdate');
    const { generateChunksTensor } = await import('../src/tf/worldgen');
    const gen = await tf.profile(() => generateChunksTensor([{ cx: 0, cz: 0 }, { cx: 1, cz: 0 }]));
    expect(gen.kernelNames).not.toContain('Tile');
    (gen.result as tf.Tensor).dispose();
    const cells = tf.zeros([8, 32, 32], 'int32') as tf.Tensor3D;
    const step = await tf.profile(() => blockUpdateStep(cells));
    expect(step.kernelNames).not.toContain('Tile');
    (step.result as tf.Tensor).dispose();
    cells.dispose();
  });
});
