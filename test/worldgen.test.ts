import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, blockIndex, cellType } from '../src/constants';
import { generateChunks } from '../src/tf/worldgen';

beforeAll(async () => {
  await tf.setBackend('cpu');
});

describe('generateChunks', () => {
  it('produces all four materials across a patch of chunks', async () => {
    const coords = [];
    for (let cz = -3; cz <= 3; cz++) for (let cx = -3; cx <= 3; cx++) coords.push({ cx, cz });
    const chunks = await generateChunks(coords);
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
