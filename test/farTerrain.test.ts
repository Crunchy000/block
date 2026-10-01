import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, blockIndex, cellType } from '../src/constants';
import { generateChunks } from '../src/tf/worldgen';
import { FAR_SNAP, SEA_SURFACE, farAxis, farHeights, farIndices, farVertices } from '../src/world/farTerrain';

beforeAll(async () => {
  await tf.setBackend('cpu');
});

describe('far terrain', () => {
  it('spaces grid points 4 apart near the centre and wider with distance, symmetric', () => {
    const axis = farAxis(2048);
    expect(axis[0]).toBeLessThanOrEqual(-2048);
    expect(axis.at(-1)).toBeGreaterThanOrEqual(2048);
    expect(axis.map((v) => 0 - v).reverse()).toEqual(axis);
    const mid = axis.indexOf(0);
    expect(axis[mid + 1] - axis[mid]).toBe(4);
    for (let i = 1; i < axis.length; i++) {
      const step = axis[i] - axis[i - 1];
      expect(FAR_SNAP % step).toBe(0); // so recentring by FAR_SNAP keeps points on the same columns
      expect(step).toBeLessThanOrEqual(FAR_SNAP);
    }
    expect(axis.length).toBeLessThan(250);
  });

  it('has the height of the generated ground', async () => {
    // Chunk (2, -1) and some grid points inside it.
    const [chunk] = await generateChunks([{ cx: 2, cz: -1 }]);
    const axis = [0, 4, 8, 12];
    const heights = await farHeights(axis, 2 * CHUNK_SIZE, -CHUNK_SIZE);
    axis.forEach((_, j) => axis.forEach((__, i) => {
      const x = axis[i], z = axis[j];
      let top = CHUNK_HEIGHT - 1;
      while (top > 0 && [Block.Air, Block.Water, Block.Wheat].includes(cellType(chunk[blockIndex(x, top, z)]))) top--;
      // Caves can open the surface's top block only if they reach it; they stay 4 below.
      expect(heights[j * axis.length + i]).toBe(top);
    }));
  });

  it('puts vertices on top of the ground or the sea, two triangles per cell', () => {
    const axis = [-4, 0, 4];
    const v = farVertices(axis, 64, 128, Float32Array.from([10, 40, 29, 30, 31, 0, 50, 20, 33]));
    expect(Array.from(v.slice(0, 3))).toEqual([60, SEA_SURFACE, 124]);
    expect(Array.from(v.slice(3, 6))).toEqual([64, 41, 124]);
    expect(v[3 * 3 + 1]).toBe(SEA_SURFACE + 0.125); // height 30 (sea level): ground top at 31
    expect(farIndices(3).length).toBe(4 * 6);
    expect(Math.max(...farIndices(3))).toBe(8);
  });
});
