import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { CHUNK_VOLUME } from '../src/constants';
import { generateChunks } from '../src/tf/worldgen';
import { generateChunkJs } from '../src/world/jsWorldgen';

beforeAll(async () => {
  await tf.setBackend('cpu');
});

describe('plain-JS world generation', () => {
  it('matches the TF.js reference, cell for cell', async () => {
    const coords = [{ cx: 0, cz: 0 }, { cx: 3, cz: -2 }, { cx: -5, cz: 7 }, { cx: 9, cz: 9 }, { cx: -40, cz: 13 }];
    const ref = await generateChunks(coords);
    let differ = 0;
    coords.forEach((c, k) => {
      const js = generateChunkJs(c);
      for (let i = 0; i < CHUNK_VOLUME; i++) if (js[i] !== ref[k][i]) differ++;
    });
    expect(differ).toBe(0); // (float32 throughout, as TF.js on the CPU)
  });

  it('is quick enough to run a chunk or two a frame', () => {
    const t0 = performance.now();
    for (let i = 0; i < 10; i++) generateChunkJs({ cx: i, cz: 2 * i });
    const ms = (performance.now() - t0) / 10;
    console.info(`plain-JS worldgen: ${ms.toFixed(1)} ms a chunk`);
    expect(ms).toBeLessThan(20);
  });
});
