import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { randomField } from '../src/tf/random';

beforeAll(async () => {
  await tf.setBackend('cpu');
});

describe('randomField', () => {
  const shape = [4, 64, 18, 18] as const;
  const sample = (seeds: [number, number, number]) => {
    const t = randomField(shape, seeds);
    const v = Float32Array.from(t.dataSync());
    t.dispose();
    return v;
  };

  it('is uniform-ish in [0, 1)', () => {
    const v = sample([12.5, 340.25, 77.75]);
    const bins = new Array(10).fill(0);
    let sum = 0;
    for (const x of v) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
      bins[Math.floor(x * 10)]++;
      sum += x;
    }
    expect(sum / v.length).toBeCloseTo(0.5, 1);
    for (const b of bins) expect(Math.abs(b / v.length - 0.1)).toBeLessThan(0.02);
  });

  it('differs between chunks in a batch and between seeds', () => {
    const a = sample([1, 2, 3]), b = sample([4, 5, 6]);
    const chunk = 64 * 18 * 18;
    let sameItem = 0, sameSeed = 0;
    for (let i = 0; i < chunk; i++) {
      if (a[i] === a[i + chunk]) sameItem++;
      if (a[i] === b[i]) sameSeed++;
    }
    expect(sameItem / chunk).toBeLessThan(0.01);
    expect(sameSeed / chunk).toBeLessThan(0.01);
  });
});

describe('cellRandom (the fused kernel’s random numbers)', () => {
  it('is uniform in [0, 1) with 24-bit steps, and depends on the seed', async () => {
    const { cellRandom } = await import('../src/tf/blockUpdateReference');
    const n = 100_000, bins = new Array(10).fill(0);
    let sameAcrossSeeds = 0;
    for (let i = 0; i < n; i++) {
      const v = cellRandom(i, 0x9e3779b9);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
      expect(Number.isInteger(v * 2 ** 24)).toBe(true);
      bins[Math.floor(v * 10)]++;
      if (v === cellRandom(i, 12345)) sameAcrossSeeds++;
    }
    for (const b of bins) expect(Math.abs(b / n - 0.1)).toBeLessThan(0.01);
    expect(sameAcrossSeeds / n).toBeLessThan(0.001);
  });
});
