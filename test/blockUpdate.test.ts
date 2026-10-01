import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Block, SOURCE_LEVEL, cell, cellLevel, cellType } from '../src/constants';
import { blockUpdateStep } from '../src/tf/blockUpdate';
import { blockUpdateReference } from '../src/tf/blockUpdateReference';

const H = 6, D = 9, W = 9;
const idx = (x: number, y: number, z: number) => (y * D + z) * W + x;

/** Region with a stone floor at y=0 and air above. */
function floorRegion(): Int32Array {
  const a = new Int32Array(H * D * W);
  for (let z = 0; z < D; z++) for (let x = 0; x < W; x++) a[idx(x, 0, z)] = cell(Block.Stone);
  return a;
}

function run(a: Int32Array, steps: number, shape: [number, number, number] = [H, D, W]): Int32Array {
  let t = tf.tensor3d(a, shape, 'int32');
  for (let i = 0; i < steps; i++) {
    const n = blockUpdateStep(t);
    t.dispose();
    t = n;
  }
  const out = t.dataSync() as Int32Array;
  t.dispose();
  return Int32Array.from(out);
}

/** Small deterministic PRNG (mulberry32). */
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomRegion(n: number, rand: () => number): Int32Array {
  const a = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const r = rand(), level = 1 + Math.floor(rand() * SOURCE_LEVEL);
    a[i] = r < 0.5 ? cell(Block.Air) : r < 0.65 ? cell(Block.Stone) : r < 0.75 ? cell(Block.Dirt)
      : r < 0.88 ? cell(Block.Water, level) : cell(Block.Lava, level);
  }
  return a;
}

beforeAll(async () => {
  await tf.setBackend('cpu');
});

describe('blockUpdateStep', () => {
  it('matches the cell-by-cell reference on random regions', () => {
    const rand = rng(42);
    const shape: [number, number, number] = [7, 10, 13]; // non-cubic, so axis mix-ups show
    const n = shape[0] * shape[1] * shape[2];
    for (let trial = 0; trial < 5; trial++) {
      let state = randomRegion(n, rand);
      for (let step = 0; step < 4; step++) {
        const expected = blockUpdateReference(state, ...shape);
        expect(run(state, 1, shape)).toEqual(expected);
        state = expected;
      }
    }
  });

  it('leaves no tensors behind', () => {
    const before = tf.memory().numTensors;
    const t = tf.tensor3d(floorRegion(), [H, D, W], 'int32');
    blockUpdateStep(t).dispose();
    t.dispose();
    expect(tf.memory().numTensors).toBe(before);
  });

  it('water falls straight down', () => {
    const a = floorRegion();
    a[idx(4, 4, 4)] = cell(Block.Water, SOURCE_LEVEL);
    const out = run(a, 1);
    expect(cellType(out[idx(4, 3, 4)])).toBe(Block.Water);
    expect(cellType(out[idx(5, 4, 4)])).toBe(Block.Air); // unsupported: no sideways spread
    expect(out[idx(4, 4, 4)]).toBe(cell(Block.Water, SOURCE_LEVEL)); // source persists
  });

  it('water spreads on the floor and decays with distance', () => {
    const a = floorRegion();
    a[idx(4, 1, 4)] = cell(Block.Water, SOURCE_LEVEL);
    const out = run(a, 4);
    expect(cellLevel(out[idx(5, 1, 4)])).toBe(7);
    expect(cellLevel(out[idx(6, 1, 4)])).toBe(6);
    expect(cellLevel(out[idx(8, 1, 4)])).toBe(4);
    expect(cellType(out[idx(4, 2, 4)])).toBe(Block.Air); // doesn't climb
  });

  it('lava spreads less far than water', () => {
    const a = floorRegion();
    a[idx(4, 1, 4)] = cell(Block.Lava, SOURCE_LEVEL);
    const out = run(a, 6);
    expect(cellLevel(out[idx(5, 1, 4)])).toBe(6);
    expect(cellLevel(out[idx(7, 1, 4)])).toBe(2);
    expect(cellType(out[idx(8, 1, 4)])).toBe(Block.Air);
  });

  it('flowing water drains when its source is removed', () => {
    const a = floorRegion();
    a[idx(4, 1, 4)] = cell(Block.Water, SOURCE_LEVEL);
    const spread = run(a, 8);
    spread[idx(4, 1, 4)] = cell(Block.Air);
    const drained = run(spread, 10);
    for (let i = 0; i < drained.length; i++) expect(cellType(drained[i])).not.toBe(Block.Water);
  });

  it('lava meeting water turns to stone', () => {
    const a = floorRegion();
    a[idx(2, 1, 4)] = cell(Block.Water, SOURCE_LEVEL);
    a[idx(6, 1, 4)] = cell(Block.Lava, SOURCE_LEVEL);
    const out = run(a, 6);
    let stoneAboveFloor = 0;
    for (let z = 0; z < D; z++) for (let x = 0; x < W; x++) if (cellType(out[idx(x, 1, z)]) === Block.Stone) stoneAboveFloor++;
    expect(stoneAboveFloor).toBeGreaterThan(0);
  });

  it('never changes solid blocks', () => {
    const a = floorRegion();
    a[idx(3, 1, 3)] = cell(Block.Dirt);
    a[idx(3, 2, 3)] = cell(Block.Water, SOURCE_LEVEL);
    const out = run(a, 3);
    expect(out[idx(3, 1, 3)]).toBe(cell(Block.Dirt));
    for (let z = 0; z < D; z++) for (let x = 0; x < W; x++) expect(out[idx(x, 0, z)]).toBe(cell(Block.Stone));
  });
});
