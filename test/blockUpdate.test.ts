import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  Block, PRIMED_DIRT, SOURCE_LEVEL, WHEAT_GROW_CHANCE, WHEAT_GROW_CHANCE_WET, WHEAT_RIPE, cell, cellLevel, cellType,
} from '../src/constants';
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
    a[i] = r < 0.45 ? cell(Block.Air) : r < 0.55 ? cell(Block.Stone) : r < 0.67 ? cell(Block.Dirt)
      : r < 0.7 ? PRIMED_DIRT : r < 0.76 ? cell(Block.Grass) : r < 0.82 ? cell(Block.Wheat, level - 1)
        : r < 0.91 ? cell(Block.Water, level) : cell(Block.Lava, level);
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

  it('matches the reference with grass rules, batched, given the same random numbers', () => {
    const rand = rng(7);
    const N = 3, H2 = 7, D2 = 10, W2 = 13, n = H2 * D2 * W2;
    const chance = 0.25;
    let states = Array.from({ length: N }, () => randomRegion(n, rand));
    for (let step = 0; step < 4; step++) {
      const randoms = states.map(() => Float32Array.from({ length: n }, rand));
      const rates = { grassSpread: chance, wheatGrow: 0.125, wheatGrowWet: 0.375 };
      const expected = states.map((st, k) => blockUpdateReference(st, H2, D2, W2, randoms[k], rates));
      const cells = tf.tensor4d(Int32Array.from(states.flatMap((st) => [...st])), [N, H2, D2, W2], 'int32');
      const random = tf.tensor4d(Float32Array.from(randoms.flatMap((r) => [...r])), [N, H2, D2, W2]);
      const out = blockUpdateStep(cells, random, rates);
      const got = out.dataSync() as Int32Array;
      tf.dispose([cells, random, out]);
      for (let k = 0; k < N; k++) expect(Int32Array.from(got.subarray(k * n, (k + 1) * n))).toEqual(expected[k]);
      states = expected;
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

describe('grass', () => {
  // A dirt floor at y = 1 on stone, open air above.
  function dirtFloor(): Int32Array {
    const a = floorRegion();
    for (let z = 0; z < D; z++) for (let x = 0; x < W; x++) a[idx(x, 1, z)] = cell(Block.Dirt);
    return a;
  }
  const step = (a: Int32Array, randomValue: number) => {
    const t = tf.tensor3d(a, [H, D, W], 'int32');
    const r = tf.fill([H, D, W], randomValue);
    const out = blockUpdateStep(t, r);
    const v = Int32Array.from(out.dataSync() as Int32Array);
    tf.dispose([t, r, out]);
    expect(v).toEqual(blockUpdateReference(a, H, D, W, new Float32Array(H * D * W).fill(randomValue)));
    return v;
  };

  it('primes exposed dirt in reach and sprouts it when the random number is under the chance', () => {
    const a = dirtFloor();
    a[idx(4, 1, 4)] = cell(Block.Grass);
    const unlucky = step(a, 0.99);
    const lucky = step(a, 0);
    for (let z = 0; z < D; z++) {
      for (let x = 0; x < W; x++) {
        const near = Math.abs(x - 4) <= 1 && Math.abs(z - 4) <= 1 && !(x === 4 && z === 4);
        expect(unlucky[idx(x, 1, z)]).toBe(near ? PRIMED_DIRT : x === 4 && z === 4 ? cell(Block.Grass) : cell(Block.Dirt));
        expect(cellType(lucky[idx(x, 1, z)])).toBe(near || (x === 4 && z === 4) ? Block.Grass : Block.Dirt);
      }
    }
  });

  it('reaches dirt up to three blocks below and one above, one block sideways', () => {
    const a = floorRegion();
    a[idx(4, 4, 4)] = cell(Block.Grass); // on a pillar
    for (let y = 1; y < 4; y++) a[idx(4, y, 4)] = cell(Block.Stone);
    a[idx(5, 1, 4)] = cell(Block.Dirt); // 3 below, beside: in reach
    a[idx(3, 1, 4)] = cell(Block.Stone);
    a[idx(3, 0, 4)] = cell(Block.Dirt); // 4 below: no (and not exposed anyway)
    a[idx(4, 5, 5)] = cell(Block.Dirt); // 1 above, beside: in reach
    a[idx(6, 1, 4)] = cell(Block.Dirt); // 2 sideways: no
    const out = step(a, 0.99);
    expect(out[idx(5, 1, 4)]).toBe(PRIMED_DIRT);
    expect(out[idx(4, 5, 5)]).toBe(PRIMED_DIRT);
    expect(out[idx(6, 1, 4)]).toBe(cell(Block.Dirt));
    expect(out[idx(3, 0, 4)]).toBe(cell(Block.Dirt));
  });

  it('spreads along the bottom layer of a region too', () => {
    const a = new Int32Array(H * D * W);
    for (let z = 0; z < D; z++) for (let x = 0; x < W; x++) a[idx(x, 0, z)] = cell(Block.Dirt);
    a[idx(4, 0, 4)] = cell(Block.Grass);
    const out = step(a, 0.99);
    expect(out[idx(5, 0, 5)]).toBe(PRIMED_DIRT);
  });

  it('does not spread onto covered dirt', () => {
    const a = dirtFloor();
    a[idx(4, 1, 4)] = cell(Block.Grass);
    a[idx(5, 2, 4)] = cell(Block.Stone);
    a[idx(3, 2, 4)] = cell(Block.Water, 3); // flowing water also covers dirt
    const out = step(a, 0);
    expect(out[idx(5, 1, 4)]).toBe(cell(Block.Dirt));
    expect(out[idx(3, 1, 4)]).toBe(cell(Block.Dirt));
    expect(cellType(out[idx(4, 1, 5)])).toBe(Block.Grass);
  });

  it('dies under solid blocks and fluid sources, survives under flowing fluid', () => {
    const a = dirtFloor();
    a[idx(2, 1, 2)] = cell(Block.Grass); a[idx(2, 2, 2)] = cell(Block.Stone);
    a[idx(4, 1, 4)] = cell(Block.Grass); a[idx(4, 2, 4)] = cell(Block.Water, SOURCE_LEVEL);
    a[idx(6, 1, 6)] = cell(Block.Grass); a[idx(6, 2, 6)] = cell(Block.Water, 3);
    const out = step(a, 0.99);
    expect(out[idx(2, 1, 2)]).toBe(cell(Block.Dirt));
    expect(out[idx(4, 1, 4)]).toBe(cell(Block.Dirt));
    expect(out[idx(6, 1, 6)]).toBe(cell(Block.Grass));
  });

  it('leaves dirt and grass alone without a random field (the no-grass fast path)', () => {
    const a = dirtFloor();
    a[idx(4, 1, 4)] = cell(Block.Grass);
    a[idx(5, 1, 4)] = PRIMED_DIRT;
    a[idx(2, 1, 2)] = cell(Block.Grass); a[idx(2, 2, 2)] = cell(Block.Stone);
    expect(run(a, 1)).toEqual(a);
  });
});

describe('wheat', () => {
  // Dirt floor at y = 1 with a wheat plant on top at (4, 2, 4).
  function field(stage = 0): Int32Array {
    const a = floorRegion();
    for (let z = 0; z < D; z++) for (let x = 0; x < W; x++) a[idx(x, 1, z)] = cell(Block.Dirt);
    a[idx(4, 2, 4)] = cell(Block.Wheat, stage);
    return a;
  }
  const step = (a: Int32Array, randomValue: number) => {
    const t = tf.tensor3d(a, [H, D, W], 'int32');
    const r = tf.fill([H, D, W], randomValue);
    const out = blockUpdateStep(t, r);
    const v = Int32Array.from(out.dataSync() as Int32Array);
    tf.dispose([t, r, out]);
    expect(v).toEqual(blockUpdateReference(a, H, D, W, new Float32Array(H * D * W).fill(randomValue)));
    return v;
  };
  const between = (WHEAT_GROW_CHANCE + WHEAT_GROW_CHANCE_WET) / 2;

  it('grows a stage when its roll comes up, and stops when ripe', () => {
    expect(step(field(3), 0)[idx(4, 2, 4)]).toBe(cell(Block.Wheat, 4));
    expect(step(field(3), 0.99)[idx(4, 2, 4)]).toBe(cell(Block.Wheat, 3));
    expect(step(field(WHEAT_RIPE), 0)[idx(4, 2, 4)]).toBe(cell(Block.Wheat, WHEAT_RIPE));
  });

  it('grows faster with water next to its soil', () => {
    const dry = field(2);
    const wet = field(2);
    wet[idx(5, 1, 4)] = cell(Block.Water, SOURCE_LEVEL); // a water block beside the soil
    expect(step(dry, between)[idx(4, 2, 4)]).toBe(cell(Block.Wheat, 2));
    expect(step(wet, between)[idx(4, 2, 4)]).toBe(cell(Block.Wheat, 3));
  });

  it('pops off without dirt or grass under it', () => {
    const a = field(5);
    a[idx(4, 1, 4)] = cell(Block.Stone);
    expect(step(a, 0.99)[idx(4, 2, 4)]).toBe(cell(Block.Air));
    const onGrass = field(5);
    onGrass[idx(4, 1, 4)] = cell(Block.Grass);
    expect(step(onGrass, 0.99)[idx(4, 2, 4)]).toBe(cell(Block.Wheat, 5));
  });

  it('is washed away by flowing water, even without the plant rules', () => {
    const a = field(6);
    a[idx(3, 2, 4)] = cell(Block.Water, SOURCE_LEVEL);
    expect(step(a, 0.99)[idx(4, 2, 4)]).toBe(cell(Block.Water, 7));
    expect(run(a, 1)[idx(4, 2, 4)]).toBe(cell(Block.Water, 7));
  });

  it('stops grass spreading onto the soil it grows in', () => {
    const a = field(1);
    a[idx(3, 1, 4)] = cell(Block.Grass);
    const out = step(a, 0);
    expect(out[idx(4, 1, 4)]).toBe(cell(Block.Dirt)); // covered by the plant: not exposed
    expect(cellType(out[idx(3, 1, 3)])).toBe(Block.Grass);
  });
});
