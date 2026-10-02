import { describe, expect, it } from 'vitest';
import { Block } from '../src/constants';
import { parseModel } from '../src/render/mobModel';
import { NOT_LOADED } from '../src/sim/store';
import { MAX_PIGS, PIG_SIZE, Pigs } from '../src/world/pigs';

const DT = 1 / 60;
/** Grass on top of stone up to y = 10 (the ground's top is y = 11), a wall at x = 30. */
const field = (x: number, y: number) => (x === 30 && y < 14 ? Block.Stone : y < 10 ? Block.Stone : y === 10 ? Block.Grass : Block.Air);
const seeded = (seed = 7) => () => (seed = (seed * 16807) % 2147483647) / 2147483647;

describe('pigs', () => {
  it('spawn only on grass with room above, around the player', () => {
    const pigs = new Pigs(seeded());
    const centre = [0.5, 11, 0.5];
    for (let t = 0; t < 30; t += DT) pigs.update(DT, centre, field);
    expect(pigs.pigs.length).toBe(MAX_PIGS);
    for (const p of pigs.pigs) {
      expect(p.feet[1]).toBeCloseTo(11, 1);
      expect(p.body.onGround).toBe(true);
    }
    // Nowhere to stand: no pigs.
    const stone = new Pigs(seeded());
    for (let t = 0; t < 10; t += DT) stone.update(DT, centre, (_x, y) => (y <= 10 ? Block.Stone : Block.Air));
    expect(stone.pigs).toHaveLength(0);
  });

  it('wander about, staying on the ground and out of walls', () => {
    const pigs = new Pigs(seeded(3));
    const centre = [0.5, 11, 0.5];
    for (let t = 0; t < 5; t += DT) pigs.update(DT, centre, field);
    const start = new Map(pigs.pigs.map((p) => [p, [...p.feet]]));
    for (let t = 0; t < 30; t += DT) pigs.update(DT, centre, field);
    let moved = 0;
    pigs.pigs.forEach((p) => {
      const s0 = start.get(p);
      if (s0 && Math.hypot(p.feet[0] - s0[0], p.feet[2] - s0[2]) > 1) moved++;
      expect(p.feet[1]).toBeGreaterThan(10.9);
      expect(p.feet[0] + PIG_SIZE.halfWidth).toBeLessThanOrEqual(30); // never inside the wall
    });
    expect(moved).toBeGreaterThan(0);
  });

  it('wait where the blocks are not known, and despawn far away', () => {
    const pigs = new Pigs(seeded());
    for (let t = 0; t < 3; t += DT) pigs.update(DT, [0.5, 11, 0.5], field);
    expect(pigs.pigs.length).toBeGreaterThan(0);
    const before = pigs.pigs.map((p) => [...p.feet]);
    pigs.update(DT, [0.5, 11, 0.5], () => NOT_LOADED);
    expect(pigs.pigs.map((p) => p.feet)).toEqual(before);
    const old = [...pigs.pigs];
    pigs.update(DT, [500, 11, 500], field);
    expect(pigs.pigs.some((p) => old.includes(p))).toBe(false);
  });

  it('are picked by a ray, and run off when hit', () => {
    const pigs = new Pigs(seeded());
    for (let t = 0; t < 3; t += DT) pigs.update(DT, [0.5, 11, 0.5], field);
    const pig = pigs.pigs[0];
    const eye = [pig.feet[0] + 3, pig.feet[1] + 0.5, pig.feet[2]];
    const hit = pigs.raycast(eye, [-1, 0, 0], 8);
    expect(hit?.pig).toBe(pig);
    expect(hit!.dist).toBeCloseTo(3 - PIG_SIZE.halfWidth, 3);
    expect(pigs.raycast(eye, [1, 0, 0], 8)).toBeUndefined();
    const x0 = pig.feet[0];
    pigs.hit(pig, eye);
    for (let t = 0; t < 1; t += DT) pigs.update(DT, [0.5, 11, 0.5], field);
    expect(pig.feet[0]).toBeLessThan(x0 - 1); // pushed away from the player, and running
  });

  it('give each pig an instance: position, yaw, waddle, bob', () => {
    const pigs = new Pigs(seeded());
    for (let t = 0; t < 3; t += DT) pigs.update(DT, [0.5, 11, 0.5], field);
    const inst = pigs.instances();
    expect(inst.length).toBe(pigs.pigs.length * 8);
    expect(inst[0]).toBeCloseTo(pigs.pigs[0].feet[0], 4);
    expect(inst[3]).toBeCloseTo(pigs.pigs[0].yaw, 4);
  });
});

describe('mob model file', () => {
  it('parses into interleaved vertices and indices', () => {
    const n = 3, m = 3;
    const buf = new ArrayBuffer(8 + n * 32 + m * 2 + 2);
    new Uint32Array(buf, 0, 2).set([n, m]);
    const f = new Float32Array(buf, 8, n * 8);
    f.set([0, 0, 0, 1, 0, 0, 0, 1, 0], 0); // positions
    f.set([0, 1, 0, 0, 1, 0, 0, 1, 0], 9); // normals
    f.set([0, 0, 1, 0, 0, 1], 18); // uvs
    new Uint16Array(buf, 8 + n * 32, m).set([0, 1, 2]);
    const { vertices, indices } = parseModel(buf);
    expect(Array.from(vertices.subarray(8, 16))).toEqual([1, 0, 0, 0, 1, 0, 1, 0]);
    expect(Array.from(indices)).toEqual([0, 1, 2]);
  });
});
