import { describe, expect, it } from 'vitest';
import { Block } from '../src/constants';
import { parseModel } from '../src/render/mobModel';
import { NOT_LOADED } from '../src/sim/store';
import { Animals, MAX_ANIMALS, SPECIES } from '../src/world/animals';

const DT = 1 / 60;
/** Grass on top of stone up to y = 10 (the ground's top is y = 11), a wall at x = 30. */
const field = (x: number, y: number, _z?: number) => (x === 30 && y < 14 ? Block.Stone : y < 10 ? Block.Stone : y === 10 ? Block.Grass : Block.Air);
const seeded = (seed = 7) => () => (seed = (seed * 16807) % 2147483647) / 2147483647;

describe('farm animals', () => {
  it('spawn only on grass with room above, around the player', () => {
    const animals = new Animals(seeded());
    const centre = [0.5, 11, 0.5];
    for (let t = 0; t < 30; t += DT) animals.update(DT, centre, field);
    expect(animals.animals.length).toBe(MAX_ANIMALS);
    for (const p of animals.animals) {
      expect(p.feet[1]).toBeCloseTo(11, 1);
      expect(p.body.onGround).toBe(true);
    }
    // Nowhere to stand: no pigs.
    const stone = new Animals(seeded());
    for (let t = 0; t < 10; t += DT) stone.update(DT, centre, (_x, y) => (y <= 10 ? Block.Stone : Block.Air));
    expect(stone.animals).toHaveLength(0);
  });

  it('wander about, staying on the ground and out of walls', () => {
    const animals = new Animals(seeded(3));
    const centre = [0.5, 11, 0.5];
    for (let t = 0; t < 5; t += DT) animals.update(DT, centre, field);
    const start = new Map(animals.animals.map((p) => [p, [...p.feet]]));
    for (let t = 0; t < 30; t += DT) animals.update(DT, centre, field);
    let moved = 0;
    animals.animals.forEach((p) => {
      const s0 = start.get(p);
      if (s0 && Math.hypot(p.feet[0] - s0[0], p.feet[2] - s0[2]) > 1) moved++;
      expect(p.feet[1]).toBeGreaterThan(10.9);
      expect(p.feet[0] + p.species.size.halfWidth).toBeLessThanOrEqual(30); // never inside the wall
    });
    expect(moved).toBeGreaterThan(0);
  });

  it('wait where the blocks are not known, and despawn far away', () => {
    const animals = new Animals(seeded());
    for (let t = 0; t < 3; t += DT) animals.update(DT, [0.5, 11, 0.5], field);
    expect(animals.animals.length).toBeGreaterThan(0);
    const before = animals.animals.map((p) => [...p.feet]);
    animals.update(DT, [0.5, 11, 0.5], () => NOT_LOADED);
    expect(animals.animals.map((p) => p.feet)).toEqual(before);
    const old = [...animals.animals];
    animals.update(DT, [500, 11, 500], field);
    expect(animals.animals.some((p) => old.includes(p))).toBe(false);
  });

  it('are picked by a ray, and run off when hit', () => {
    const animals = new Animals(seeded());
    for (let t = 0; t < 3; t += DT) animals.update(DT, [0.5, 11, 0.5], field);
    const pig = animals.animals[0];
    const eye = [pig.feet[0] + 3, pig.feet[1] + 0.2, pig.feet[2]];
    const hit = animals.raycast(eye, [-1, 0, 0], 8);
    expect(hit?.animal).toBe(pig);
    expect(hit!.dist).toBeCloseTo(3 - pig.species.size.halfWidth, 3);
    expect(animals.raycast(eye, [1, 0, 0], 8)).toBeUndefined();
    const x0 = pig.feet[0];
    animals.hit(pig, eye);
    for (let t = 0; t < 1; t += DT) animals.update(DT, [0.5, 11, 0.5], field);
    expect(pig.feet[0]).toBeLessThan(x0 - 1); // pushed away from the player, and running
  });

  it('spawn a mix of species, and give each an instance by species: position, yaw, waddle, bob, middle', () => {
    const animals = new Animals(seeded(11));
    for (let t = 0; t < 30; t += DT) animals.update(DT, [0.5, 11, 0.5], field);
    const kinds = new Set(animals.animals.map((a) => a.species.name));
    expect(kinds.size).toBeGreaterThan(2);
    const inst = animals.instances();
    let total = 0;
    for (const [name, data] of inst) {
      const of = animals.animals.filter((a) => a.species.name === name);
      expect(data.length).toBe(of.length * 8);
      expect(data[0]).toBeCloseTo(of[0].feet[0], 4);
      expect(data[3]).toBeCloseTo(of[0].yaw, 4);
      expect(data[6]).toBeCloseTo(of[0].species.size.height / 2, 4);
      total += of.length;
    }
    expect(total).toBe(animals.animals.length);
    expect(SPECIES.map((s) => s.name).sort()).toEqual(['cat', 'chicken', 'cow', 'horse', 'mouse', 'pig', 'rabbit', 'sheep']);
  });
});

describe('mob model file', () => {
  it('parses into interleaved vertices (position, normal, uv, colour) and indices', () => {
    const n = 3, m = 3;
    const buf = new ArrayBuffer(8 + n * 48 + m * 2 + 2);
    new Uint32Array(buf, 0, 2).set([n, m]);
    const f = new Float32Array(buf, 8, n * 12);
    f.set([1, 0, 0, 0, 1, 0, 1, 0, 0.5, 0.25, 0, 1], 12);
    new Uint16Array(buf, 8 + n * 48, m).set([0, 1, 2]);
    const { vertices, indices } = parseModel(buf);
    expect(Array.from(vertices.subarray(12, 24))).toEqual([1, 0, 0, 0, 1, 0, 1, 0, 0.5, 0.25, 0, 1]);
    expect(Array.from(indices)).toEqual([0, 1, 2]);
  });
});
