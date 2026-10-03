import { describe, expect, it } from 'vitest';
import { Block } from '../src/constants';
import { parseModel, poseFrames } from '../src/render/mobModel';
import { NOT_LOADED } from '../src/sim/store';
import { Animals, MAX_ANIMALS, SPECIES } from '../src/world/animals';

const DT = 1 / 60;
/** Grass on top of stone up to y = 10 (the ground's top is y = 11), a wall at x = 30. */
const field = (x: number, y: number, _z?: number) => (x === 30 && y < 14 ? Block.Stone : y < 10 ? Block.Stone : y === 10 ? Block.Grass : Block.Air);
const seeded = (seed = 7) => () => (seed = (seed * 16807) % 2147483647) / 2147483647;

describe('animals', () => {
  it('spawn only on grass with room above, around the player', () => {
    const animals = new Animals(seeded());
    const centre = [0.5, 11, 0.5];
    for (let t = 0; t < 30; t += DT) animals.update(DT, centre, field);
    expect(animals.animals.length).toBe(MAX_ANIMALS);
    for (const p of animals.animals.filter((a) => !a.species.hops)) { // (rabbits may be mid-hop)
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

  it('spawn a mix of species, and give each an instance by species: position, yaw, and its pose frames', () => {
    const animals = new Animals(seeded(11));
    for (let t = 0; t < 30; t += DT) animals.update(DT, [0.5, 11, 0.5], field);
    const kinds = new Set(animals.animals.map((a) => a.species.name));
    expect(kinds.size).toBeGreaterThan(2);
    const asked: string[] = [];
    const inst = animals.instances((name, clip, time) => { asked.push(`${name} ${clip}`); return [1, 2, time]; });
    let total = 0;
    for (const [name, data] of inst) {
      const of = animals.animals.filter((a) => a.species.name === name);
      expect(data.length).toBe(of.length * 8);
      expect(data[0]).toBeCloseTo(of[0].feet[0], 4);
      expect(data[3]).toBeCloseTo(of[0].yaw, 4);
      expect([data[4], data[5]]).toEqual([1, 2]);
      expect(data[6]).toBeCloseTo(of[0].clipTime, 4);
      total += of.length;
    }
    expect(total).toBe(animals.animals.length);
    expect(asked).toHaveLength(animals.animals.length);
    expect(SPECIES.map((s) => s.name).sort()).toEqual(['cat', 'chicken', 'cow', 'deer', 'dog', 'fox', 'pig', 'rabbit']);
  });

  it('play the clip that fits what they do: walk, stand (idle or eat), run off when hit', () => {
    const animals = new Animals(seeded(5));
    const clips = new Set<string>();
    for (let t = 0; t < 40; t += DT) {
      animals.update(DT, [0.5, 11, 0.5], field);
      for (const a of animals.animals) {
        clips.add(a.clip);
        expect(a.clip).toBe(a.fleeing > 0 ? 'run' : a.walking ? 'walk' : a.eating ? 'eat' : 'idle');
      }
    }
    expect([...clips].sort()).toEqual(['eat', 'idle', 'walk']);
    const a = animals.animals[0];
    animals.hit(a, [a.feet[0] + 2, a.feet[1], a.feet[2]]);
    animals.update(DT, [0.5, 11, 0.5], field);
    expect(a.clip).toBe('run');
  });
});

describe('mob model file', () => {
  it('parses its header, vertices (position, normal, uv, part), indices and poses', () => {
    const header = { vertices: 3, indices: 3, parts: 2, frames: 2, fps: 30, clips: { idle: { start: 0, frames: 2 } } };
    const json = new TextEncoder().encode(JSON.stringify(header));
    const jsonBytes = Math.ceil(json.length / 4) * 4;
    const buf = new ArrayBuffer(4 + jsonBytes + 3 * 36 + 8 + 2 * 2 * 48);
    new Uint32Array(buf, 0, 1)[0] = jsonBytes;
    new Uint8Array(buf, 4, jsonBytes).fill(0x20).set(json);
    new Float32Array(buf, 4 + jsonBytes, 27).set([1, 0, 0, 0, 1, 0, 0.5, 0.25, 1], 18);
    new Uint16Array(buf, 4 + jsonBytes + 108, 3).set([0, 1, 2]);
    new Float32Array(buf, 4 + jsonBytes + 116, 48)[47] = 7;
    const m = parseModel(buf);
    expect(m.clips.idle).toEqual({ start: 0, frames: 2 });
    expect(Array.from(m.vertexData.subarray(18, 27))).toEqual([1, 0, 0, 0, 1, 0, 0.5, 0.25, 1]);
    expect(Array.from(m.indexData.subarray(0, 3))).toEqual([0, 1, 2]);
    expect(m.poseData.length).toBe(48);
    expect(m.poseData[47]).toBe(7);
  });

  it('finds the frames a time into a looping clip is between', () => {
    const model = { fps: 10, clips: { idle: { start: 0, frames: 11 }, walk: { start: 11, frames: 5 } } };
    expect(poseFrames(model, 'walk', 0)).toEqual([11, 12, 0]);
    const [a, b, f] = poseFrames(model, 'walk', 0.25);
    expect([a, b]).toEqual([13, 14]);
    expect(f).toBeCloseTo(0.5, 5);
    expect(poseFrames(model, 'walk', 0.4)[0]).toBe(11); // looped
    expect(poseFrames(model, 'missing', 0)[0]).toBe(0); // idle instead
  });
});

describe('animals in water', () => {
  // A pool: water from y = 6 to 10 (its surface is the top of y = 10) over stone, x < 20; grass beyond.
  const pool = (x: number, y: number) => (y < 6 ? Block.Stone : x >= 20 ? (y < 10 ? Block.Stone : y === 10 ? Block.Grass : Block.Air) : y <= 10 ? Block.Water : Block.Air);

  it('float at the surface rather than sinking, and swim out onto the bank', () => {
    const animals = new Animals(seeded(9));
    const a = animals.trySpawn([0, 11, 0], (_x, y) => (y === 10 ? Block.Grass : y < 10 ? Block.Stone : Block.Air), SPECIES[0])!;
    a.feet = [8.5, 7, 0.5]; // at the bottom
    a.yaw = a.heading = -Math.PI / 2; // facing +x, toward the bank
    a.until = 99;
    const heights: number[] = [];
    let out = false;
    for (let t = 0; t < 40 && !out; t += DT) {
      a.heading = -Math.PI / 2;
      animals.update(DT, [10, 11, 0], pool);
      if (t > 5 && a.feet[0] < 19) heights.push(a.feet[1]); // (once it has risen)
      out = a.feet[0] > 20.5 && a.body.onGround;
    }
    // Afloat, a little over half under the water's drawn surface (10.83), and never down at the bottom.
    const mean = heights.reduce((x, y) => x + y, 0) / heights.length, h = a.species.size.height;
    expect(mean).toBeGreaterThan(10.83 - h * 0.65);
    expect(mean).toBeLessThan(10.83 - h * 0.45);
    expect(Math.min(...heights)).toBeGreaterThan(8);
    expect(out).toBe(true);
    expect(a.feet[1]).toBeCloseTo(11, 1);
  });
});
