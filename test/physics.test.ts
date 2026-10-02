import { describe, expect, it } from 'vitest';
import { Block, CHUNK_SIZE, cell } from '../src/constants';
import {
  BODY_HEIGHT, Body, HALF_WIDTH, JUMP_SPEED, GRAVITY, Nearby, moveBody, overlaps, type WalkInput,
} from '../src/player/physics';
import { CpuStore } from '../src/sim/cpuStore';
import { NOT_LOADED, ringSize } from '../src/sim/store';
import { World } from '../src/world/world';

type TypeAt = (x: number, y: number, z: number) => number;
/** Stone up to y = 9 (so the ground's top is y = 10), plus extra blocks. */
const ground = (extra: Record<string, number> = {}): TypeAt => (x, y, z) =>
  extra[`${x},${y},${z}`] ?? (y < 10 ? Block.Stone : Block.Air);
const still: WalkInput = { right: 0, forward: 0, jump: false, run: false };
const DT = 1 / 60;

/** Run `seconds` of steps; yaw 0 faces -z, so "forward" walks toward -z. */
function run(body: Body, feet: number[], typeAt: TypeAt, seconds: number, input = still, yaw = 0): void {
  for (let t = 0; t < seconds; t += DT) body.step(feet, DT, yaw, input, typeAt);
}

describe('player physics', () => {
  it('falls and lands on the ground', () => {
    const body = new Body(), feet = [0.5, 20, 0.5];
    run(body, feet, ground(), 2);
    expect(feet[1]).toBeCloseTo(10, 2);
    expect(feet[1]).toBeGreaterThanOrEqual(10);
    expect(body.onGround).toBe(true);
  });

  it("doesn't fall through the floor however fast it goes", () => {
    const feet = [0.5, 30, 0.5];
    const hit = moveBody(feet, [0, -25, 0], ground());
    expect(hit[1]).toBe(true);
    expect(feet[1]).toBeCloseTo(10, 2);
  });

  it('stops flush against a wall, and slides along it', () => {
    const wall: TypeAt = (x, y) => (x === 5 || y < 10 ? Block.Stone : Block.Air);
    const feet = [2.5, 10, 0.5];
    const hit = moveBody(feet, [4, 0, 3], wall);
    expect(hit).toEqual([true, false, false]);
    expect(feet[0]).toBeCloseTo(5 - HALF_WIDTH, 2);
    expect(feet[2]).toBeCloseTo(3.5, 5);
  });

  it("bumps its head on a ceiling 2 blocks up, and fits under it", () => {
    const roof = ground({ '0,12,0': Block.Stone });
    const feet = [0.5, 10, 0.5];
    expect(overlaps(feet, roof)).toBe(false); // 1.8 tall in a 2-block gap
    const hit = moveBody(feet, [0, 1, 0], roof);
    expect(hit[1]).toBe(true);
    expect(feet[1] + BODY_HEIGHT).toBeCloseTo(12, 2);
  });

  it('jumps about 1.25 blocks: onto one block, not two', () => {
    expect(JUMP_SPEED ** 2 / (2 * GRAVITY)).toBeGreaterThan(1.1);
    expect(JUMP_SPEED ** 2 / (2 * GRAVITY)).toBeLessThan(1.5);
    const forwardJump: WalkInput = { right: 0, forward: 1, jump: true, run: false };
    // A one-block step at z = -2.
    const step = (h: number): TypeAt => (_x, y, z) => (y < 10 || (z <= -2 && y < 10 + h) ? Block.Stone : Block.Air);
    for (const [h, top] of [[1, 11], [2, 10]]) {
      const body = new Body(), feet = [0.5, 10, 0.5];
      run(body, feet, step(h), 0.6, { ...forwardJump, jump: false }); // walk up to it
      run(body, feet, step(h), 0.5, forwardJump);
      run(body, feet, step(h), 1, { ...forwardJump, jump: false });
      expect(feet[1]).toBeCloseTo(top, 1);
    }
  });

  it('walks and runs at their speeds, and stops when the controls do', () => {
    for (const runs of [false, true]) {
      const body = new Body(), feet = [0.5, 10, 0.5];
      run(body, feet, ground(), 0.1);
      const z0 = feet[2];
      run(body, feet, ground(), 2, { right: 0, forward: 1, jump: false, run: runs });
      const speed = (z0 - feet[2]) / 2;
      expect(speed).toBeGreaterThan(runs ? 6 : 4);
      expect(speed).toBeLessThan(runs ? 7.1 : 4.6);
      const z1 = feet[2];
      run(body, feet, ground(), 1);
      expect(Math.abs(feet[2] - z1)).toBeLessThan(1);
    }
  });

  it('climbs out of a block it is stuck in', () => {
    const body = new Body(), feet = [0.5, 5, 0.5]; // inside the ground
    run(body, feet, ground(), 0.5);
    expect(feet[1]).toBeCloseTo(10, 2);
  });

  it("waits while the blocks around it aren't loaded", () => {
    const body = new Body(), feet = [0.5, 20, 0.5];
    run(body, feet, () => NOT_LOADED, 1);
    expect(feet).toEqual([0.5, 20, 0.5]);
    // And unloaded chunks count as solid to walk into.
    const edge: TypeAt = (x, y) => (x >= 3 ? NOT_LOADED : y < 10 ? Block.Stone : Block.Air);
    const at = [0.5, 10, 0.5];
    moveBody(at, [10, 0, 0], edge);
    expect(at[0]).toBeCloseTo(3 - HALF_WIDTH, 2);
  });

  it('sinks slowly in water and swims up', () => {
    const lake: TypeAt = (_x, y) => (y < 2 ? Block.Stone : y < 10 ? Block.Water : Block.Air);
    const body = new Body(), feet = [0.5, 8, 0.5];
    run(body, feet, lake, 0.5);
    expect(body.inFluid).toBe(true);
    expect(feet[1]).toBeGreaterThan(6.5);
    const y = feet[1];
    run(body, feet, lake, 0.5, { right: 0, forward: 0, jump: true, run: false });
    expect(feet[1]).toBeGreaterThan(y + 1);
  });
});

describe('Nearby / readBox', () => {
  it('reads the blocks around a point from the world, unloaded chunks marked', async () => {
    const world = new World(new CpuStore(ringSize(1)), 0, 1);
    world.recenter(8, 8);
    for (const c of world.missingChunks(100)) if (c.cx !== 1 || c.cz !== 0) world.addGenerated(c, new Int32Array(16 * 16 * 64).fill(0).map((_, i) => (i < 256 * 10 ? cell(Block.Stone) : 0)));
    world.setCell(3, 10, 4, cell(Block.Dirt));
    const min = [0, 8, 0], size = [CHUNK_SIZE + 4, 4, 6];
    const near = new Nearby(min, size, await world.store.readBox(min, size));
    expect(near.at(3, 10, 4)).toBe(Block.Dirt);
    expect(near.at(3, 9, 4)).toBe(Block.Stone);
    expect(near.at(3, 11, 4)).toBe(Block.Air);
    expect(near.at(CHUNK_SIZE + 1, 9, 2)).toBe(NOT_LOADED); // chunk (1, 0) isn't loaded
    expect(near.at(-1, 9, 2)).toBe(NOT_LOADED); // outside the box
    near.set(3, 11, 4, Block.Stone);
    expect(near.at(3, 11, 4)).toBe(Block.Stone);
  });
});
