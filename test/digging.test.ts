import { describe, expect, it } from 'vitest';
import { Block } from '../src/constants';
import { DIG_COOLDOWN, DIG_SECONDS, Digging } from '../src/player/digging';
import { NOT_LOADED } from '../src/sim/store';
import { Drops } from '../src/world/drops';

const DT = 1 / 60;

/** Seconds of holding until the block breaks. */
function timeToBreak(d: Digging, block: number[], type: Block): number {
  let t = 0;
  while (!d.step(DT, true, block, type)) {
    t += DT;
    if (t > 10) throw new Error('never broke');
  }
  return t;
}

describe('digging', () => {
  it('takes longer for harder blocks; wheat goes at once', () => {
    for (const type of [Block.Dirt, Block.Grass, Block.Stone, Block.Diamond]) {
      expect(timeToBreak(new Digging(), [0, 5, 0], type)).toBeCloseTo(DIG_SECONDS[type]!, 1);
    }
    expect(new Digging().step(DT, true, [0, 5, 0], Block.Wheat)).toBe(true);
    expect(DIG_SECONDS[Block.Diamond]!).toBeGreaterThan(DIG_SECONDS[Block.Stone]!);
  });

  it('starts over when the target changes or the button is let go', () => {
    const d = new Digging();
    for (let i = 0; i < 30; i++) d.step(DT, true, [0, 5, 0], Block.Stone);
    expect(d.progress).toBeGreaterThan(0.5);
    d.step(DT, true, [1, 5, 0], Block.Stone);
    expect(d.progress).toBeLessThan(0.1);
    d.step(DT, false, [1, 5, 0], Block.Stone);
    expect(d.progress).toBe(0);
  });

  it('pauses briefly after breaking a block', () => {
    const d = new Digging();
    timeToBreak(d, [0, 5, 0], Block.Dirt);
    expect(timeToBreak(d, [0, 4, 0], Block.Dirt)).toBeCloseTo(DIG_SECONDS[Block.Dirt]! + DIG_COOLDOWN, 1);
  });
});

describe('dropped items', () => {
  const floor = (_x: number, y: number) => (y < 10 ? Block.Stone : Block.Air);

  it('pop out of the block, fall and rest on the ground', () => {
    const drops = new Drops();
    drops.spawn(4, 12, 4, () => 0);
    const far = [100, 11, 100];
    for (let t = 0; t < 3; t += DT) drops.update(DT, far, floor);
    expect(drops.items).toHaveLength(1);
    const [x, y, z] = drops.items[0].pos;
    expect(y).toBeGreaterThan(10);
    expect(y).toBeLessThan(10.3);
    expect(Math.hypot(x - 4.5, z - 4.5)).toBeLessThan(1.5);
  });

  it('fly to the player when close and are collected', () => {
    const drops = new Drops();
    drops.spawn(4, 9, 4); // a block in the ground that was just dug
    let got = 0;
    for (let t = 0; t < 0.2; t += DT) got += drops.update(DT, [6.5, 10.9, 4.5], floor);
    expect(got).toBe(0); // not straight away
    for (let t = 0; t < 2; t += DT) got += drops.update(DT, [6.5, 10.9, 4.5], floor);
    expect(got).toBe(1);
    expect(drops.items).toHaveLength(0);
  });

  it('stay put where the blocks are not known yet', () => {
    const drops = new Drops();
    drops.spawn(4, 20, 4);
    for (let t = 0; t < 1; t += DT) drops.update(DT, [100, 0, 100], () => NOT_LOADED);
    expect(drops.items[0].pos[1]).toBeCloseTo(20.5, 5);
  });

  it('are drawn as small diamonds of line segments', () => {
    const drops = new Drops();
    drops.spawn(0, 0, 0);
    const lines = drops.lines(1);
    expect(lines.length % 12).toBe(0); // pairs of (x, y, z, r, g, b)
    expect(lines.length).toBeGreaterThan(0);
  });
});
