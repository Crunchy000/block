import { describe, expect, it } from 'vitest';
import { readStick } from '../src/player/stick';

const R = 56, DZ = 0.12;

describe('readStick', () => {
  it('reads zero with no drag', () => {
    expect(readStick(0, 0, R, DZ)).toEqual({ right: 0, forward: 0, magnitude: 0, knobX: 0, knobY: 0 });
  });

  it('ignores small drags inside the dead zone but still moves the knob', () => {
    const r = readStick(5, 0, R, DZ);
    expect(r.magnitude).toBe(0);
    expect(r.right).toBe(0);
    expect(r.knobX).toBe(5);
  });

  it('maps up the screen to forward and clamps the knob to the radius', () => {
    const r = readStick(0, -200, R, DZ);
    expect(r.forward).toBeCloseTo(1);
    expect(r.right).toBeCloseTo(0);
    expect(r.magnitude).toBe(1);
    expect(r.knobY).toBe(-R);
  });

  it('is analog between the dead zone and the rim', () => {
    const half = R * (DZ + (1 - DZ) * 0.5);
    const r = readStick(half, 0, R, DZ);
    expect(r.magnitude).toBeCloseTo(0.5);
    expect(r.right).toBeCloseTo(0.5);
  });

  it('keeps direction on diagonals and never exceeds full speed', () => {
    const r = readStick(100, 100, R, DZ); // down-right
    expect(r.right).toBeCloseTo(Math.SQRT1_2);
    expect(r.forward).toBeCloseTo(-Math.SQRT1_2);
    expect(Math.hypot(r.right, r.forward)).toBeCloseTo(1);
  });
});
