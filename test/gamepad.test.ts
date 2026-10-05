import { describe, expect, it } from 'vitest';
import { Button, GamepadInput, stick } from '../src/player/gamepad';

/** A fake standard-layout pad: 17 buttons, 4 axes. */
function pad(axes: number[] = [0, 0, 0, 0], down: number[] = [], triggers: { lt?: number; rt?: number } = {}): Gamepad {
  const buttons = Array.from({ length: 17 }, (_, i) => {
    const value = i === Button.LT ? triggers.lt ?? 0 : i === Button.RT ? triggers.rt ?? 0 : down.includes(i) ? 1 : 0;
    return { pressed: value > 0.5, touched: value > 0, value };
  });
  return { connected: true, axes, buttons, id: 'test', index: 0, mapping: 'standard', timestamp: 0 } as unknown as Gamepad;
}

describe('gamepad', () => {
  it('ignores sticks resting near the centre, and rescales the rest', () => {
    expect(stick(0.1, -0.05)).toEqual([0, 0]);
    const [x, y] = stick(1, 0);
    expect([x, y]).toEqual([1, 0]);
    const [hx] = stick(0.575, 0); // halfway between the dead zone and the edge
    expect(hx).toBeCloseTo(0.5, 5);
  });

  it('reads the left stick as walking (up is forward) and eases the right stick for aiming', () => {
    const input = new GamepadInput();
    const s = input.read([null, pad([0.5, -1, 1, 0.5])])!;
    expect(s.move.forward).toBeCloseTo(1 / Math.hypot(0.5, 1), 5); // past full: clamped to length 1
    expect(s.move.right).toBeGreaterThan(0.3);
    expect(s.look.x).toBeCloseTo((1 / Math.hypot(1, 0.5)) ** 2, 5); // eased: squared
    expect(s.look.y).toBeGreaterThan(0);
    expect(s.look.y).toBeLessThan(0.3); // a half push turns less than half as fast
  });

  it('tells new presses from held buttons, and triggers count past a threshold', () => {
    const input = new GamepadInput();
    let s = input.read([pad(undefined, [Button.A], { rt: 0.2 })])!;
    expect([...s.pressed]).toEqual([Button.A]);
    expect(s.held.has(Button.RT)).toBe(false);
    s = input.read([pad(undefined, [Button.A], { rt: 0.9 })])!;
    expect(s.pressed.has(Button.A)).toBe(false); // still held, not pressed again
    expect(s.held.has(Button.A)).toBe(true);
    expect([...s.pressed]).toEqual([Button.RT]);
    expect(input.read([])).toBeUndefined(); // unplugged
  });
});
