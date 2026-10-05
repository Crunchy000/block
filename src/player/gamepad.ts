/**
 * Gamepads (the browser's Gamepad API, standard layout: Xbox / PlayStation style). Read once
 * a frame: the sticks (with a dead zone), the triggers, and which buttons went down since the
 * last read. Controls (player/controls.ts) turns that into play:
 *
 *   left stick: walk (click it in to run) · right stick: look · right trigger: dig (hold)
 *   left trigger: place · A: jump / swim / fly up · B: fly down · Y: fly on / off
 *   LB / RB or d-pad left / right: hotbar block · any button on the start screen: play · Start: back
 */
export const Button = {
  A: 0, B: 1, X: 2, Y: 3, LB: 4, RB: 5, LT: 6, RT: 7, Back: 8, Start: 9, LeftStick: 10, RightStick: 11,
  Up: 12, Down: 13, Left: 14, Right: 15,
} as const;

/** Stick movement smaller than this is ignored (worn sticks don't rest exactly at 0). */
const DEAD_ZONE = 0.15;
/** A trigger counts as pressed past this. */
const TRIGGER = 0.3;

export interface PadState {
  /** Left stick: strafe right / forward, each in [-1, 1]. */
  move: { right: number; forward: number };
  /** Right stick: right / down, each in [-1, 1], eased (fine aim near the centre). */
  look: { x: number; y: number };
  /** Held now. */
  held: Set<number>;
  /** Went down since the last read. */
  pressed: Set<number>;
}

/** A stick's two axes with a radial dead zone, rescaled so just past the dead zone is just above 0. */
export function stick(x: number, y: number): [number, number] {
  const len = Math.hypot(x, y);
  if (len < DEAD_ZONE) return [0, 0];
  const scaled = Math.min(1, (len - DEAD_ZONE) / (1 - DEAD_ZONE));
  return [(x / len) * scaled, (y / len) * scaled];
}

/** Reads the first connected gamepad each frame, remembering its buttons to tell new presses. */
export class GamepadInput {
  private was: boolean[] = [];
  /** Whether a gamepad has been seen (shown in the help). */
  seen = false;

  /** The pad's state, or undefined if none is connected. */
  read(pads: ReadonlyArray<Gamepad | null>): PadState | undefined {
    const pad = pads.find((p) => p?.connected) ?? undefined;
    if (!pad) {
      this.was = [];
      return undefined;
    }
    this.seen = true;
    const down = pad.buttons.map((b, i) => (i === Button.LT || i === Button.RT ? b.value > TRIGGER : b.pressed));
    const held = new Set<number>(), pressed = new Set<number>();
    down.forEach((d, i) => {
      if (!d) return;
      held.add(i);
      if (!this.was[i]) pressed.add(i);
    });
    this.was = down;
    const a = pad.axes;
    const [mx, my] = stick(a[0] ?? 0, a[1] ?? 0);
    const [lx, ly] = stick(a[2] ?? 0, a[3] ?? 0);
    return {
      move: { right: mx, forward: -my },
      look: { x: lx * Math.abs(lx), y: ly * Math.abs(ly) },
      held,
      pressed,
    };
  }
}
