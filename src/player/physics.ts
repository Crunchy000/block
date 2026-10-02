import { Block } from '../constants';
import { NOT_LOADED } from '../sim/store';

/** The player's box: Minecraft's size, eyes near the top. */
export const HALF_WIDTH = 0.3;
export const BODY_HEIGHT = 1.8;
export const EYE_HEIGHT = 1.62;

/** Blocks a second. */
export const WALK_SPEED = 4.5;
export const RUN_SPEED = 7;
const SWIM_FACTOR = 0.6;
/** Blocks a second², and the take-off speed of a jump (about 1.25 blocks high). */
export const GRAVITY = 28;
export const JUMP_SPEED = 8.4;
const MAX_FALL = 60;
const SWIM_UP = 3.5;
const MAX_SINK = 3;
/** How quickly the body's speed follows the controls (1/s): firm on the ground, loose in the air. */
const GRIP_GROUND = 14, GRIP_FLUID = 6, GRIP_AIR = 3;
/** Moves are split into steps no longer than this, so a fast fall can't skip through a block. */
const MAX_STEP = 0.4;
/** Gap kept from a face after stopping against it. */
const SKIN = 1e-3;

/** Blocks the body can't pass: solid blocks, and chunks not loaded yet (so nobody falls out of the world). */
export const blocks = (type: number): boolean =>
  type === Block.Stone || type === Block.Dirt || type === Block.Grass || type === NOT_LOADED;
const isFluid = (type: number): boolean => type === Block.Water || type === Block.Lava;

/** The block types around the player (read from the store with CellStore.readBox). */
export class Nearby {
  constructor(readonly min: readonly number[], readonly size: readonly number[], readonly types: Uint8Array) {}

  /** The block type at a cell; NOT_LOADED outside the box. */
  at(x: number, y: number, z: number): number {
    const lx = x - this.min[0], ly = y - this.min[1], lz = z - this.min[2];
    const [sx, sy, sz] = this.size;
    if (lx < 0 || ly < 0 || lz < 0 || lx >= sx || ly >= sy || lz >= sz) return NOT_LOADED;
    return this.types[(ly * sz + lz) * sx + lx];
  }

  /** Reflect an edit at once (the next read brings it anyway). */
  set(x: number, y: number, z: number, type: number): void {
    const lx = x - this.min[0], ly = y - this.min[1], lz = z - this.min[2];
    const [sx, sy, sz] = this.size;
    if (lx < 0 || ly < 0 || lz < 0 || lx >= sx || ly >= sy || lz >= sz) return;
    this.types[(ly * sz + lz) * sx + lx] = type;
  }
}

/** The cells a body standing at `feet` overlaps. */
function* cellsOf(feet: readonly number[]): Generator<[number, number, number]> {
  const e = 1e-6;
  for (let y = Math.floor(feet[1] + e); y <= Math.floor(feet[1] + BODY_HEIGHT - e); y++)
    for (let z = Math.floor(feet[2] - HALF_WIDTH + e); z <= Math.floor(feet[2] + HALF_WIDTH - e); z++)
      for (let x = Math.floor(feet[0] - HALF_WIDTH + e); x <= Math.floor(feet[0] + HALF_WIDTH - e); x++) yield [x, y, z];
}

/** Whether a body at `feet` overlaps a block that `type` says it can't be in. */
export function overlaps(feet: readonly number[], typeAt: (x: number, y: number, z: number) => number, test = blocks): boolean {
  for (const [x, y, z] of cellsOf(feet)) if (test(typeAt(x, y, z))) return true;
  return false;
}

/**
 * Move a body at `feet` by `delta`, one axis at a time (up/down first), in short steps,
 * stopping flush against blocks. Moves `feet` in place; returns which axes were stopped.
 */
export function moveBody(feet: number[], delta: readonly number[], typeAt: (x: number, y: number, z: number) => number): [boolean, boolean, boolean] {
  const hit: [boolean, boolean, boolean] = [false, false, false];
  // How far the body reaches from its feet, below and above, along each axis.
  const below = [HALF_WIDTH, 0, HALF_WIDTH], above = [HALF_WIDTH, BODY_HEIGHT, HALF_WIDTH];
  for (const axis of [1, 0, 2]) {
    let left = delta[axis];
    while (left !== 0 && !hit[axis]) {
      const d = Math.max(-MAX_STEP, Math.min(MAX_STEP, left));
      left -= d;
      feet[axis] += d;
      // Of the blocking cells now overlapped, the one nearest where the body came from.
      let stop: number | undefined;
      for (const cell of cellsOf(feet)) {
        if (!blocks(typeAt(cell[0], cell[1], cell[2]))) continue;
        const c = cell[axis];
        stop = stop === undefined ? c : d > 0 ? Math.min(stop, c) : Math.max(stop, c);
      }
      if (stop !== undefined) {
        feet[axis] = d > 0 ? stop - above[axis] - SKIN : stop + 1 + below[axis] + SKIN;
        hit[axis] = true;
      }
    }
  }
  return hit;
}

export interface WalkInput {
  /** Strafe right / forward, each in [-1, 1] (keys or the touch stick). */
  right: number;
  forward: number;
  jump: boolean;
  run: boolean;
}

/** Walking, running, jumping and swimming, with collisions. */
export class Body {
  velocity: [number, number, number] = [0, 0, 0];
  onGround = false;
  inFluid = false;

  /**
   * One step for a body standing at `feet` (moved in place), facing `yaw`. Waits (doesn't
   * move) while the blocks around it aren't known yet.
   */
  step(feet: number[], dt: number, yaw: number, input: WalkInput, typeAt: (x: number, y: number, z: number) => number): void {
    const v = this.velocity;
    // Stuck in a block (spawned in the ground, or a block appeared): climb out, a block at a time.
    if (overlaps(feet, typeAt, (t) => t === NOT_LOADED)) return; // not known yet
    if (overlaps(feet, typeAt)) {
      feet[1] = Math.floor(feet[1]) + 1;
      v[0] = v[1] = v[2] = 0;
      return;
    }
    this.inFluid = overlaps(feet, typeAt, isFluid);

    // Along the ground, toward where the controls point.
    const len = Math.hypot(input.right, input.forward);
    const speed = (input.run ? RUN_SPEED : WALK_SPEED) * (this.inFluid ? SWIM_FACTOR : 1) / Math.max(1, len);
    const sin = Math.sin(yaw), cos = Math.cos(yaw);
    const wish = [(-sin * input.forward + cos * input.right) * speed, (-cos * input.forward - sin * input.right) * speed];
    const grip = 1 - Math.exp(-(this.onGround ? GRIP_GROUND : this.inFluid ? GRIP_FLUID : GRIP_AIR) * dt);
    v[0] += (wish[0] - v[0]) * grip;
    v[2] += (wish[1] - v[2]) * grip;

    if (this.inFluid) {
      v[1] = input.jump ? SWIM_UP : Math.max(v[1] - GRAVITY * 0.2 * dt, -MAX_SINK);
    } else {
      if (input.jump && this.onGround) v[1] = JUMP_SPEED;
      v[1] = Math.max(v[1] - GRAVITY * dt, -MAX_FALL);
    }

    const falling = v[1] < 0;
    const hit = moveBody(feet, [v[0] * dt, v[1] * dt, v[2] * dt], typeAt);
    this.onGround = hit[1] && falling;
    if (hit[1]) v[1] = 0;
    if (hit[0]) v[0] = 0;
    if (hit[2]) v[2] = 0;
    // Swimming up against a bank: hop out onto it.
    if (this.inFluid && input.jump && (hit[0] || hit[2])) v[1] = JUMP_SPEED * 0.7;
  }
}
