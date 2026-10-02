import { NOT_LOADED } from '../sim/store';
import { blocks } from '../player/physics';

/** Items lying in the world, waiting to be picked up (diamonds dropped by diamond ore). */
export interface Drop {
  pos: [number, number, number];
  vel: [number, number, number];
  /** Seconds since it dropped. */
  age: number;
}

const GRAVITY = 20;
/** It can't be picked up straight away, so you see it pop out. */
const PICKUP_DELAY = 0.4;
/** Within this distance it flies to the player; within PICKUP it's collected. */
const PULL_RANGE = 3;
const PICKUP = 0.9;
const PULL_SPEED = 9;
/** Items left lying this long disappear. */
const LIFETIME = 300;
/** Half the size of the drawn diamond. */
const SIZE = 0.16;
const COLOUR = [0.4, 1, 1];

export class Drops {
  readonly items: Drop[] = [];

  /** Drop an item from the block at (x, y, z): it pops out of the block's middle. */
  spawn(x: number, y: number, z: number, rand = Math.random): void {
    const a = rand() * Math.PI * 2;
    this.items.push({ pos: [x + 0.5, y + 0.5, z + 0.5], vel: [Math.cos(a) * 1.2, 4, Math.sin(a) * 1.2], age: 0 });
  }

  /**
   * Move the items (falling onto blocks, pulled to the player when close) and collect the
   * ones that reach the player's `centre`. Items where the blocks aren't known stay put.
   * Returns how many were collected.
   */
  update(dt: number, centre: readonly number[], typeAt: (x: number, y: number, z: number) => number): number {
    let collected = 0;
    for (let i = this.items.length - 1; i >= 0; i--) {
      const d = this.items[i];
      d.age += dt;
      const to = [centre[0] - d.pos[0], centre[1] - d.pos[1], centre[2] - d.pos[2]];
      const dist = Math.hypot(to[0], to[1], to[2]);
      if (d.age > PICKUP_DELAY && dist < PICKUP) {
        this.items.splice(i, 1);
        collected++;
        continue;
      }
      if (d.age > LIFETIME) {
        this.items.splice(i, 1);
        continue;
      }
      if (d.age > PICKUP_DELAY && dist < PULL_RANGE) {
        // Fly straight to the player, through anything.
        const s = Math.min(dist, PULL_SPEED * dt) / dist;
        for (let k = 0; k < 3; k++) d.pos[k] += to[k] * s;
        d.vel = [0, 0, 0];
        continue;
      }
      const cell = (p: readonly number[]) => typeAt(Math.floor(p[0]), Math.floor(p[1]), Math.floor(p[2]));
      if (cell(d.pos) === NOT_LOADED) continue;
      d.vel[1] -= GRAVITY * dt;
      for (let k = 0; k < 3; k++) {
        const next: [number, number, number] = [...d.pos];
        next[k] += d.vel[k] * dt;
        // Rest on (or bounce off) blocks: the item's bottom point mustn't enter one.
        const probe: [number, number, number] = [...next];
        probe[1] -= SIZE;
        if (blocks(cell(probe))) {
          if (k === 1 && d.vel[1] < 0) d.pos[1] = Math.floor(probe[1]) + 1 + SIZE;
          d.vel[k] = 0;
          if (k === 1) { d.vel[0] *= 0.5; d.vel[2] *= 0.5; }
        } else {
          d.pos[k] = next[k];
        }
      }
    }
    return collected;
  }

  /** Line segments (x, y, z, r, g, b pairs) drawing each item as a small spinning, bobbing diamond. */
  lines(time: number): number[] {
    const out: number[] = [];
    for (const d of this.items) {
      const a = time * 2.5 + d.pos[0], bob = Math.sin(time * 3 + d.pos[2]) * 0.05 + SIZE * 0.6;
      const c = [d.pos[0], d.pos[1] + bob, d.pos[2]];
      for (const s of [SIZE, SIZE * 0.6]) {
        const ring = [0, 1, 2, 3].map((k) => [c[0] + Math.cos(a + (k * Math.PI) / 2) * s, c[1], c[2] + Math.sin(a + (k * Math.PI) / 2) * s]);
        const top = [c[0], c[1] + s * 1.3, c[2]], bottom = [c[0], c[1] - s * 1.3, c[2]];
        for (let k = 0; k < 4; k++) {
          const p = ring[k], q = ring[(k + 1) % 4];
          out.push(...p, ...COLOUR, ...q, ...COLOUR, ...p, ...COLOUR, ...top, ...COLOUR, ...p, ...COLOUR, ...bottom, ...COLOUR);
        }
      }
    }
    return out;
  }
}
