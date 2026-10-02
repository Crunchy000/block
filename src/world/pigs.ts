import { Block } from '../constants';
import { Body, blocks, type BodySize, type WalkInput } from '../player/physics';
import { NOT_LOADED } from '../sim/store';

/** A pig's box: Minecraft's pig is 0.9 x 0.9 blocks. */
export const PIG_SIZE: BodySize = { halfWidth: 0.45, height: 0.9 };
const WALK = 1.2, FLEE = 3.5;
/** Up to this many pigs around the player, spawned on grass this far away, despawned beyond DESPAWN. */
export const MAX_PIGS = 6;
const SPAWN_MIN = 8, SPAWN_MAX = 20, DESPAWN = 40;
/** How often (seconds) to try to spawn one when there are fewer than MAX_PIGS. */
const SPAWN_EVERY = 1.5;
const FLEE_SECONDS = 3;

export interface Pig {
  feet: number[];
  /** Facing (as the camera's yaw: 0 looks toward -z). */
  yaw: number;
  /** Which way it's turning toward. */
  heading: number;
  body: Body;
  walking: boolean;
  /** Seconds until it picks something else to do. */
  until: number;
  /** Seconds left running away after being hit. */
  fleeing: number;
  /** For the waddle: how far it has walked. */
  stride: number;
}

type TypeAt = (x: number, y: number, z: number) => number;

/** Pigs wandering on the grass around the player. */
export class Pigs {
  readonly pigs: Pig[] = [];
  private spawnIn = 0;

  constructor(private readonly rand = Math.random) {}

  /** Try to put a pig on a grass block with room above it, SPAWN_MIN..SPAWN_MAX from `centre`. */
  trySpawn(centre: readonly number[], typeAt: TypeAt): Pig | undefined {
    const a = this.rand() * Math.PI * 2, d = SPAWN_MIN + this.rand() * (SPAWN_MAX - SPAWN_MIN);
    const x = Math.floor(centre[0] + Math.cos(a) * d), z = Math.floor(centre[2] + Math.sin(a) * d);
    // Look down the column for grass with two cells of air above.
    for (let y = Math.floor(centre[1]) + 12; y > Math.floor(centre[1]) - 16; y--) {
      const t = typeAt(x, y, z);
      if (t === NOT_LOADED) continue;
      if (blocks(t)) {
        if (t !== Block.Grass || typeAt(x, y + 1, z) !== Block.Air || typeAt(x, y + 2, z) !== Block.Air) return undefined;
        const yaw = this.rand() * Math.PI * 2;
        const pig: Pig = {
          feet: [x + 0.5, y + 1, z + 0.5], yaw, heading: yaw, body: new Body(PIG_SIZE, WALK, FLEE),
          walking: false, until: this.rand() * 2, fleeing: 0, stride: 0,
        };
        this.pigs.push(pig);
        return pig;
      }
    }
    return undefined;
  }

  /** Knock a pig back (away from `from`) and send it running. */
  hit(pig: Pig, from: readonly number[]): void {
    const dx = pig.feet[0] - from[0], dz = pig.feet[2] - from[2], l = Math.hypot(dx, dz) || 1;
    pig.body.velocity = [(dx / l) * 6, 5, (dz / l) * 6];
    pig.body.onGround = false;
    pig.fleeing = FLEE_SECONDS;
    pig.heading = Math.atan2(-dx, -dz); // away from the player
    pig.walking = true;
  }

  /** Spawn, despawn and move the pigs around the player at `centre`. */
  update(dt: number, centre: readonly number[], typeAt: TypeAt): void {
    this.spawnIn -= dt;
    if (this.spawnIn <= 0 && this.pigs.length < MAX_PIGS) {
      this.spawnIn = SPAWN_EVERY;
      this.trySpawn(centre, typeAt);
    }
    for (let i = this.pigs.length - 1; i >= 0; i--) {
      const pig = this.pigs[i];
      if (Math.hypot(pig.feet[0] - centre[0], pig.feet[2] - centre[2]) > DESPAWN) {
        this.pigs.splice(i, 1);
        continue;
      }
      this.think(pig, dt);
      // Turn toward the heading, a few radians a second.
      const turn = Math.atan2(Math.sin(pig.heading - pig.yaw), Math.cos(pig.heading - pig.yaw));
      pig.yaw += Math.max(-4 * dt, Math.min(4 * dt, turn));
      const before = [...pig.feet];
      const input: WalkInput = { right: 0, forward: pig.walking ? 1 : 0, jump: false, run: pig.fleeing > 0 };
      // Hop up a step it walks into.
      const ahead = [pig.feet[0] - Math.sin(pig.yaw) * 0.7, pig.feet[1], pig.feet[2] - Math.cos(pig.yaw) * 0.7];
      if (pig.walking && pig.body.onGround && blocks(typeAt(Math.floor(ahead[0]), Math.floor(ahead[1]), Math.floor(ahead[2])))) input.jump = true;
      pig.body.step(pig.feet, dt, pig.yaw, input, typeAt);
      pig.stride += Math.hypot(pig.feet[0] - before[0], pig.feet[2] - before[2]);
    }
  }

  /** Wander: walk a while in some direction, stand a while, turn. Running away when hit. */
  private think(pig: Pig, dt: number): void {
    if (pig.fleeing > 0) {
      pig.fleeing -= dt;
      if (pig.fleeing <= 0) pig.until = 0;
      return;
    }
    pig.until -= dt;
    if (pig.until > 0) return;
    pig.walking = this.rand() < 0.6;
    pig.until = pig.walking ? 1.5 + this.rand() * 3 : 1 + this.rand() * 4;
    pig.heading = pig.yaw + (this.rand() - 0.5) * Math.PI * 1.5;
  }

  /**
   * The first pig along a ray (origin, unit dir) within maxDist, and how far: picking pigs
   * to hit before blocks.
   */
  raycast(origin: readonly number[], dir: readonly number[], maxDist: number): { pig: Pig; dist: number } | undefined {
    let best: { pig: Pig; dist: number } | undefined;
    for (const pig of this.pigs) {
      const w = PIG_SIZE.halfWidth;
      const min = [pig.feet[0] - w, pig.feet[1], pig.feet[2] - w], max = [pig.feet[0] + w, pig.feet[1] + PIG_SIZE.height, pig.feet[2] + w];
      let t0 = 0, t1 = maxDist;
      for (let k = 0; k < 3 && t0 <= t1; k++) {
        if (dir[k] === 0) {
          if (origin[k] < min[k] || origin[k] > max[k]) t0 = Infinity;
          continue;
        }
        const a = (min[k] - origin[k]) / dir[k], b = (max[k] - origin[k]) / dir[k];
        t0 = Math.max(t0, Math.min(a, b));
        t1 = Math.min(t1, Math.max(a, b));
      }
      if (t0 <= t1 && (!best || t0 < best.dist)) best = { pig, dist: t0 };
    }
    return best;
  }

  /** Per pig: x, y, z, yaw, then a waddle (roll) and a bob for drawing it. */
  instances(): Float32Array<ArrayBuffer> {
    const out = new Float32Array(this.pigs.length * 8);
    this.pigs.forEach((p, i) => {
      const moving = p.walking ? 1 : 0;
      out.set([p.feet[0], p.feet[1], p.feet[2], p.yaw, Math.sin(p.stride * 5) * 0.06 * moving, Math.abs(Math.sin(p.stride * 5)) * 0.05 * moving, 0, 0], i * 8);
    });
    return out;
  }
}
