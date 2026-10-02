import { Block } from '../constants';
import { Body, blocks, type BodySize, type WalkInput } from '../player/physics';
import { NOT_LOADED } from '../sim/store';

/** A kind of animal: its model (public/models/<name>.bin), box, speeds and how often it spawns. */
export interface Species {
  name: string;
  size: BodySize;
  /** Blocks a second, wandering and running off. */
  walk: number;
  run: number;
  /** Relative chance of being the one spawned. */
  weight: number;
  /** Hops as it goes (rabbits). */
  hops?: boolean;
}

/**
 * The farm animals ("Cube Farm Animals" by ezgi bakim, CC-BY-4.0). Boxes a little narrower
 * than the models (boxes don't turn with the animal), as tall.
 */
export const SPECIES: readonly Species[] = [
  { name: 'cow', size: { halfWidth: 0.45, height: 1.4 }, walk: 1, run: 3, weight: 3 },
  { name: 'sheep', size: { halfWidth: 0.45, height: 1.3 }, walk: 1, run: 3, weight: 3 },
  { name: 'pig', size: { halfWidth: 0.35, height: 0.9 }, walk: 1.2, run: 3.5, weight: 3 },
  { name: 'chicken', size: { halfWidth: 0.28, height: 0.7 }, walk: 1, run: 3, weight: 3 },
  { name: 'horse', size: { halfWidth: 0.45, height: 1.6 }, walk: 1.6, run: 6, weight: 1.5 },
  { name: 'rabbit', size: { halfWidth: 0.16, height: 0.5 }, walk: 1.6, run: 5, weight: 2, hops: true },
  { name: 'cat', size: { halfWidth: 0.24, height: 0.7 }, walk: 1.6, run: 5, weight: 1 },
  { name: 'mouse', size: { halfWidth: 0.18, height: 0.45 }, walk: 1.8, run: 5, weight: 1 },
];

/** Up to this many animals around the player, spawned on grass this far away, despawned beyond DESPAWN. */
export const MAX_ANIMALS = 10;
const SPAWN_MIN = 8, SPAWN_MAX = 20, DESPAWN = 40;
/** How often (seconds) to try to spawn one when there are fewer than MAX_ANIMALS. */
const SPAWN_EVERY = 1;
const FLEE_SECONDS = 3;

export interface Animal {
  species: Species;
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

/** Farm animals wandering on the grass around the player. */
export class Animals {
  readonly animals: Animal[] = [];
  private spawnIn = 0;

  constructor(private readonly rand = Math.random, readonly species: readonly Species[] = SPECIES) {}

  /** Pick a species, by weight. */
  private pick(): Species {
    let r = this.rand() * this.species.reduce((a, s) => a + s.weight, 0);
    for (const s of this.species) if ((r -= s.weight) < 0) return s;
    return this.species[this.species.length - 1];
  }

  /** Try to put an animal on a grass block with room above it, SPAWN_MIN..SPAWN_MAX from `centre`. */
  trySpawn(centre: readonly number[], typeAt: TypeAt, species = this.pick()): Animal | undefined {
    const a = this.rand() * Math.PI * 2, d = SPAWN_MIN + this.rand() * (SPAWN_MAX - SPAWN_MIN);
    const x = Math.floor(centre[0] + Math.cos(a) * d), z = Math.floor(centre[2] + Math.sin(a) * d);
    const room = Math.ceil(species.size.height);
    // Look down the column for grass with room above.
    for (let y = Math.floor(centre[1]) + 12; y > Math.floor(centre[1]) - 16; y--) {
      const t = typeAt(x, y, z);
      if (t === NOT_LOADED) continue;
      if (!blocks(t)) continue;
      if (t !== Block.Grass) return undefined;
      for (let k = 1; k <= room; k++) if (typeAt(x, y + k, z) !== Block.Air) return undefined;
      const yaw = this.rand() * Math.PI * 2;
      const animal: Animal = {
        species, feet: [x + 0.5, y + 1, z + 0.5], yaw, heading: yaw, body: new Body(species.size, species.walk, species.run),
        walking: false, until: this.rand() * 2, fleeing: 0, stride: 0,
      };
      this.animals.push(animal);
      return animal;
    }
    return undefined;
  }

  /** Knock an animal back (away from `from`) and send it running. */
  hit(animal: Animal, from: readonly number[]): void {
    const dx = animal.feet[0] - from[0], dz = animal.feet[2] - from[2], l = Math.hypot(dx, dz) || 1;
    animal.body.velocity = [(dx / l) * 6, 5, (dz / l) * 6];
    animal.body.onGround = false;
    animal.fleeing = FLEE_SECONDS;
    animal.heading = Math.atan2(-dx, -dz); // away from the player
    animal.walking = true;
  }

  /** Spawn, despawn and move the animals around the player at `centre`. */
  update(dt: number, centre: readonly number[], typeAt: TypeAt): void {
    this.spawnIn -= dt;
    if (this.spawnIn <= 0 && this.animals.length < MAX_ANIMALS) {
      this.spawnIn = SPAWN_EVERY;
      this.trySpawn(centre, typeAt);
    }
    for (let i = this.animals.length - 1; i >= 0; i--) {
      const a = this.animals[i];
      if (Math.hypot(a.feet[0] - centre[0], a.feet[2] - centre[2]) > DESPAWN) {
        this.animals.splice(i, 1);
        continue;
      }
      this.think(a, dt);
      // Turn toward the heading, a few radians a second.
      const turn = Math.atan2(Math.sin(a.heading - a.yaw), Math.cos(a.heading - a.yaw));
      a.yaw += Math.max(-4 * dt, Math.min(4 * dt, turn));
      const before = [...a.feet];
      const input: WalkInput = { right: 0, forward: a.walking ? 1 : 0, jump: false, run: a.fleeing > 0 };
      // Hop up a step it walks into (rabbits hop all the time).
      const reach = a.species.size.halfWidth + 0.3;
      const ahead = [a.feet[0] - Math.sin(a.yaw) * reach, a.feet[1], a.feet[2] - Math.cos(a.yaw) * reach];
      const blocked = blocks(typeAt(Math.floor(ahead[0]), Math.floor(ahead[1]), Math.floor(ahead[2])));
      if (a.walking && a.body.onGround && (blocked || a.species.hops)) input.jump = true;
      a.body.step(a.feet, dt, a.yaw, input, typeAt);
      if (a.species.hops && input.jump && a.body.velocity[1] > 0) a.body.velocity[1] = Math.min(a.body.velocity[1], blocked ? 8.4 : 5);
      a.stride += Math.hypot(a.feet[0] - before[0], a.feet[2] - before[2]);
    }
  }

  /** Wander: walk a while in some direction, stand a while, turn. Running away when hit. */
  private think(a: Animal, dt: number): void {
    if (a.fleeing > 0) {
      a.fleeing -= dt;
      if (a.fleeing <= 0) a.until = 0;
      return;
    }
    a.until -= dt;
    if (a.until > 0) return;
    a.walking = this.rand() < 0.6;
    a.until = a.walking ? 1.5 + this.rand() * 3 : 1 + this.rand() * 4;
    a.heading = a.yaw + (this.rand() - 0.5) * Math.PI * 1.5;
  }

  /** The first animal along a ray (origin, unit dir) within maxDist, and how far: hitting animals before blocks. */
  raycast(origin: readonly number[], dir: readonly number[], maxDist: number): { animal: Animal; dist: number } | undefined {
    let best: { animal: Animal; dist: number } | undefined;
    for (const animal of this.animals) {
      const { halfWidth: w, height } = animal.species.size, f = animal.feet;
      const min = [f[0] - w, f[1], f[2] - w], max = [f[0] + w, f[1] + height, f[2] + w];
      let t0 = 0, t1 = maxDist;
      for (let k = 0; k < 3 && t0 <= t1; k++) {
        if (dir[k] === 0) {
          if (origin[k] < min[k] || origin[k] > max[k]) t0 = Infinity;
          continue;
        }
        const p = (min[k] - origin[k]) / dir[k], q = (max[k] - origin[k]) / dir[k];
        t0 = Math.max(t0, Math.min(p, q));
        t1 = Math.min(t1, Math.max(p, q));
      }
      if (t0 <= t1 && (!best || t0 < best.dist)) best = { animal, dist: t0 };
    }
    return best;
  }

  /** Per species with any about: per animal x, y, z, yaw, then a waddle (roll) and a bob, for drawing. */
  instances(): Map<string, Float32Array<ArrayBuffer>> {
    const groups = new Map<string, Animal[]>();
    for (const a of this.animals) groups.set(a.species.name, [...(groups.get(a.species.name) ?? []), a]);
    const out = new Map<string, Float32Array<ArrayBuffer>>();
    for (const [name, list] of groups) {
      const data = new Float32Array(list.length * 8);
      list.forEach((a, i) => {
        const moving = a.walking ? 1 : 0, h = a.species.size.height;
        data.set([a.feet[0], a.feet[1], a.feet[2], a.yaw, Math.sin(a.stride * 5) * 0.06 * moving, Math.abs(Math.sin(a.stride * 5)) * 0.05 * h * moving, h / 2, 0], i * 8);
      });
      out.set(name, data);
    }
    return out;
  }
}
