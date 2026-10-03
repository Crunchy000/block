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
 * The animals ("Cube Pets" by Kenney, CC0; models made by scripts/convert-models.mjs, as tall
 * as these boxes). Boxes a little narrower than the models (boxes don't turn with the animal).
 */
export const SPECIES: readonly Species[] = [
  { name: 'cow', size: { halfWidth: 0.5, height: 1.3 }, walk: 1, run: 3, weight: 3 },
  { name: 'pig', size: { halfWidth: 0.33, height: 0.85 }, walk: 1.2, run: 3.5, weight: 3 },
  { name: 'chicken', size: { halfWidth: 0.24, height: 0.6 }, walk: 1, run: 3, weight: 3 },
  { name: 'rabbit', size: { halfWidth: 0.18, height: 0.6 }, walk: 1.6, run: 5, weight: 2, hops: true },
  { name: 'cat', size: { halfWidth: 0.26, height: 0.7 }, walk: 1.6, run: 5, weight: 1 },
  { name: 'dog', size: { halfWidth: 0.3, height: 0.8 }, walk: 1.5, run: 5, weight: 1 },
  { name: 'deer', size: { halfWidth: 0.5, height: 1.5 }, walk: 1.4, run: 6, weight: 1.5 },
  { name: 'fox', size: { halfWidth: 0.28, height: 0.75 }, walk: 1.5, run: 5.5, weight: 1 },
];

/** The animation clips an animal plays: standing about, walking, running off, grazing. */
export type AnimalClip = 'idle' | 'walk' | 'run' | 'eat';
/** For instances(): the frames of a species' model at a time into a clip (render/mobModel.ts poseFrames). */
export type PoseFrames = (species: string, clip: AnimalClip, time: number) => readonly number[];

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
  /** Standing still grazing (rather than looking about). */
  eating: boolean;
  /** The clip it's playing, and how far into it (seconds). */
  clip: AnimalClip;
  clipTime: number;
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
        walking: false, until: this.rand() * 2, fleeing: 0, eating: false, clip: 'idle', clipTime: 0,
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
      const input: WalkInput = { right: 0, forward: a.walking ? 1 : 0, jump: false, run: a.fleeing > 0 };
      // Hop up a step it walks into (rabbits hop all the time).
      const reach = a.species.size.halfWidth + 0.3;
      const ahead = [a.feet[0] - Math.sin(a.yaw) * reach, a.feet[1], a.feet[2] - Math.cos(a.yaw) * reach];
      const blocked = blocks(typeAt(Math.floor(ahead[0]), Math.floor(ahead[1]), Math.floor(ahead[2])));
      if (a.walking && a.body.onGround && (blocked || a.species.hops)) input.jump = true;
      a.body.step(a.feet, dt, a.yaw, input, typeAt);
      if (a.species.hops && input.jump && a.body.velocity[1] > 0) a.body.velocity[1] = Math.min(a.body.velocity[1], blocked ? 8.4 : 5);
      const clip: AnimalClip = a.fleeing > 0 ? 'run' : a.walking ? 'walk' : a.eating ? 'eat' : 'idle';
      if (clip !== a.clip) { a.clip = clip; a.clipTime = 0; }
      a.clipTime += dt;
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
    a.eating = !a.walking && this.rand() < 0.4;
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

  /**
   * Per species with any about, for drawing: per animal x, y, z, yaw, then where it is in its
   * clip (the 3 numbers `frames` gives for it: two frames of its model's poses and a blend), 0.
   */
  instances(frames: PoseFrames): Map<string, Float32Array<ArrayBuffer>> {
    const groups = new Map<string, Animal[]>();
    for (const a of this.animals) groups.set(a.species.name, [...(groups.get(a.species.name) ?? []), a]);
    const out = new Map<string, Float32Array<ArrayBuffer>>();
    for (const [name, list] of groups) {
      const data = new Float32Array(list.length * 8);
      list.forEach((a, i) => data.set([a.feet[0], a.feet[1], a.feet[2], a.yaw, ...frames(name, a.clip, a.clipTime).slice(0, 3), 0], i * 8));
      out.set(name, data);
    }
    return out;
  }
}
