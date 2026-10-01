import {
  Block, DEFAULT_RATES, FALLING_LEVEL, LAVA_DECAY, PRIMED_DIRT, SOURCE_LEVEL, WATER_DECAY, WHEAT_RIPE,
  cell, cellLevel, cellType, isFluid, isSolid, type PlantRates,
} from '../constants';

// Plain-JS, cell-at-a-time version of the block-update rules. It is the readable
// spec for the WGSL rules (sim/rules.ts) and the tensor-op version (blockUpdate.ts),
// the oracle the tests compare them against, and what CpuStore runs: the world of the
// Node tests, and the fallback for a GPU that fails the startup check.

/** Cells that hold up a fluid resting on them, and that smother grass under them: solids and fluid sources. */
export const supports = (c: number) => isSolid(cellType(c)) || cellLevel(c) === SOURCE_LEVEL;
export const waterLevel = (c: number) => (cellType(c) === Block.Water ? cellLevel(c) : 0);
export const lavaLevel = (c: number) => (cellType(c) === Block.Lava ? cellLevel(c) : 0);
/** Air and flowing fluid are recomputed from their neighbours every tick. */
export const replaceable = (c: number) =>
  cellType(c) === Block.Air || (isFluid(cellType(c)) && cellLevel(c) < SOURCE_LEVEL);
export const isGrass = (c: number) => cellType(c) === Block.Grass;
export const isDirt = (c: number) => cellType(c) === Block.Dirt;
export const isAir = (c: number) => cellType(c) === Block.Air;
export const isWheat = (c: number) => cellType(c) === Block.Wheat;
export const wheatStage = (c: number) => (isWheat(c) ? cellLevel(c) : 0);
/** What wheat can grow on. */
export const isSoil = (c: number) => isDirt(c) || isGrass(c);

/**
 * Where grass can reach from, relative to the dirt it spreads onto: one block
 * either side, from one below to three above (Minecraft's spread box, seen from
 * the target rather than the source).
 */
export const GRASS_REACH = { dx: [-1, 1], dy: [-1, 3], dz: [-1, 1] } as const;

/** Constants of the PCG hash below, shared with the GPU kernel's copy of it. */
export const PCG = { mul: 747796405, inc: 2891336453, out: 277803737 } as const;

/** PCG-RXS-M-XS 32-bit integer hash. Integer maths, so the GPU computes exactly the same values. */
export function pcgHash(v: number): number {
  const state = (Math.imul(v, PCG.mul) + PCG.inc) >>> 0;
  const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, PCG.out) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

/**
 * The random number the fused GPU kernel uses for the cell at flat `index` of its input
 * on a tick with this `seed`: uniform in [0, 1), 24 bits, so exact in float32.
 */
export const cellRandom = (index: number, seed: number) => (pcgHash((index ^ seed) >>> 0) >>> 8) / 16777216;

/** Water near wheat that speeds it up: one block either side, at the plant's level or its soil's. */
export const WHEAT_WATER_REACH = { dx: [-1, 1], dy: [-1, 0], dz: [-1, 1] } as const;

/**
 * One tick over a [H, D, W] region (index (y * D + z) * W + x); cells outside read as air.
 *
 * `random` holds one uniform [0, 1) number per cell. Without it the plant rules (grass,
 * wheat) are skipped and those cells are left as they are, apart from fluids washing
 * wheat away (the simulation only omits it for regions without plants).
 */
export function blockUpdateReference(
  cells: ArrayLike<number>, H: number, D: number, W: number,
  random?: ArrayLike<number>, rates: PlantRates = DEFAULT_RATES,
): Int32Array {
  const at = (x: number, y: number, z: number) =>
    x < 0 || x >= W || z < 0 || z >= D || y >= H ? Block.Air : y < 0 ? Block.Stone : cells[(y * D + z) * W + x];
  const emit = (x: number, y: number, z: number, level: (c: number) => number, decay: number) => {
    if (x < 0 || x >= W || z < 0 || z >= D) return 0;
    return supports(at(x, y - 1, z)) ? Math.max(level(at(x, y, z)) - decay, 0) : 0;
  };
  const want = (x: number, y: number, z: number, level: (c: number) => number, decay: number) =>
    Math.max(
      emit(x - 1, y, z, level, decay), emit(x + 1, y, z, level, decay),
      emit(x, y, z - 1, level, decay), emit(x, y, z + 1, level, decay),
      level(at(x, y + 1, z)) > 0 ? FALLING_LEVEL : 0,
    );
  // Grass is alive unless a solid block or a fluid source sits on it.
  const aliveGrass = (x: number, y: number, z: number) =>
    x >= 0 && x < W && z >= 0 && z < D && y >= 0 && y < H && isGrass(at(x, y, z)) && !supports(at(x, y + 1, z));
  const grassInReach = (x: number, y: number, z: number) => {
    for (let dy = GRASS_REACH.dy[0]; dy <= GRASS_REACH.dy[1]; dy++)
      for (let dz = GRASS_REACH.dz[0]; dz <= GRASS_REACH.dz[1]; dz++)
        for (let dx = GRASS_REACH.dx[0]; dx <= GRASS_REACH.dx[1]; dx++)
          if (aliveGrass(x + dx, y + dy, z + dz)) return true;
    return false;
  };
  const wetNear = (x: number, y: number, z: number) => {
    for (let dy = WHEAT_WATER_REACH.dy[0]; dy <= WHEAT_WATER_REACH.dy[1]; dy++)
      for (let dz = WHEAT_WATER_REACH.dz[0]; dz <= WHEAT_WATER_REACH.dz[1]; dz++)
        for (let dx = WHEAT_WATER_REACH.dx[0]; dx <= WHEAT_WATER_REACH.dx[1]; dx++) {
          const xx = x + dx, yy = y + dy, zz = z + dz;
          if (xx >= 0 && xx < W && zz >= 0 && zz < D && yy >= 0 && yy < H && waterLevel(at(xx, yy, zz)) > 0) return true;
        }
    return false;
  };
  // The TF.js version compares in float32.
  const chance = Math.fround(rates.grassSpread), dry = Math.fround(rates.wheatGrow), wet = Math.fround(rates.wheatGrowWet);

  const out = new Int32Array(H * D * W);
  for (let y = 0; y < H; y++) {
    for (let z = 0; z < D; z++) {
      for (let x = 0; x < W; x++) {
        const i = (y * D + z) * W + x;
        const c = at(x, y, z);
        let next = c;
        if (replaceable(c)) {
          const ww = want(x, y, z, waterLevel, WATER_DECAY), wl = want(x, y, z, lavaLevel, LAVA_DECAY);
          if (ww > 0 && wl > 0) next = cell(Block.Stone);
          else if (ww > 0) next = cell(Block.Water, ww);
          else if (wl > 0) next = cell(Block.Lava, wl);
          else next = cell(Block.Air);
        }
        // Lava with water beside or above it solidifies (water below doesn't count, as in Minecraft).
        if (lavaLevel(c) > 0 && [at(x - 1, y, z), at(x + 1, y, z), at(x, y, z - 1), at(x, y, z + 1), at(x, y + 1, z)].some((n) => waterLevel(n) > 0)) {
          next = cell(Block.Stone);
        }
        if (isWheat(c)) {
          // Flowing fluid washes wheat away, as it would flow into air.
          const ww = want(x, y, z, waterLevel, WATER_DECAY), wl = want(x, y, z, lavaLevel, LAVA_DECAY);
          if (ww > 0 || wl > 0) {
            next = ww > 0 && wl > 0 ? cell(Block.Stone) : ww > 0 ? cell(Block.Water, ww) : cell(Block.Lava, wl);
          } else if (random) {
            // Wheat pops off without soil under it; otherwise it may grow a stage.
            const stage = wheatStage(c);
            const grows = stage < WHEAT_RIPE && random[i] < (wetNear(x, y, z) ? wet : dry);
            next = !isSoil(at(x, y - 1, z)) ? cell(Block.Air) : cell(Block.Wheat, stage + (grows ? 1 : 0));
          }
        } else if (random && isGrass(c)) {
          // Covered grass dies back to dirt.
          next = aliveGrass(x, y, z) ? cell(Block.Grass) : cell(Block.Dirt);
        } else if (random && isDirt(c)) {
          // Exposed dirt near living grass is primed, and sprouts with probability rates.grassSpread.
          const primed = rates.grassSpread > 0 && isAir(at(x, y + 1, z)) && grassInReach(x, y, z);
          next = primed && random[i] < chance ? cell(Block.Grass) : primed ? PRIMED_DIRT : cell(Block.Dirt);
        }
        out[i] = next;
      }
    }
  }
  return out;
}
