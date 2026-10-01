import {
  Block, FALLING_LEVEL, LAVA_DECAY, LEVEL_MUL, PRIMED_DIRT, SOURCE_LEVEL, WATER_DECAY, WHEAT_RIPE,
} from '../constants';
import { GRASS_REACH, PCG, WHEAT_WATER_REACH } from '../tf/blockUpdateReference';

/**
 * Every block-update rule as WGSL, for the compute shaders that run them: the
 * GPU-resident world (gpuStore.ts) and the TF.js custom kernel (blockUpdateKernel.ts).
 * Same rules as blockUpdateReference.ts (the spec); the GPU tests check they agree.
 *
 * The shader that includes this provides:
 *   fn cellAt(y: i32, z: i32, x: i32) -> i32
 *       the cell at a position (any cell the rules for (y, z, x) read is at most one
 *       block away sideways); above the top it reads as air, below y = 0 as bedrock
 *       (stone), and past the sides of a bounded region as air
 *   uniforms.seed: u32, uniforms.grassSpread / wheatGrow / wheatGrowWet: f32,
 *   uniforms.plants: i32 (0 skips the grass and wheat rules, apart from fluids washing wheat away)
 *
 * and calls nextCell(y, z, x, randomIndex) for the cell's next state. randomIndex picks
 * the cell's random number for the tick (see cellRandom in the spec).
 */
export const RULES_WGSL = /* wgsl */ `
const AIR: i32 = ${Block.Air};
const STONE: i32 = ${Block.Stone};
const DIRT: i32 = ${Block.Dirt};
const WATER: i32 = ${Block.Water};
const LAVA: i32 = ${Block.Lava};
const GRASS: i32 = ${Block.Grass};
const WHEAT: i32 = ${Block.Wheat};
const LEVEL_MUL: i32 = ${LEVEL_MUL};
const SOURCE: i32 = ${SOURCE_LEVEL};
const FALLING: i32 = ${FALLING_LEVEL};
const RIPE: i32 = ${WHEAT_RIPE};
const PRIMED: i32 = ${PRIMED_DIRT};

fn blockOf(c: i32) -> i32 { return c & 7; }
fn levelOf(c: i32) -> i32 { return c >> 3u; }
fn fluidLevel(c: i32, fluid: i32) -> i32 { return select(0, levelOf(c), blockOf(c) == fluid); }
fn isSolidCell(c: i32) -> bool {
  let t = blockOf(c);
  return t == STONE || t == DIRT || t == GRASS;
}
// Holds up fluid resting on it, and smothers grass under it.
fn supportsCell(c: i32) -> bool { return isSolidCell(c) || levelOf(c) == SOURCE; }

// What a fluid cell at (y, z, x) passes sideways: level - decay, if it rests on something.
// (Past the sides of a region the cell below reads as air, which holds nothing up.)
fn emitFrom(y: i32, z: i32, x: i32, fluid: i32, decay: i32) -> i32 {
  if (!supportsCell(cellAt(y - 1, z, x))) { return 0; }
  return max(fluidLevel(cellAt(y, z, x), fluid) - decay, 0);
}
// The level a fluid wants to put in (y, z, x): from the sides, or falling from above.
fn wantOf(y: i32, z: i32, x: i32, fluid: i32, decay: i32) -> i32 {
  var w = max(max(emitFrom(y, z, x - 1, fluid, decay), emitFrom(y, z, x + 1, fluid, decay)),
              max(emitFrom(y, z - 1, x, fluid, decay), emitFrom(y, z + 1, x, fluid, decay)));
  if (fluidLevel(cellAt(y + 1, z, x), fluid) > 0) { w = max(w, FALLING); }
  return w;
}
// A cell fluid flows into: stone where water and lava meet.
fn flowed(ww: i32, wl: i32) -> i32 {
  if (ww > 0 && wl > 0) { return STONE; }
  if (ww > 0) { return WATER + LEVEL_MUL * ww; }
  if (wl > 0) { return LAVA + LEVEL_MUL * wl; }
  return AIR;
}
// Grass that nothing smothers. (Positions outside the world read as air or stone: no grass.)
fn aliveGrassAt(y: i32, z: i32, x: i32) -> bool {
  return blockOf(cellAt(y, z, x)) == GRASS && !supportsCell(cellAt(y + 1, z, x));
}
fn grassInReach(y: i32, z: i32, x: i32) -> bool {
  for (var dy = ${GRASS_REACH.dy[0]}; dy <= ${GRASS_REACH.dy[1]}; dy++) {
    for (var dz = ${GRASS_REACH.dz[0]}; dz <= ${GRASS_REACH.dz[1]}; dz++) {
      for (var dx = ${GRASS_REACH.dx[0]}; dx <= ${GRASS_REACH.dx[1]}; dx++) {
        if (aliveGrassAt(y + dy, z + dz, x + dx)) { return true; }
      }
    }
  }
  return false;
}
fn wetNear(y: i32, z: i32, x: i32) -> bool {
  for (var dy = ${WHEAT_WATER_REACH.dy[0]}; dy <= ${WHEAT_WATER_REACH.dy[1]}; dy++) {
    for (var dz = ${WHEAT_WATER_REACH.dz[0]}; dz <= ${WHEAT_WATER_REACH.dz[1]}; dz++) {
      for (var dx = ${WHEAT_WATER_REACH.dx[0]}; dx <= ${WHEAT_WATER_REACH.dx[1]}; dx++) {
        if (fluidLevel(cellAt(y + dy, z + dz, x + dx), WATER) > 0) { return true; }
      }
    }
  }
  return false;
}
fn pcgHash(v: u32) -> u32 {
  let state = v * ${PCG.mul}u + ${PCG.inc}u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * ${PCG.out}u;
  return (word >> 22u) ^ word;
}
fn cellRandom(index: u32) -> f32 {
  return f32(pcgHash(index ^ uniforms.seed) >> 8u) / 16777216.0;
}

fn nextCell(y: i32, z: i32, x: i32, randomIndex: u32) -> i32 {
  let c = cellAt(y, z, x);
  let t = blockOf(c);
  var next = c;

  // Air and flowing fluid are recomputed from their neighbours.
  if (t == AIR || ((t == WATER || t == LAVA) && levelOf(c) < SOURCE)) {
    next = flowed(wantOf(y, z, x, WATER, ${WATER_DECAY}), wantOf(y, z, x, LAVA, ${LAVA_DECAY}));
  }
  // Lava with water beside or above it turns to stone.
  if (t == LAVA && levelOf(c) > 0 && (
      fluidLevel(cellAt(y, z, x - 1), WATER) > 0 || fluidLevel(cellAt(y, z, x + 1), WATER) > 0 ||
      fluidLevel(cellAt(y, z - 1, x), WATER) > 0 || fluidLevel(cellAt(y, z + 1, x), WATER) > 0 ||
      fluidLevel(cellAt(y + 1, z, x), WATER) > 0)) {
    next = STONE;
  }

  if (t == WHEAT) {
    // Flowing fluid washes wheat away; otherwise it pops off without soil, or may grow.
    let ww = wantOf(y, z, x, WATER, ${WATER_DECAY});
    let wl = wantOf(y, z, x, LAVA, ${LAVA_DECAY});
    if (ww > 0 || wl > 0) {
      next = flowed(ww, wl);
    } else if (uniforms.plants != 0) {
      let soil = blockOf(cellAt(y - 1, z, x));
      if (soil != DIRT && soil != GRASS) {
        next = AIR;
      } else {
        let stage = levelOf(c);
        let chance = select(uniforms.wheatGrow, uniforms.wheatGrowWet, wetNear(y, z, x));
        if (stage < RIPE && cellRandom(randomIndex) < chance) { next = WHEAT + LEVEL_MUL * (stage + 1); }
      }
    }
  } else if (uniforms.plants != 0 && t == GRASS) {
    // Covered grass dies back to dirt.
    next = select(DIRT, GRASS, aliveGrassAt(y, z, x));
  } else if (uniforms.plants != 0 && t == DIRT) {
    // Exposed dirt near living grass is primed, and sprouts with the spread chance.
    let primed = uniforms.grassSpread > 0.0 && blockOf(cellAt(y + 1, z, x)) == AIR && grassInReach(y, z, x);
    if (primed && cellRandom(randomIndex) < uniforms.grassSpread) {
      next = GRASS;
    } else {
      next = select(DIRT, PRIMED, primed);
    }
  }
  return next;
}
`;

/** A fresh per-tick seed for the plant rules' random numbers. */
export const randomSeed = () => (Math.random() * 2 ** 32) >>> 0;
