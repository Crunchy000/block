/** The world generator's settings (tf/worldgen.ts, sim/gpuWorldgen.ts, world/jsWorldgen.ts), in a module without TF.js. */
export interface ChunkCoord { cx: number; cz: number }

export const DEFAULT_SEED = 1337;

/** Diamond ore: deep stone (y up to DIAMOND_MAX_Y) where 3D noise at a 3-block scale is above this. */
export const DIAMOND_THRESHOLD = 0.82;
export const DIAMOND_MAX_Y = 24;

/** Chance that a 4x4-column area of dry land has a patch of wild wheat (at random growth stages). */
export const WHEAT_PATCH_CHANCE = 1 / 256;
/** Whether wild wheat is generated at all (off for now: its texture is being redone). */
export const WILD_WHEAT = false;
/**
 * Sand and gravel instead of dirt in the top 3 blocks: beaches of sand where the ground is
 * from SAND_BELOW blocks under sea level to SAND_ABOVE above it (no grass on them), gravel
 * on the sea floor deeper than that.
 */
export const SAND_BELOW = 3;
export const SAND_ABOVE = 1;
