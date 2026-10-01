import * as tf from '@tensorflow/tfjs';
import {
  Block, DEFAULT_RATES, FALLING_LEVEL, LAVA_DECAY, LEVEL_MUL, SOURCE_LEVEL, WATER_DECAY, WHEAT_RIPE,
  type PlantRates,
} from '../constants';
import {
  isAir, isDirt, isGrass, isSoil, isWheat, lavaLevel, replaceable, supports, waterLevel, wheatStage,
} from './blockUpdateReference';

// One block-update tick as a cellular automaton over int32 cells shaped [H, Z, X],
// or a batch [N, H, Z, X] of independent regions (the simulation batches chunks).
// Every cell reads only its neighbours from the previous state, so all cells update
// in parallel. Cells past a region's edge read as air (bedrock below y = 0);
// callers surround the cells they keep with a ghost border and discard it.
//
// Rules (Minecraft-like; blockUpdateReference.ts is the cell-by-cell spec):
//   - a fluid source has level 8; flowing fluid has level 1..7
//   - fluid directly above an air/flowing cell makes it falling fluid (level 7)
//   - a fluid cell resting on a solid block or a source spreads level - decay sideways
//   - flowing cells are recomputed from neighbours each tick, so they drain when cut off
//   - lava with water beside or above it turns to stone; a cell both fluids flow into becomes stone
//   - grass under a solid block or fluid source dies back to dirt
//   - dirt with air above and living grass in reach (1 sideways, 1 below to 3 above) is
//     primed, and turns to grass when its random number is below the spread chance
//   - wheat grows a stage at a time (faster with water beside it or its soil), pops off
//     without dirt or grass under it, and flowing fluid washes it away

/** Every possible cell value: types 0..7 × levels 0..8. */
const CELL_VALUES = LEVEL_MUL * (SOURCE_LEVEL + 1);
const table = (f: (c: number) => number | boolean) => Int32Array.from({ length: CELL_VALUES }, (_, c) => Number(f(c)));

// Lookup tables indexed by cell value: one tf.gather replaces a chain of compare ops.
const SUPPORTS = table(supports);
const WATER_LEVEL = table(waterLevel);
const LAVA_LEVEL = table(lavaLevel);
const REPLACEABLE = table(replaceable);
const IS_GRASS = table(isGrass);
const IS_DIRT = table(isDirt);
const IS_AIR = table(isAir);
const IS_WHEAT = table(isWheat);
const WHEAT_STAGE = table(wheatStage);
const IS_SOIL = table(isSoil);

/**
 * @param random uniform [0, 1) per cell, same shape as `cells` (see randomField). Without it
 *   the plant rules are skipped and grass, dirt and wheat are left unchanged (except for
 *   fluids washing wheat away); pass it whenever the cells contain grass, primed dirt or wheat.
 */
export function blockUpdateStep<T extends tf.Tensor3D | tf.Tensor4D>(
  cells: T, random?: tf.Tensor, rates: PlantRates = DEFAULT_RATES,
): T {
  return tf.tidy(() => {
    const batch = (t: tf.Tensor) => (t.rank === 4 ? t : t.expandDims(0)) as tf.Tensor4D;
    const input = batch(cells);
    const [N, H, D, W] = input.shape;
    const size: [number, number, number, number] = [N, H, D, W];
    // Explicit int32 scalars: a bare number would be read as float32 and upcast the result.
    const int = (v: number) => tf.scalar(v, 'int32');
    // Full-size constants for tf.where. A scalar would make TF.js expand it on the CPU (main thread).
    const full = (v: number) => tf.fill(size, v, 'int32');
    const lookup = (t: Int32Array): tf.Tensor => tf.gather(tf.tensor1d(t, 'int32'), input);

    // Neighbour views: pad once, then slice. `below` fills under y = 0 with `fill`, `above` past the top.
    const below = (t: tf.Tensor, fill: number) => t.pad([[0, 0], [1, 0], [0, 0], [0, 0]], fill).slice([0, 0, 0, 0], size);
    const above = (t: tf.Tensor, fill = 0) => t.pad([[0, 0], [0, 1], [0, 0], [0, 0]], fill).slice([0, 1, 0, 0], size);
    const sides = (t: tf.Tensor) => {
      const p = t.pad([[0, 0], [0, 0], [1, 1], [1, 1]]);
      return [p.slice([0, 0, 0, 1], size), p.slice([0, 0, 2, 1], size), p.slice([0, 0, 1, 0], size), p.slice([0, 0, 1, 2], size)];
    };

    const water = lookup(WATER_LEVEL), lava = lookup(LAVA_LEVEL), support = lookup(SUPPORTS);
    // Nested tidies release each stage's temporaries as soon as it's done (peak GPU memory).
    const supportBelow = tf.tidy(() => below(support, 1));
    const want = (level: tf.Tensor, decay: number) => tf.tidy(() => {
      const emit = level.sub(int(decay)).maximum(int(0)).mul(supportBelow);
      const [n, s, w, e] = sides(emit);
      const fall = above(level).minimum(int(1)).mul(int(FALLING_LEVEL));
      return n.maximum(s).maximum(w.maximum(e)).maximum(fall);
    });
    const wantWater = want(water, WATER_DECAY);
    const wantLava = want(lava, LAVA_DECAY);

    const ww = wantWater.greater(int(0)), wl = wantLava.greater(int(0));
    const wheat = lookup(IS_WHEAT).cast('bool');
    // Fluid flows into air and flowing fluid, and washes wheat away.
    const flooded = wheat.logicalAnd(ww.logicalOr(wl));
    let out = tf.tidy(() => {
      const flowed = tf.where(ww,
        wantWater.mul(int(LEVEL_MUL)).add(int(Block.Water)),
        wantLava.mul(int(LEVEL_MUL)).add(wl.cast('int32').mul(int(Block.Lava))));
      const floodable = lookup(REPLACEABLE).cast('bool').logicalOr(flooded);
      const touchesWater = tf.tidy(() => {
        const present = water.minimum(int(1));
        return tf.addN([...sides(present), above(present)]).greater(int(0));
      });
      const toStone = ww.logicalAnd(wl).logicalAnd(floodable)
        .logicalOr(lava.greater(int(0)).logicalAnd(touchesWater));
      const moved = tf.where(floodable, flowed, input);
      return tf.where(toStone, full(Block.Stone), moved);
    });

    if (random) {
      const fluids = out;
      out = tf.tidy(() => {
        const grass = lookup(IS_GRASS).cast('bool'), dirt = lookup(IS_DIRT).cast('bool');
        const alive = grass.logicalAnd(above(support).cast('bool').logicalNot());
        // Any living grass in reach: a 5x3x3 max-pool over the alive mask, padded so each
        // window spans one below to three above and one block to each side.
        const padded = alive.cast('float32').pad([[0, 0], [1, 3], [1, 1], [1, 1]]).reshape([N, H + 4, D + 2, W + 2, 1]);
        const reach = tf.maxPool3d(padded as tf.Tensor5D, [5, 3, 3], 1, 'valid').reshape(size).greater(0);
        const exposed = above(lookup(IS_AIR), 1).cast('bool');
        // With no chance to spread, nothing is primed (so grass can't keep chunks awake).
        const primed = rates.grassSpread > 0 ? dirt.logicalAnd(exposed).logicalAnd(reach) : tf.zerosLike(dirt);
        const becomesGrass = primed.logicalAnd(batch(random).less(rates.grassSpread)).logicalOr(alive);
        const dirtValue = primed.cast('int32').mul(int(LEVEL_MUL)).add(int(Block.Dirt)); // PRIMED_DIRT or plain dirt
        const grassOrDirt = tf.where(becomesGrass, full(Block.Grass), dirtValue);

        // Wheat: any water one block either side, at the plant's level or its soil's.
        const wetPadded = water.minimum(int(1)).cast('float32').pad([[0, 0], [1, 0], [1, 1], [1, 1]]).reshape([N, H + 1, D + 2, W + 2, 1]);
        const wet = tf.maxPool3d(wetPadded as tf.Tensor5D, [2, 3, 3], 1, 'valid').reshape(size).greater(0);
        const growChance = tf.where(wet, tf.fill(size, rates.wheatGrowWet), tf.fill(size, rates.wheatGrow));
        const stage = lookup(WHEAT_STAGE);
        const grows = stage.less(int(WHEAT_RIPE)).logicalAnd(batch(random).less(growChance));
        const grown = stage.add(grows.cast('int32')).mul(int(LEVEL_MUL)).add(int(Block.Wheat));
        const hasSoil = below(lookup(IS_SOIL), 0).cast('bool');
        const wheatValue = tf.where(hasSoil, grown, full(Block.Air));

        const plants = tf.where(grass.logicalOr(dirt), grassOrDirt, fluids);
        return tf.where(wheat.logicalAnd(flooded.logicalNot()), wheatValue, plants);
      });
    }
    return (cells.rank === 4 ? out : out.squeeze([0])) as T;
  });
}
