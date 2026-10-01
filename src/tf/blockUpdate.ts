import * as tf from '@tensorflow/tfjs';
import { Block, FALLING_LEVEL, LAVA_DECAY, LEVEL_MUL, SOURCE_LEVEL, WATER_DECAY } from '../constants';
import { lavaLevel, replaceable, supports, waterLevel } from './blockUpdateReference';

// One block-update tick as a cellular automaton over a [H, Z, X] int32 cell tensor.
// Every cell reads only its neighbours from the previous state, so the whole region
// updates in parallel. Cells past the region edge read as air (bedrock below y = 0);
// callers surround the cells they keep with a ghost border and discard it.
//
// Fluid rules (Minecraft-like; blockUpdateReference.ts is the cell-by-cell spec):
//   - a source has level 8; flowing fluid has level 1..7
//   - fluid directly above an air/flowing cell makes it falling fluid (level 7)
//   - a fluid cell resting on a solid block or a source spreads level - decay sideways
//   - flowing cells are recomputed from neighbours each tick, so they drain when cut off
//   - lava with water beside or above it turns to stone; a cell both fluids flow into becomes stone

/** Every possible cell value: types 0..7 × levels 0..8. */
const CELL_VALUES = LEVEL_MUL * (SOURCE_LEVEL + 1);
const table = (f: (c: number) => number | boolean) => Int32Array.from({ length: CELL_VALUES }, (_, c) => Number(f(c)));

// Lookup tables indexed by cell value: one tf.gather replaces a chain of compare ops.
const SUPPORTS = table(supports);
const WATER_LEVEL = table(waterLevel);
const LAVA_LEVEL = table(lavaLevel);
const REPLACEABLE = table(replaceable);

export function blockUpdateStep(cells: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    const [H, D, W] = cells.shape;
    const size: [number, number, number] = [H, D, W];
    // Explicit int32 scalars: a bare number would be read as float32 and upcast the result.
    const int = (v: number) => tf.scalar(v, 'int32');
    const lookup = (t: Int32Array): tf.Tensor => tf.gather(tf.tensor1d(t, 'int32'), cells);

    // Neighbour views: pad once, then slice. `below` fills under y = 0 with `fill`; others with 0.
    const below = (t: tf.Tensor, fill: number) => t.pad([[1, 0], [0, 0], [0, 0]], fill).slice([0, 0, 0], size);
    const above = (t: tf.Tensor) => t.pad([[0, 1], [0, 0], [0, 0]]).slice([1, 0, 0], size);
    const sides = (t: tf.Tensor) => {
      const p = t.pad([[0, 0], [1, 1], [1, 1]]);
      return [p.slice([0, 0, 1], size), p.slice([0, 2, 1], size), p.slice([0, 1, 0], size), p.slice([0, 1, 2], size)];
    };

    const water = lookup(WATER_LEVEL), lava = lookup(LAVA_LEVEL);
    // Nested tidies release each stage's temporaries as soon as it's done (peak GPU memory).
    const supportBelow = tf.tidy(() => below(lookup(SUPPORTS), 1));
    const want = (level: tf.Tensor, decay: number) => tf.tidy(() => {
      const emit = level.sub(int(decay)).maximum(int(0)).mul(supportBelow);
      const [n, s, w, e] = sides(emit);
      const fall = above(level).minimum(int(1)).mul(int(FALLING_LEVEL));
      return n.maximum(s).maximum(w.maximum(e)).maximum(fall);
    });
    const wantWater = want(water, WATER_DECAY);
    const wantLava = want(lava, LAVA_DECAY);

    return tf.tidy(() => {
      const ww = wantWater.greater(int(0)), wl = wantLava.greater(int(0));
      const flowed = tf.where(ww,
        wantWater.mul(int(LEVEL_MUL)).add(int(Block.Water)),
        wantLava.mul(int(LEVEL_MUL)).add(wl.cast('int32').mul(int(Block.Lava))));
      const isReplaceable = lookup(REPLACEABLE).cast('bool');
      const touchesWater = tf.tidy(() => {
        const present = water.minimum(int(1));
        return tf.addN([...sides(present), above(present)]).greater(int(0));
      });
      const toStone = ww.logicalAnd(wl).logicalAnd(isReplaceable)
        .logicalOr(lava.greater(int(0)).logicalAnd(touchesWater));
      const moved = tf.where(isReplaceable, flowed, cells);
      // A full-size fill, not a scalar: tf.where would expand a scalar on the CPU (main thread).
      return tf.where(toStone, tf.fill(size, Block.Stone, 'int32'), moved) as tf.Tensor3D;
    });
  });
}
