import * as tf from '@tensorflow/tfjs';
import {
  Block, FALLING_LEVEL, LAVA_DECAY, LEVEL_MUL, SOURCE_LEVEL, WATER_DECAY,
} from '../constants';

// One block-update tick as a cellular automaton over a [H, Z, X] int32 cell tensor.
// Every cell reads only its 6 neighbours from the previous state, so the whole
// region updates in parallel. Cells on the outer edge see "air" beyond the region;
// callers surround the cells they care about with a ghost halo and discard it.
//
// Fluid rules (Minecraft-like):
//   - a source has level 8; flowing fluid has level 1..7
//   - fluid directly above an air/flowing cell makes it falling fluid (level 7)
//   - a fluid cell resting on solid ground (or on a source) emits level - decay sideways
//   - flowing cells are recomputed from neighbours each tick, so they drain when cut off
//   - lava touching water turns to stone; a cell both fluids want to fill becomes stone

type Axis = 0 | 1 | 2; // 0 = y, 1 = z, 2 = x

/**
 * Value of the neighbour at offset `dir` (+1 / -1) along `axis` for every cell,
 * filling cells past the region edge with `fill`.
 */
function neighbour(t: tf.Tensor3D, axis: Axis, dir: 1 | -1, fill: number): tf.Tensor3D {
  const shape = t.shape;
  const n = shape[axis];
  const begin = [0, 0, 0], size = [...shape] as [number, number, number];
  size[axis] = n - 1;
  if (dir === 1) begin[axis] = 1;
  const body = t.slice(begin, size);
  const padShape = [...shape] as [number, number, number];
  padShape[axis] = 1;
  const pad = tf.fill(padShape, fill, t.dtype);
  return (dir === 1 ? tf.concat([body, pad], axis) : tf.concat([pad, body], axis)) as tf.Tensor3D;
}

const HORIZONTAL: Array<[Axis, 1 | -1]> = [[1, 1], [1, -1], [2, 1], [2, -1]];

export function blockUpdateStep(cells: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    const type = cells.mod(LEVEL_MUL) as tf.Tensor3D;
    const level = cells.floorDiv(LEVEL_MUL) as tf.Tensor3D;

    const isType = (b: Block) => type.equal(b);
    const solid = isType(Block.Stone).logicalOr(isType(Block.Dirt));
    const isSource = level.equal(SOURCE_LEVEL);
    const water = isType(Block.Water), lava = isType(Block.Lava);
    const flowing = water.logicalOr(lava).logicalAnd(isSource.logicalNot());

    // Below the world is bedrock-like stone, so a fluid on y=0 counts as supported.
    const typeBelow = neighbour(type, 0, -1, Block.Stone);
    const levelBelow = neighbour(level, 0, -1, 0);
    const supportedBelow = typeBelow.equal(Block.Stone)
      .logicalOr(typeBelow.equal(Block.Dirt))
      .logicalOr(levelBelow.equal(SOURCE_LEVEL));

    const zero = tf.zerosLike(level);

    /** Level a fluid wants to put in each cell this tick (0 = none). */
    const want = (isF: tf.Tensor, decay: number) => {
      const emit = tf.where(isF.logicalAnd(supportedBelow), level.sub(decay), zero) as tf.Tensor3D;
      let side: tf.Tensor = zero;
      for (const [axis, dir] of HORIZONTAL) side = tf.maximum(side, neighbour(emit, axis, dir, 0));
      const above = neighbour(isF.cast('int32') as tf.Tensor3D, 0, 1, 0).mul(FALLING_LEVEL);
      return tf.maximum(side, above);
    };
    const wantWater = want(water, WATER_DECAY);
    const wantLava = want(lava, LAVA_DECAY);

    // Recompute air and flowing cells from their neighbours.
    const replaceable = isType(Block.Air).logicalOr(flowing);
    const ww = wantWater.greater(0), wl = wantLava.greater(0);
    const int = (v: number) => tf.scalar(v, 'int32');
    let next: tf.Tensor = tf.zerosLike(cells); // air
    next = tf.where(wl, wantLava.mul(LEVEL_MUL).add(Block.Lava), next);
    next = tf.where(ww, wantWater.mul(LEVEL_MUL).add(Block.Water), next);
    next = tf.where(ww.logicalAnd(wl), int(Block.Stone), next);
    let out: tf.Tensor = tf.where(replaceable, next, cells);

    // Lava touching water solidifies.
    let touchesWater: tf.Tensor = tf.zerosLike(water);
    for (const [axis, dir] of [...HORIZONTAL, [0, 1], [0, -1]] as Array<[Axis, 1 | -1]>) {
      touchesWater = touchesWater.logicalOr(neighbour(water.cast('int32') as tf.Tensor3D, axis, dir, 0).equal(1));
    }
    out = tf.where(lava.logicalAnd(touchesWater), int(Block.Stone), out);

    // Solid blocks never change here; keep the mask explicit for clarity.
    return tf.where(solid, cells, out) as tf.Tensor3D;
  });
}
