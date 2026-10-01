import {
  Block, FALLING_LEVEL, LAVA_DECAY, SOURCE_LEVEL, WATER_DECAY, cell, cellLevel, cellType, isFluid, isSolid,
} from '../constants';

// Plain-JS, cell-at-a-time version of the block-update rules. It is the readable
// spec for blockUpdateStep (the TF.js version) and the oracle the tests compare
// it against. Not used at runtime.

/** Cells that hold up a fluid resting on them: solid blocks and fluid sources. */
export const supports = (c: number) => isSolid(cellType(c)) || cellLevel(c) === SOURCE_LEVEL;
export const waterLevel = (c: number) => (cellType(c) === Block.Water ? cellLevel(c) : 0);
export const lavaLevel = (c: number) => (cellType(c) === Block.Lava ? cellLevel(c) : 0);
/** Air and flowing fluid are recomputed from their neighbours every tick. */
export const replaceable = (c: number) =>
  cellType(c) === Block.Air || (isFluid(cellType(c)) && cellLevel(c) < SOURCE_LEVEL);

/** One tick over a [H, D, W] region (index (y * D + z) * W + x); cells outside read as air. */
export function blockUpdateReference(cells: ArrayLike<number>, H: number, D: number, W: number): Int32Array {
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

  const out = new Int32Array(H * D * W);
  for (let y = 0; y < H; y++) {
    for (let z = 0; z < D; z++) {
      for (let x = 0; x < W; x++) {
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
        out[(y * D + z) * W + x] = next;
      }
    }
  }
  return out;
}
