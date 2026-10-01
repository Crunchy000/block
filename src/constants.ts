/** Horizontal chunk size (chunks are CHUNK_SIZE x CHUNK_SIZE columns). */
export const CHUNK_SIZE = 16;
/** Vertical size of the world / every chunk column. */
export const CHUNK_HEIGHT = 64;
export const CHUNK_VOLUME = CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT;

export const SEA_LEVEL = 30;

/**
 * Chunks with Chebyshev distance <= ACTIVE_RADIUS from the player's chunk are
 * "active": simulated and rendered. The ring at ACTIVE_RADIUS + 1 are "ghost"
 * chunks: generated and kept in memory so block updates and meshing have valid
 * neighbour data at the edge, but never stepped or drawn.
 */
export const ACTIVE_RADIUS = 3;
export const GHOST_RADIUS = ACTIVE_RADIUS + 1;

/** Block types. Stored in the low 3 bits of a cell. */
export const enum Block {
  Air = 0,
  Stone = 1,
  Dirt = 2,
  Water = 3,
  Lava = 4,
}

export const BLOCK_NAMES = ['air', 'stone', 'dirt', 'water', 'lava'] as const;

/**
 * Fluid level, stored in the high bits of a cell (cell = type + LEVEL_MUL * level).
 * SOURCE_LEVEL marks a source block; 1..7 are flowing; 0 for non-fluids.
 */
export const LEVEL_MUL = 8;
export const SOURCE_LEVEL = 8;
export const FALLING_LEVEL = 7;
export const WATER_DECAY = 1;
export const LAVA_DECAY = 2;

export const cell = (type: Block, level = 0): number => type + LEVEL_MUL * level;
export const cellType = (c: number): Block => (c & 7) as Block;
export const cellLevel = (c: number): number => c >> 3;

export const isSolid = (t: Block): boolean => t === Block.Stone || t === Block.Dirt;
export const isFluid = (t: Block): boolean => t === Block.Water || t === Block.Lava;

/** Index into a chunk's data array; layout is [y][z][x] to match the [H, Z, X] tensors. */
export const blockIndex = (x: number, y: number, z: number): number =>
  (y * CHUNK_SIZE + z) * CHUNK_SIZE + x;

export const chunkKey = (cx: number, cz: number): string => `${cx},${cz}`;
