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
  Grass = 5,
  Wheat = 6,
  /** Diamond ore: deep in the stone, mined (broken) to collect diamonds. */
  Diamond = 7,
  /**
   * Sand and gravel aren't cell types (the 3 type bits are all taken): they're stone with
   * level SAND_LEVEL or GRAVEL_LEVEL. Picking and CellStore.readBox report them as these
   * (blockId), so the game can tell them apart from stone.
   */
  Sand = 8,
  Gravel = 9,
}

export const BLOCK_NAMES = ['air', 'stone', 'dirt', 'water', 'lava', 'grass', 'wheat', 'diamond', 'sand', 'gravel'] as const;

/**
 * Pastel concrete, the player's building blocks: stone with a colour in its level bits
 * (level 1..8; generated stone is level 0; sand and gravel are 9 and 10). Solid like stone
 * everywhere; drawn smooth in its colour. The block-update tables cover levels up to MAX_LEVEL.
 */
export const CONCRETE_COLOURS: ReadonlyArray<{ name: string; rgb: [number, number, number] }> = [
  { name: 'pink', rgb: [0.96, 0.71, 0.76] },
  { name: 'peach', rgb: [0.98, 0.79, 0.66] },
  { name: 'butter', rgb: [0.97, 0.9, 0.62] },
  { name: 'mint', rgb: [0.7, 0.89, 0.78] },
  { name: 'sky', rgb: [0.68, 0.83, 0.95] },
  { name: 'periwinkle', rgb: [0.73, 0.76, 0.94] },
  { name: 'lavender', rgb: [0.84, 0.76, 0.93] },
  { name: 'cream', rgb: [0.95, 0.93, 0.88] },
];

/**
 * Level, stored in the high bits of a cell (cell = type + LEVEL_MUL * level).
 * Fluids: SOURCE_LEVEL marks a source block; 1..7 are flowing.
 * Dirt: 1 marks PRIMED_DIRT (see below). Wheat: growth stage 0..WHEAT_RIPE. Everything else: 0.
 */
export const LEVEL_MUL = 8;
export const SOURCE_LEVEL = 8;
export const FALLING_LEVEL = 7;
export const WATER_DECAY = 1;
export const LAVA_DECAY = 2;

/** Sand and gravel: stone with these levels (after the concrete colours). */
export const SAND_LEVEL = 9;
export const GRAVEL_LEVEL = 10;
/** The highest level any cell has. */
export const MAX_LEVEL = GRAVEL_LEVEL;

export const cell = (type: Block, level = 0): number => type + LEVEL_MUL * level;
export const SAND = cell(Block.Stone, SAND_LEVEL);
export const GRAVEL = cell(Block.Stone, GRAVEL_LEVEL);
/** The cell of pastel concrete colour `i` (an index into CONCRETE_COLOURS). */
export const concrete = (i: number): number => cell(Block.Stone, i + 1);
export const cellType = (c: number): Block => (c & 7) as Block;
export const cellLevel = (c: number): number => c >> 3;
/** What the player can place (hotbar keys 1–9, 0): the concrete colours, then sand and gravel. */
export const BUILDING_BLOCKS: ReadonlyArray<{ name: string; cell: number; block: Block; swatch: string }> = [
  ...CONCRETE_COLOURS.map(({ name, rgb }, i) => ({
    name: `${name} concrete`, cell: concrete(i), block: Block.Stone, swatch: `rgb(${rgb.map((v) => Math.round(v * 255)).join(' ')})`,
  })),
  { name: 'sand', cell: SAND, block: Block.Sand, swatch: 'url(textures/sand.png) center / cover' },
  { name: 'gravel', cell: GRAVEL, block: Block.Gravel, swatch: 'url(textures/gravel.png) center / cover' },
];
/** What block a cell is, for the game: its type, or Sand / Gravel for those (see Block.Sand). */
export const blockId = (c: number): Block =>
  c === SAND ? Block.Sand : c === GRAVEL ? Block.Gravel : cellType(c);

/**
 * Dirt that grass can spread onto this tick: air above it and living grass in reach.
 * The block-update step recomputes the flag every tick. It changes nothing about how
 * the block looks or behaves; it lets the scheduler see that a chunk still has
 * grass to grow (a random process, so "nothing changed" doesn't mean "settled").
 */
export const PRIMED_DIRT = cell(Block.Dirt, 1);

/** Chance per tick that primed dirt turns into grass (5 ticks / second, so ~3 s per block). */
export const GRASS_SPREAD_CHANCE = 1 / 16;

/** Wheat grows through stages 0..7 and stops when ripe. */
export const WHEAT_RIPE = 7;
/** Chance per tick that wheat grows a stage (~8 s per stage, ~1 minute to ripen). */
export const WHEAT_GROW_CHANCE = 1 / 40;
/** With water beside its soil (or beside the plant) wheat grows faster (~2.5 s per stage). */
export const WHEAT_GROW_CHANCE_WET = 1 / 12;

/** Per-tick chances for the random plant rules. */
export interface PlantRates {
  grassSpread: number;
  wheatGrow: number;
  wheatGrowWet: number;
}
export const DEFAULT_RATES: PlantRates = {
  grassSpread: GRASS_SPREAD_CHANCE, wheatGrow: WHEAT_GROW_CHANCE, wheatGrowWet: WHEAT_GROW_CHANCE_WET,
};

/** Cells with plant rules, which need random numbers each tick: grass, primed dirt and wheat. */
export const isPlant = (c: number): boolean =>
  cellType(c) === Block.Grass || cellType(c) === Block.Wheat || c === PRIMED_DIRT;
/** Plant cells that may still change by chance: primed dirt, and wheat that isn't ripe. */
export const isGrowing = (c: number): boolean =>
  c === PRIMED_DIRT || (cellType(c) === Block.Wheat && cellLevel(c) < WHEAT_RIPE);

export const isSolid = (t: Block): boolean =>
  t === Block.Stone || t === Block.Dirt || t === Block.Grass || t === Block.Diamond || t === Block.Sand || t === Block.Gravel;
export const isFluid = (t: Block): boolean => t === Block.Water || t === Block.Lava;

/** Index into a chunk's data array; layout is [y][z][x] to match the [H, Z, X] tensors. */
export const blockIndex = (x: number, y: number, z: number): number =>
  (y * CHUNK_SIZE + z) * CHUNK_SIZE + x;

export const chunkKey = (cx: number, cz: number): string => `${cx},${cz}`;
