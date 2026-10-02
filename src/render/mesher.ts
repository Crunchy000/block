import {
  Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, FALLING_LEVEL, SOURCE_LEVEL, cellLevel, cellType, isSolid,
} from '../constants';

// Chunk meshes are lists of face records: one u32 per quad, which the vertex shader
// expands into the quad's two triangles (render/shaders.ts). The game meshes on the GPU
// (render/gpuMesher.ts); meshFaces here is the same mesher in plain JS: the reference
// the GPU tests compare against, and the mesher for the CPU fallback.
//
// Record layout, low bits first:
//   x 4 | z 4 | y 6       the block, chunk-local
//   face 4                a Face
//   type 3                the block type
//   aux 4                 fluids: surface height code (heightCode); wheat: growth stage; stone: concrete colour (0 plain); else 0

export const enum Face {
  // The six sides of a block (normals +x, -x, +y, -y, +z, -z).
  East = 0, West, Top, Bottom, South, North,
  // A plant is two crossed, double-sided quads: the usual way to draw one in a block world.
  PlantA, PlantABack, PlantB, PlantBBack,
}

/** Records a chunk can need at most: six faces (or four plant quads) per cell. */
export const FACE_CAPACITY = 6 * CHUNK_VOLUME;

/** Height code of fluid with the same fluid on top of it: the block is full. */
export const FULL_HEIGHT = 15;

export const packFace = (x: number, y: number, z: number, face: Face, type: Block, aux: number): number =>
  (x | (z << 4) | (y << 8) | (face << 14) | (type << 18) | (aux << 21)) >>> 0;

export const unpackFace = (r: number) => ({
  x: r & 15, z: (r >> 4) & 15, y: (r >> 8) & 63, face: ((r >> 14) & 15) as Face, type: ((r >> 18) & 7) as Block, aux: (r >> 21) & 15,
});

/**
 * Surface height of a fluid block as a small code: FULL_HEIGHT with the same fluid on
 * top, otherwise its level (1..8, 8 for a source). Higher codes are higher surfaces.
 */
export const heightCode = (c: number, above: number): number =>
  cellType(above) === cellType(c) ? FULL_HEIGHT : Math.min(SOURCE_LEVEL, Math.max(1, cellLevel(c)));

/** The surface height (0..1) for a height code. */
export const fluidHeight = (code: number): number =>
  code === FULL_HEIGHT ? 1 : code >= SOURCE_LEVEL ? 0.875 : Math.max(0.125, (code / FALLING_LEVEL) * 0.8);

/** Height of a wheat plant: a whole block at every stage (its texture shows it growing). */
export const WHEAT_HEIGHT = 1;

type Vec3 = readonly [number, number, number];
/** The normal of each side Face. */
export const FACE_NORMALS: readonly Vec3[] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
/** Corner offsets of each side Face, counter-clockwise seen from outside. */
export const FACE_CORNERS: readonly (readonly Vec3[])[] = [
  [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
  [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]],
  [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
  [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
  [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]],
  [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
];
/** Where a plant quad runs across its block: [x0, z0, x1, z1] for PlantA (and back), PlantB (and back). */
const PLANT_INSET = 0.15;
export const PLANT_QUADS = [
  [PLANT_INSET, PLANT_INSET, 1 - PLANT_INSET, 1 - PLANT_INSET],
  [1 - PLANT_INSET, PLANT_INSET, PLANT_INSET, 1 - PLANT_INSET],
] as const;

export interface ChunkFaces {
  /** Stone, dirt, grass, lava, wheat. */
  opaque: number[];
  /** Translucent water, drawn after opaque geometry. */
  water: number[];
}

/**
 * Face-culled mesh of one chunk. `cellAt` reads chunk-local cells: x and z from -1 to
 * CHUNK_SIZE (the neighbouring chunks' edges, so border faces are culled correctly) and
 * y from 0 to CHUNK_HEIGHT - 1. Records come out in cell order.
 */
export function meshFaces(cellAt: (x: number, y: number, z: number) => number): ChunkFaces {
  const at = (x: number, y: number, z: number) => (y < 0 ? Block.Stone : y >= CHUNK_HEIGHT ? Block.Air : cellAt(x, y, z));
  const opaque: number[] = [], water: number[] = [];
  for (let y = 0; y < CHUNK_HEIGHT; y++) {
    for (let z = 0; z < CHUNK_SIZE; z++) {
      for (let x = 0; x < CHUNK_SIZE; x++) {
        const c = at(x, y, z), t = cellType(c);
        if (t === Block.Air) continue;
        if (t === Block.Wheat) {
          const stage = Math.min(cellLevel(c), 15);
          for (const face of [Face.PlantA, Face.PlantABack, Face.PlantB, Face.PlantBBack]) opaque.push(packFace(x, y, z, face, t, stage));
          continue;
        }
        const fluid = t === Block.Water || t === Block.Lava;
        // aux: a fluid's surface height; stone's level (its concrete colour, 0 for plain stone).
        const h = fluid ? heightCode(c, at(x, y + 1, z)) : t === Block.Stone ? Math.min(cellLevel(c), 15) : 0;
        for (let f = Face.East; f <= Face.North; f++) {
          const [nx, ny, nz] = FACE_NORMALS[f];
          const nc = at(x + nx, y + ny, z + nz), nt = cellType(nc);
          // Only full solid blocks hide a neighbour's face (fluids can be partial height).
          // Fluid shows against air or the other fluid, and against the same fluid only
          // where the neighbour's surface is lower.
          const visible = fluid
            ? (nt !== t && !isSolid(nt)) || (nt === t && ny === 0 && heightCode(nc, at(x + nx, y + 1, z + nz)) < h)
            : !isSolid(nt);
          if (visible) (t === Block.Water ? water : opaque).push(packFace(x, y, z, f, t, h));
        }
      }
    }
  }
  return { opaque, water };
}

export interface Quad {
  /** Four corners in world space; triangles (0, 1, 2) and (0, 2, 3). */
  corners: number[][];
  normal: readonly number[];
  /** Block type, plus 16 x growth stage for wheat (what the fragment shader gets). */
  kind: number;
}

/** The quad a face record draws, for a chunk whose corner is at world (ox, 0, oz): what the vertex shader does. */
export function faceQuad(record: number, ox: number, oz: number): Quad {
  const { x, y, z, face, type, aux } = unpackFace(record);
  const bx = ox + x, bz = oz + z;
  if (face >= Face.PlantA) {
    const [x0, z0, x1, z1] = PLANT_QUADS[(face - Face.PlantA) >> 1], h = WHEAT_HEIGHT;
    const b0 = [bx + x0, y, bz + z0], b1 = [bx + x1, y, bz + z1], t0 = [bx + x0, y + h, bz + z0], t1 = [bx + x1, y + h, bz + z1];
    const front = (face - Face.PlantA) % 2 === 0;
    return { corners: front ? [b0, b1, t1, t0] : [b1, b0, t0, t1], normal: [0, 1, 0], kind: type + 16 * aux };
  }
  const h = type === Block.Water || type === Block.Lava ? fluidHeight(aux) : 1;
  const corners = FACE_CORNERS[face].map((o) => [bx + o[0], y + (o[1] === 1 ? h : 0), bz + o[2]]);
  return { corners, normal: FACE_NORMALS[face], kind: type === Block.Stone ? type + 16 * aux : type };
}
