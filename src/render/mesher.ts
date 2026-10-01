import {
  Block, CHUNK_HEIGHT, CHUNK_SIZE, FALLING_LEVEL, SOURCE_LEVEL, blockIndex, cellLevel, cellType, isSolid,
} from '../constants';
import type { World } from '../world/world';

/** Floats per vertex: position (3), normal (3), block type (1; for wheat, type + 16 * growth stage). */
export const VERTEX_FLOATS = 7;

export interface MeshData {
  vertices: Float32Array<ArrayBuffer>;
  indices: Uint32Array<ArrayBuffer>;
}

export interface ChunkMesh {
  /** Stone, dirt, lava. */
  opaque: MeshData;
  /** Translucent water, drawn after opaque geometry. */
  water: MeshData;
}

class MeshBuilder {
  v: number[] = [];
  i: number[] = [];
  quad(corners: number[][], n: readonly number[], type: number): void {
    const base = this.v.length / VERTEX_FLOATS;
    for (const c of corners) this.v.push(c[0], c[1], c[2], n[0], n[1], n[2], type);
    this.i.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  build(): MeshData {
    return { vertices: new Float32Array(this.v), indices: new Uint32Array(this.i) };
  }
}

// For each face: normal and the 4 corner offsets (counter-clockwise seen from outside).
const FACES = [
  { n: [1, 0, 0], c: [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]] },
  { n: [-1, 0, 0], c: [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]] },
  { n: [0, 1, 0], c: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]] },
  { n: [0, -1, 0], c: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]] },
  { n: [0, 0, 1], c: [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]] },
  { n: [0, 0, -1], c: [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]] },
] as const;

/** Only full solid blocks hide a neighbour's face (fluids can be partial height). */
const occludes = isSolid;

/** Height of a wheat plant at a growth stage, 0..1. */
export const wheatHeight = (stage: number) => 0.25 + stage * 0.1;

/** Two crossed, double-sided quads: the usual way to draw a plant in a block world. */
function plant(builder: MeshBuilder, x: number, y: number, z: number, h: number, kind: number): void {
  const a = 0.15, b = 0.85, up = [0, 1, 0] as const; // lit like a top face
  for (const [x0, z0, x1, z1] of [[a, a, b, b], [b, a, a, b]]) {
    const bottom0 = [x + x0, y, z + z0], bottom1 = [x + x1, y, z + z1];
    const top0 = [x + x0, y + h, z + z0], top1 = [x + x1, y + h, z + z1];
    builder.quad([bottom0, bottom1, top1, top0], up, kind);
    builder.quad([bottom1, bottom0, top0, top1], up, kind);
  }
}

/** Height of a fluid block's surface, 0..1. */
function fluidHeight(c: number, above: number): number {
  if (cellType(above) === cellType(c)) return 1;
  const level = cellLevel(c);
  return level >= SOURCE_LEVEL ? 0.875 : Math.max(0.125, (level / FALLING_LEVEL) * 0.8);
}

/**
 * Build a face-culled mesh for one chunk. Neighbouring chunks (active or ghost)
 * are read through the world so border faces are culled correctly.
 */
export function meshChunk(world: World, cx: number, cz: number): ChunkMesh {
  const S = CHUNK_SIZE, H = CHUNK_HEIGHT, P = S + 2;
  // Padded copy: 1-block border from the 4 neighbours (missing neighbours read as air).
  const pad = new Uint8Array(P * P * H);
  const pidx = (x: number, y: number, z: number) => (y * P + (z + 1)) * P + (x + 1);
  const self = world.getChunk(cx, cz)!;
  for (let y = 0; y < H; y++) {
    for (let z = -1; z <= S; z++) {
      for (let x = -1; x <= S; x++) {
        const inside = x >= 0 && x < S && z >= 0 && z < S;
        if (!inside && (x < 0 || x >= S) && (z < 0 || z >= S)) continue; // corners unused
        pad[pidx(x, y, z)] = inside
          ? self.data[blockIndex(x, y, z)]
          : world.getCell(cx * S + x, y, cz * S + z);
      }
    }
  }
  const at = (x: number, y: number, z: number) =>
    y < 0 ? Block.Stone : y >= H ? Block.Air : pad[pidx(x, y, z)];

  const solid = new MeshBuilder(), water = new MeshBuilder();
  const ox = cx * S, oz = cz * S;
  for (let y = 0; y < H; y++) {
    for (let z = 0; z < S; z++) {
      for (let x = 0; x < S; x++) {
        const c = at(x, y, z), t = cellType(c);
        if (t === Block.Air) continue;
        if (t === Block.Wheat) {
          const stage = cellLevel(c);
          plant(solid, ox + x, y, oz + z, wheatHeight(stage), t + 16 * stage);
          continue;
        }
        const fluid = t === Block.Water || t === Block.Lava;
        const h = fluid ? fluidHeight(c, at(x, y + 1, z)) : 1;
        for (const f of FACES) {
          const nc = at(x + f.n[0], y + f.n[1], z + f.n[2]), nt = cellType(nc);
          let visible: boolean;
          if (fluid) {
            // Against air or the other fluid; against the same fluid only where the neighbour's surface is lower.
            visible = (nt !== t && !occludes(nt))
              || (nt === t && f.n[1] === 0 && fluidHeight(nc, at(x + f.n[0], y + 1, z + f.n[2])) < h);
          } else {
            visible = !occludes(nt);
          }
          if (!visible) continue;
          const corners = f.c.map((o) => [ox + x + o[0], y + (o[1] === 1 ? h : 0), oz + z + o[2]]);
          (t === Block.Water ? water : solid).quad(corners, f.n, t);
        }
      }
    }
  }
  return { opaque: solid.build(), water: water.build() };
}
