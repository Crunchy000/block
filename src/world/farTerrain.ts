import * as tf from '@tensorflow/tfjs';
import { SEA_LEVEL } from '../constants';
import { log } from '../log';
import { terrainHeight } from '../tf/worldgen';

/**
 * Far terrain: the landscape beyond the loaded chunks, as one low-detail height field.
 *
 * The ground's height at any column is a formula (terrainHeight, the one world generation
 * uses), so the far terrain needs no chunks: it samples the formula on a grid around the
 * player and draws the result as plain triangles, grass on land and water at sea level.
 * Grid points are 4 blocks apart near the player and further apart with distance (up to
 * 64), so the whole ring is a few tens of thousands of points out to a couple of km.
 * Inside the area the real chunks cover it isn't drawn (the shader discards it there).
 * No caves, plants or edits: at that distance only the shape of the land shows.
 */

/** The grid recentres in steps of this many blocks (its widest spacing, so points don't swim). */
export const FAR_SNAP = 64;
/** The top of a sea-level water block, as chunks draw it (a source block: 0.875 high). */
export const SEA_SURFACE = SEA_LEVEL + 0.875;

/** Spacing at distance d from the grid's centre: 4 blocks out to 128, doubling every doubling of distance, at most 64. */
const spacing = (d: number) => Math.min(FAR_SNAP, Math.max(4, 2 ** Math.floor(Math.log2(Math.max(1, d / 16)))));

/** Offsets from the centre along one axis, symmetric, from -extent to extent (or a step past it). */
export function farAxis(extent: number): number[] {
  const out = [0];
  for (let d = 0; d < extent;) out.push(d += spacing(d));
  return [...out.slice(1).reverse().map((v) => -v), ...out];
}

/** Ground heights at every grid point (row-major: z rows of x), from the world generation formula. */
export async function farHeights(axis: number[], cx: number, cz: number, seed?: number): Promise<Float32Array> {
  const n = axis.length;
  const t = tf.tidy(() => {
    const a = tf.tensor1d(axis);
    const wx = a.add(cx).reshape([1, n]).add(tf.zeros([n, 1]));
    const wz = a.add(cz).reshape([n, 1]).add(tf.zeros([1, n]));
    return terrainHeight(wx, wz, seed);
  });
  try {
    return Float32Array.from(await t.data());
  } finally {
    t.dispose();
  }
}

/** Vertex positions (x, y, z) for the grid: the top of the ground, or the sea's surface over water. */
export function farVertices(axis: number[], cx: number, cz: number, heights: Float32Array): Float32Array<ArrayBuffer> {
  const n = axis.length;
  const v = new Float32Array(n * n * 3);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const k = j * n + i;
      v.set([cx + axis[i], Math.max(heights[k] + 1, SEA_SURFACE), cz + axis[j]], k * 3);
    }
  }
  return v;
}

/** Two triangles per grid cell. */
export function farIndices(n: number): Uint32Array<ArrayBuffer> {
  const idx = new Uint32Array((n - 1) * (n - 1) * 6);
  let o = 0;
  for (let j = 0; j < n - 1; j++) {
    for (let i = 0; i < n - 1; i++) {
      const a = j * n + i, b = a + 1, c = a + n, d = c + 1;
      idx.set([a, c, b, b, c, d], o);
      o += 6;
    }
  }
  return idx;
}

/** The far terrain's mesh, rebuilt around the player as they travel (a renderer uploads it when `version` changes). */
export class FarTerrain {
  readonly axis: number[];
  readonly vertices: Float32Array<ArrayBuffer>;
  readonly indices: Uint32Array<ArrayBuffer>;
  /** Counts the rebuilds (0 before the first). */
  version = 0;
  /** Where the current mesh is centred. */
  centre?: [number, number];
  private building = false;

  constructor(readonly extent: number, private readonly seed?: number) {
    this.axis = farAxis(extent);
    const n = this.axis.length;
    this.indices = farIndices(n);
    this.vertices = new Float32Array(n * n * 3);
  }

  /** Has a mesh to draw. */
  get ready(): boolean {
    return this.version > 0;
  }

  get points(): number {
    return this.axis.length ** 2;
  }

  get bytes(): number {
    return this.vertices.byteLength + this.indices.byteLength;
  }

  /** Recentre on the player once they've moved to another FAR_SNAP cell (in the background). */
  update(x: number, z: number): void {
    const cx = Math.round(x / FAR_SNAP) * FAR_SNAP, cz = Math.round(z / FAR_SNAP) * FAR_SNAP;
    if (this.building || (this.centre?.[0] === cx && this.centre[1] === cz)) return;
    this.building = true;
    farHeights(this.axis, cx, cz, this.seed)
      .then((heights) => {
        this.vertices.set(farVertices(this.axis, cx, cz, heights));
        this.centre = [cx, cz];
        this.version++;
      })
      .catch((e: unknown) => {
        const [text, stack] = log.describe(e);
        log.warn(`Far terrain failed: ${text}`, stack);
      })
      .finally(() => { this.building = false; });
  }
}
