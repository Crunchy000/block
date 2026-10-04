import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, GRAVEL, SAND, SEA_LEVEL, SOURCE_LEVEL, WHEAT_RIPE, blockIndex, cell } from '../constants';
import {
  DEFAULT_SEED, DIAMOND_MAX_Y, DIAMOND_THRESHOLD, SAND_ABOVE, SAND_BELOW, WHEAT_PATCH_CHANCE, WILD_WHEAT, type ChunkCoord,
} from './worldgenParams';

/**
 * World generation in plain JavaScript, for safe mode (no TensorFlow.js there): the layers of
 * tf/worldgen.ts, a column at a time as the worldgen shader (sim/gpuWorldgen.ts) does them.
 * Every step is rounded to float32 (f) in the order TF.js does its tensor ops, so the world
 * comes out as TF.js's CPU backend makes it (the sine hash amplifies the smallest rounding
 * difference into a different value).
 */
const f = Math.fround;
const K1 = f(127.1), K2 = f(311.7), K3 = f(74.7), SCALE = f(43758.5453);
const fract = (v: number) => f(v - Math.floor(v));
const smooth = (t: number) => f(f(t * t) * f(f(t * -2) + 3));
const lerp = (a: number, b: number, t: number) => f(a + f(f(b - a) * t));
const sineHash = (dot: number) => fract(f(f(Math.sin(dot)) * SCALE));
const hash2 = (a: number, b: number) => sineHash(f(f(a * K1) + f(b * K2)));
const hash3 = (a: number, b: number, c: number) => sineHash(f(f(f(a * K1) + f(b * K2)) + f(c * K3)));

function valueNoise2(x: number, z: number, seed: number): number {
  const xi = Math.floor(x), zi = f(Math.floor(z) + seed);
  const u = smooth(f(x - xi)), v = smooth(f(z - Math.floor(z)));
  const xi1 = f(xi + 1), zi1 = f(zi + 1);
  return lerp(lerp(hash2(xi, zi), hash2(xi1, zi), u), lerp(hash2(xi, zi1), hash2(xi1, zi1), u), v);
}

function valueNoise3(x: number, y: number, z: number, seed: number): number {
  const xi = Math.floor(x), yi = Math.floor(y), zi = f(Math.floor(z) + seed);
  const u = smooth(f(x - xi)), v = smooth(f(y - yi)), w = smooth(f(z - Math.floor(z)));
  const xi1 = f(xi + 1), yi1 = f(yi + 1), zi1 = f(zi + 1);
  const plane = (zz: number) =>
    lerp(lerp(hash3(xi, yi, zz), hash3(xi1, yi, zz), u), lerp(hash3(xi, yi1, zz), hash3(xi1, yi1, zz), u), v);
  return lerp(plane(zi), plane(zi1), w);
}

function fbm2(x: number, z: number, seed: number, baseScale: number, octaves: number): number {
  let sum = 0, amp = 1, norm = 0, scale = baseScale;
  for (let o = 0; o < octaves; o++) {
    sum = f(sum + f(valueNoise2(f(x / scale), f(z / scale), seed + o * 101) * amp));
    norm += amp;
    amp *= 0.5;
    scale *= 0.5;
  }
  return f(sum / norm);
}

/** Dave Hoskins' hash12, as tf/noise.ts hash12 (coordinates wrapped to [0, 4096)). */
function hash12(x: number, z: number, seed: number): number {
  const wrap = (t: number) => f(t - f(Math.floor(f(t / 4096)) * 4096));
  const a = fract(f(f(wrap(x) + f(seed)) * f(0.1031))), b = fract(f(f(wrap(z) + f(seed * 0.618)) * f(0.1031)));
  const c = f(33.33);
  const d = f(f(f(a * f(b + c)) + f(b * f(a + c))) + f(a * f(a + c)));
  return fract(f(f(f(a + b) + f(d * 2)) * f(a + d)));
}

/** The ground's height (its top block's y) at world column (wx, wz): tf/worldgen.ts terrainHeight. */
export function terrainHeightAt(wx: number, wz: number, seed = DEFAULT_SEED): number {
  const continents = fbm2(wx, wz, seed, 128, 3), hills = fbm2(wx, wz, seed + 17, 32, 4);
  return Math.floor(f(f(f(f(continents - 0.5) * 36) + f(f(hills - 0.5) * 14)) + SEA_LEVEL + 2));
}

/** Ground heights on the far terrain's grid (as world/farTerrain.ts farHeights, without TF.js). */
export async function farHeightsJs(axis: number[], cx: number, cz: number, seed?: number): Promise<Float32Array> {
  const n = axis.length, out = new Float32Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) out[j * n + i] = terrainHeightAt(cx + axis[i], cz + axis[j], seed);
  return out;
}

/** One chunk's cells ([y][z][x], as CellStore.writeChunk takes them). */
export function generateChunkJs({ cx, cz }: ChunkCoord, seed = DEFAULT_SEED): Int32Array {
  const out = new Int32Array(CHUNK_VOLUME);
  const roll = (x: number, z: number, s1: number, s2: number) =>
    hash12(x, z, s1) < f(Math.sqrt(WHEAT_PATCH_CHANCE)) && hash12(x, z, s2) < f(Math.sqrt(WHEAT_PATCH_CHANCE));
  for (let z = 0; z < CHUNK_SIZE; z++) {
    for (let x = 0; x < CHUNK_SIZE; x++) {
      const wx = cx * CHUNK_SIZE + x, wz = cz * CHUNK_SIZE + z;
      // The column: ground height, then grass, wheat, sand and gravel.
      const height = terrainHeightAt(wx, wz, seed);
      const dry = height >= SEA_LEVEL;
      const sandy = height >= SEA_LEVEL - SAND_BELOW && height <= SEA_LEVEL + SAND_ABOVE;
      const gravelly = height < SEA_LEVEL - SAND_BELOW;
      const grassy = dry && !sandy;
      const wheat = WILD_WHEAT && grassy && roll(Math.floor(wx / 4), Math.floor(wz / 4), (seed * 3) % 983, (seed * 11) % 977 + 0.25)
        && hash12(wx, wz, (seed * 13) % 971 + 0.75) < f(0.6);
      for (let y = 0; y < CHUNK_HEIGHT; y++) {
        const ground = y <= height, dirt = ground && y > height - 3, water = y > height && y <= SEA_LEVEL;
        // Caves: only well under the surface, so the noise is only worked out there.
        let cave = false;
        if (y > 0 && y < height - 4) {
          cave = f(valueNoise3(f(wx / 14), f(y / 9), f(wz / 14), seed + 999)
            + f(valueNoise3(f(wx / 6), f(y / 6), f(wz / 6), seed + 555) * f(0.35))) > f(0.98);
        }
        const lava = cave && y <= 10;
        const grass = y === height && grassy;
        const plant = y === height + 1 && wheat;
        const solid = ground && !cave;
        const stone = (solid && !dirt) || y === 0;
        const diamond = stone && y > 0 && y <= DIAMOND_MAX_Y && valueNoise3(f(wx / 3), f(y / 3), f(wz / 3), seed + 333) > f(DIAMOND_THRESHOLD);
        let c = cell(Block.Air);
        if (stone) c = diamond ? cell(Block.Diamond) : cell(Block.Stone);
        else if (solid && dirt && y > 0) c = grass ? cell(Block.Grass) : sandy ? SAND : gravelly ? GRAVEL : cell(Block.Dirt);
        else if (water && y > 0) c = cell(Block.Water, SOURCE_LEVEL);
        else if (lava) c = cell(Block.Lava, SOURCE_LEVEL);
        else if (plant) c = cell(Block.Wheat, WHEAT_RIPE);
        out[blockIndex(x, y, z)] = c;
      }
    }
  }
  return out;
}

/** Generates chunks in a Web Worker (world/worldgenWorker.ts), so the game keeps its frame rate. */
export class WorldgenWorker {
  private readonly worker = new Worker(new URL('./worldgenWorker.ts', import.meta.url), { type: 'module' });
  private readonly waiting = new Map<number, { resolve: (c: Int32Array[]) => void; reject: (e: unknown) => void }>();
  private next = 0;

  constructor() {
    this.worker.onmessage = (e: MessageEvent<{ id: number; chunks: Int32Array[] }>) => {
      this.waiting.get(e.data.id)?.resolve(e.data.chunks);
      this.waiting.delete(e.data.id);
    };
    this.worker.onerror = (e) => {
      for (const w of this.waiting.values()) w.reject(new Error(`World generation worker: ${e.message}`));
      this.waiting.clear();
    };
  }

  /** The cells of these chunks ([y][z][x] each). */
  generate(coords: ChunkCoord[], seed?: number): Promise<Int32Array[]> {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.worker.postMessage({ id, coords, seed });
    });
  }
}
