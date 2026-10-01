import type * as tf from '@tensorflow/tfjs';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, PRIMED_DIRT, type PlantRates } from '../constants';
import type { RayHit } from '../player/raycast';
import type { ChunkFaces } from '../render/mesher';
import type { GenJob } from './gpuWorldgen';

/**
 * Where the world's cells live, and the operations on them. The game keeps them on the
 * GPU (GpuStore): nothing but a few bytes of flags per tick comes back to the CPU.
 * CpuStore holds them in a typed array and runs the reference code: the oracle the
 * GPU tests compare against, the store the Node tests run on, and the fallback for a
 * GPU that fails the startup check.
 *
 * Cells are kept per chunk in slots of a ring around the player: `ring` x `ring`
 * slots, a chunk at (cx, cz) in slot slotOf(cx, cz, ring), each slot a chunk's
 * CHUNK_VOLUME cells in the chunk layout ([y][z][x]). The ring is as wide as the
 * ghost halo, so every loaded chunk has its own slot, and moving reuses the slots of
 * chunks that drop out.
 */
export type { GenJob };

export interface CellStore {
  /** Slots per side of the ring. */
  readonly ring: number;
  /** Chunks meshed per frame at most (meshing on the CPU is slow). */
  readonly meshBudget: number;
  /** Which chunk a slot holds, and whether its cells are there yet (picking reads only loaded chunks). */
  setSlot(slot: number, cx: number, cz: number, loaded: boolean): void;
  writeChunk(slot: number, cells: ArrayLike<number>): void;
  writeCell(slot: number, index: number, value: number): void;
  /** The chunk's cells as they are once everything issued so far has run. */
  readChunk(slot: number): Promise<Int32Array>;
  /** Generated chunks [N, H, 16, 16] (int32), ready to be written into slots. */
  stage(cells: tf.Tensor): Promise<StagedChunks>;
  /**
   * Generate chunks straight into their slots, if this store can (the GPU store: one
   * compute shader, see sim/gpuWorldgen.ts). Otherwise chunks come from TF.js via stage().
   */
  generate?(chunks: GenJob[], seed?: number): void;
  /**
   * One block-update tick for some chunks. `jobs` holds AROUND slots per chunk: the
   * 3x3 chunks around it, row by row (z, then x), the chunk itself in the middle.
   * Every chunk is updated from the state before the tick, then written back.
   * Resolves to a flags word per chunk (TickFlag).
   */
  tick(jobs: Uint32Array, seed: number, rates: PlantRates): Promise<Uint32Array>;
  /** The first solid block or plant along a ray (see player/raycast.ts). */
  raycast(origin: readonly number[], dir: readonly number[], maxDist: number): Promise<RayHit | null>;
  /** Mesh chunks into their mesh slots (see render/mesher.ts for the face records). */
  mesh(jobs: MeshJob[], target: MeshTarget | undefined): Promise<void>;
}

/** Generated chunks waiting to be written into slots. */
export interface StagedChunks {
  /** Write generated chunk `index` of the batch into `slot`. */
  write(slot: number, index: number): void;
  /** Done writing. */
  release(): void;
}

/** Where meshes go: GPU mesh slots (MeshPool), or CPU-built buffers in safe mode (ClassicMeshes). */
export interface MeshTarget {
  upload(slot: number, cx: number, cz: number, faces: ChunkFaces): void;
}

export interface MeshJob {
  /** The AROUND slots around the chunk (see CellStore.tick). Meshing reads the four side neighbours. */
  around: ArrayLike<number>;
  meshSlot: number;
  cx: number;
  cz: number;
  /** Still wanted? (Checked before a mesh made asynchronously is kept.) */
  current?: () => boolean;
}

/** Slots per chunk in a tick job: the chunk and its eight neighbours. */
export const AROUND = 9;
/** Index of the chunk itself among its AROUND slots. */
export const SELF = 4;

/** What a tick did to a chunk: bits of the flags word CellStore.tick returns for it. */
export const enum TickFlag {
  /** Bits 0-7: the Border bits (world.ts) of cells that changed in a way that matters. */
  Borders = 0xff,
  /** A cell changed in a way that matters: anything but priming (see PRIMED_DIRT). */
  Changed = 1 << 8,
  /** Plants that may still change by chance (primed dirt, unripe wheat): the chunk isn't settled. */
  Growing = 1 << 9,
}

export const ringSize = (ghostRadius: number) => 2 * ghostRadius + 1;
const mod = (a: number, n: number) => ((a % n) + n) % n;
/** The ring slot of chunk (cx, cz). */
export const slotOf = (cx: number, cz: number, ring: number) => mod(cz, ring) * ring + mod(cx, ring);

/**
 * The cell at chunk-local (x, y, z) of the chunk in the middle of `around`, for x and z
 * from -1 to CHUNK_SIZE (a block into the neighbouring chunks); `cells` holds every slot.
 * Above the top reads as air, below y = 0 as bedrock.
 */
export function cellNear(cells: ArrayLike<number>, around: ArrayLike<number>, x: number, y: number, z: number): number {
  if (y >= CHUNK_HEIGHT) return Block.Air;
  if (y < 0) return Block.Stone;
  const gx = x < 0 ? 0 : x >= CHUNK_SIZE ? 2 : 1, gz = z < 0 ? 0 : z >= CHUNK_SIZE ? 2 : 1;
  const lx = x - (gx - 1) * CHUNK_SIZE, lz = z - (gz - 1) * CHUNK_SIZE;
  return cells[around[gz * 3 + gx] * CHUNK_VOLUME + (y * CHUNK_SIZE + lz) * CHUNK_SIZE + lx];
}

/** Priming is bookkeeping: it changes neither how a block looks nor how its neighbours behave. */
export const unprimed = (c: number) => (c === PRIMED_DIRT ? Block.Dirt : c);
