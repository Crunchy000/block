import type * as tf from '@tensorflow/tfjs';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, cellType, isGrowing, type PlantRates } from '../constants';
import { raycast, type RayHit } from '../player/raycast';
import { meshFaces, type ChunkFaces } from '../render/mesher';
import type { MeshPool } from '../render/meshPool';
import { blockUpdateReference, cellRandom } from '../tf/blockUpdateReference';
import { borderOf } from '../world/world';
import {
  AROUND, SELF, TickFlag, cellNear, slotOf, unprimed, type CellStore, type MeshJob, type StagedChunks,
} from './store';

const P = CHUNK_SIZE + 2;

/**
 * Cells in a typed array, updated by the reference rules (blockUpdateReference.ts) and
 * meshed by the JS mesher. What GpuStore does, one cell at a time on the CPU.
 */
export class CpuStore implements CellStore {
  readonly cells: Int32Array;
  readonly meshBudget = 4;
  /** Per slot: chunk x, chunk z, loaded (1) or not. */
  private readonly slotInfo: Int32Array;

  constructor(readonly ring: number) {
    this.cells = new Int32Array(ring * ring * CHUNK_VOLUME);
    this.slotInfo = new Int32Array(ring * ring * 3);
  }

  setSlot(slot: number, cx: number, cz: number, loaded: boolean): void {
    this.slotInfo.set([cx, cz, loaded ? 1 : 0], slot * 3);
  }

  writeChunk(slot: number, cells: ArrayLike<number>): void {
    this.cells.set(cells, slot * CHUNK_VOLUME);
  }

  writeCell(slot: number, index: number, value: number): void {
    this.cells[slot * CHUNK_VOLUME + index] = value;
  }

  readChunk(slot: number): Promise<Int32Array> {
    return Promise.resolve(this.cells.slice(slot * CHUNK_VOLUME, (slot + 1) * CHUNK_VOLUME));
  }

  async stage(cells: tf.Tensor): Promise<StagedChunks> {
    const data = (await cells.data()) as Int32Array;
    return {
      write: (slot, index) => this.writeChunk(slot, data.subarray(index * CHUNK_VOLUME, (index + 1) * CHUNK_VOLUME)),
      release: () => {},
    };
  }

  tick(jobs: Uint32Array, seed: number, rates: PlantRates): Promise<Uint32Array> {
    const n = jobs.length / AROUND;
    const flags = new Uint32Array(n);
    const results: Int32Array[] = [];
    const padded = new Int32Array(CHUNK_HEIGHT * P * P), random = new Float32Array(CHUNK_HEIGHT * P * P);
    const at = (y: number, z: number, x: number) => (y * P + z + 1) * P + x + 1;
    for (let j = 0; j < n; j++) {
      const around = jobs.subarray(j * AROUND, (j + 1) * AROUND), self = around[SELF];
      // The chunk with a one-cell border from its neighbours, the plant rules' random
      // numbers for its cells (the GPU's: cellRandom of the cell's index in the store),
      // and one step of the reference rules on that.
      for (let y = 0; y < CHUNK_HEIGHT; y++)
        for (let z = -1; z <= CHUNK_SIZE; z++)
          for (let x = -1; x <= CHUNK_SIZE; x++) padded[at(y, z, x)] = cellNear(this.cells, around, x, y, z);
      for (let i = 0; i < CHUNK_VOLUME; i++) random[at(i >> 8, (i >> 4) & 15, i & 15)] = cellRandom(self * CHUNK_VOLUME + i, seed);
      const out = blockUpdateReference(padded, CHUNK_HEIGHT, P, P, random, rates);
      const next = new Int32Array(CHUNK_VOLUME);
      let f = 0;
      for (let i = 0; i < CHUNK_VOLUME; i++) {
        const x = i & 15, z = (i >> 4) & 15, v = out[at(i >> 8, z, x)], old = this.cells[self * CHUNK_VOLUME + i];
        next[i] = v;
        f |= cellFlags(old, v, x, z);
      }
      flags[j] = f;
      results.push(next);
    }
    // Every chunk stepped from the state before the tick; now write them all.
    results.forEach((next, j) => this.cells.set(next, jobs[j * AROUND + SELF] * CHUNK_VOLUME));
    return Promise.resolve(flags);
  }

  raycast(origin: readonly number[], dir: readonly number[], maxDist: number): Promise<RayHit | null> {
    return Promise.resolve(raycast((x, y, z) => cellType(this.cellAt(x, y, z)), origin, dir, maxDist));
  }

  /** The cell at a world position: what picking sees (chunks that aren't loaded read as air). */
  cellAt(x: number, y: number, z: number): number {
    if (y < 0) return Block.Stone;
    if (y >= CHUNK_HEIGHT) return Block.Air;
    const cx = Math.floor(x / CHUNK_SIZE), cz = Math.floor(z / CHUNK_SIZE), slot = slotOf(cx, cz, this.ring);
    const info = this.slotInfo.subarray(slot * 3, slot * 3 + 3);
    if (info[0] !== cx || info[1] !== cz || info[2] === 0) return Block.Air;
    return this.cells[slot * CHUNK_VOLUME + (y * CHUNK_SIZE + z - cz * CHUNK_SIZE) * CHUNK_SIZE + x - cx * CHUNK_SIZE];
  }

  /** The face records of one chunk. */
  meshChunk(around: ArrayLike<number>): ChunkFaces {
    return meshFaces((x, y, z) => cellNear(this.cells, around, x, y, z));
  }

  mesh(jobs: MeshJob[], pool: MeshPool | undefined): void {
    if (!pool) return; // nothing to draw with (Node tests)
    for (const job of jobs) pool.upload(job.meshSlot, job.cx, job.cz, this.meshChunk(job.around));
  }
}

/** The TickFlag bits for one cell going from `old` to `next` at chunk-local column (x, z). */
export function cellFlags(old: number, next: number, x: number, z: number): number {
  return (isGrowing(next) ? TickFlag.Growing : 0) | (unprimed(next) !== unprimed(old) ? TickFlag.Changed | borderOf(x, z) : 0);
}
