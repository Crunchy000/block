import * as tf from '@tensorflow/tfjs';
import { CHUNK_HEIGHT, CHUNK_SIZE, blockIndex } from '../constants';
import { Border, type World } from '../world/world';
import type { Chunk } from '../world/chunk';
import { blockUpdateStep } from './blockUpdate';

/**
 * Runs block updates for the active area.
 *
 * Only chunks that can change are simulated: the world keeps an "awake" set
 * (chunks that were edited, freshly loaded, or next to a change). Each tick
 * packs the bounding box of the awake chunks plus a one-chunk ghost border
 * into one [H, Z, X] tensor. The border supplies real neighbour cells so
 * fluids at the edge flow correctly; only the interior is read back and
 * written to the world. When nothing is awake the tick is skipped entirely.
 */
export class Simulation {
  ticks = 0;
  /** Wall time of the last tick, including waiting for the GPU. */
  lastTickMs = 0;
  /** Main-thread time of the last tick (packing, dispatch, unpacking). */
  lastCpuMs = 0;
  lastChangedChunks = 0;
  /** Size of the last simulated region in chunks, ghost border included. */
  lastRegion: [number, number] = [0, 0];
  asleep = false;

  constructor(private readonly world: World) {}

  /** Returns true if a tick ran (false when not ready, busy, or nothing is awake). */
  async tick(): Promise<boolean> {
    const world = this.world;
    if (world.locked || !world.haloReady()) return false;
    const awake = world.takeAwake();
    this.asleep = awake.length === 0;
    if (this.asleep) return false;

    const t0 = performance.now();
    world.locked = true;
    let cpuMs = 0;
    try {
      // Region: bounding box of the awake chunks, grown by one chunk of ghost border.
      // Awake chunks are active, so the border is always inside the loaded halo.
      let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
      for (const c of awake) {
        x0 = Math.min(x0, c.cx); x1 = Math.max(x1, c.cx);
        z0 = Math.min(z0, c.cz); z1 = Math.max(z1, c.cz);
      }
      x0--; z0--; x1++; z1++;
      const S = CHUNK_SIZE, H = CHUNK_HEIGHT;
      const nx = x1 - x0 + 1, nz = z1 - z0 + 1, W = nx * S, D = nz * S;
      this.lastRegion = [nx, nz];

      // Pack chunk rows into the region: region index = (y * D + gz) * W + gx.
      const region = new Int32Array(H * D * W);
      for (let iz = 0; iz < nz; iz++) {
        for (let ix = 0; ix < nx; ix++) {
          const data = world.getChunk(x0 + ix, z0 + iz)!.data;
          for (let y = 0; y < H; y++) {
            for (let z = 0; z < S; z++) {
              const src = blockIndex(0, y, z);
              region.set(data.subarray(src, src + S), (y * D + iz * S + z) * W + ix * S);
            }
          }
        }
      }

      const interior = tf.tidy(() => {
        const next = blockUpdateStep(tf.tensor3d(region, [H, D, W], 'int32'));
        return next.slice([0, S, S], [H, D - 2 * S, W - 2 * S]);
      });
      cpuMs += performance.now() - t0;
      let next: Int32Array;
      try {
        next = (await interior.data()) as Int32Array;
      } finally {
        interior.dispose();
      }

      const t1 = performance.now();
      this.lastChangedChunks = this.writeBack(next, x0 + 1, z0 + 1, nx - 2, nz - 2);
      cpuMs += performance.now() - t1;
      this.ticks++;
      return true;
    } catch (e) {
      world.wake(awake); // retry these next time (e.g. after a backend fallback)
      throw e;
    } finally {
      world.locked = false;
      world.flushPendingEdits();
      this.lastTickMs = performance.now() - t0;
      this.lastCpuMs = cpuMs;
    }
  }

  /** Copy the stepped interior back into its chunks. Returns how many chunks changed. */
  private writeBack(next: Int32Array, cx0: number, cz0: number, nx: number, nz: number): number {
    const S = CHUNK_SIZE, H = CHUNK_HEIGHT, W = nx * S, D = nz * S;
    let changed = 0;
    for (let iz = 0; iz < nz; iz++) {
      for (let ix = 0; ix < nx; ix++) {
        const chunk: Chunk | undefined = this.world.getChunk(cx0 + ix, cz0 + iz);
        if (!chunk) continue;
        let dirty = false, borders = Border.None;
        for (let y = 0; y < H; y++) {
          for (let z = 0; z < S; z++) {
            const dst = blockIndex(0, y, z);
            const src = (y * D + iz * S + z) * W + ix * S;
            for (let x = 0; x < S; x++) {
              const v = next[src + x];
              if (chunk.data[dst + x] === v) continue;
              chunk.data[dst + x] = v;
              dirty = true;
              if (x === 0) borders |= Border.West;
              else if (x === S - 1) borders |= Border.East;
              if (z === 0) borders |= Border.North;
              else if (z === S - 1) borders |= Border.South;
            }
          }
        }
        if (dirty) {
          changed++;
          this.world.markChanged(chunk, borders);
        }
      }
    }
    return changed;
  }
}
