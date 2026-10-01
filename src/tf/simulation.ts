import * as tf from '@tensorflow/tfjs';
import { CHUNK_HEIGHT, CHUNK_SIZE, blockIndex } from '../constants';
import type { World } from '../world/world';
import { blockUpdateStep } from './blockUpdate';

/**
 * Runs block updates for the active area.
 *
 * Each tick the active chunks *and* their ghost halo are packed into one
 * [H, Z, X] tensor. The ghost ring supplies boundary values so fluids at the
 * edge of the active area see real neighbours; after stepping, only the
 * active chunks are written back and the ghost results are discarded.
 */
export class Simulation {
  ticks = 0;
  lastTickMs = 0;
  lastChangedChunks = 0;

  constructor(private readonly world: World, readonly stepsPerTick = 1) {}

  async tick(): Promise<boolean> {
    const world = this.world;
    if (world.locked || !world.haloReady()) return false;
    const t0 = performance.now();
    world.locked = true;
    try {
      const { cx: ccx, cz: ccz, ghostRadius: r } = world.window;
      const n = 2 * r + 1, W = n * CHUNK_SIZE, H = CHUNK_HEIGHT;
      const region = new Int32Array(H * W * W);

      // Pack chunks into the region: region index = (y * W + gz) * W + gx.
      for (let dz = -r; dz <= r; dz++) {
        for (let dx = -r; dx <= r; dx++) {
          const chunk = world.getChunk(ccx + dx, ccz + dz)!;
          const bx = (dx + r) * CHUNK_SIZE, bz = (dz + r) * CHUNK_SIZE;
          for (let y = 0; y < H; y++) {
            for (let z = 0; z < CHUNK_SIZE; z++) {
              const src = blockIndex(0, y, z);
              const dst = (y * W + bz + z) * W + bx;
              for (let x = 0; x < CHUNK_SIZE; x++) region[dst + x] = chunk.data[src + x];
            }
          }
        }
      }

      const out = tf.tidy(() => {
        let t = tf.tensor3d(region, [H, W, W], 'int32');
        for (let i = 0; i < this.stepsPerTick; i++) t = blockUpdateStep(t);
        return t;
      });
      const next = (await out.data()) as Int32Array;
      out.dispose();

      // Unpack only the active chunks; ghost cells are boundary data and are discarded.
      let changed = 0;
      const ar = world.window.activeRadius;
      for (let dz = -ar; dz <= ar; dz++) {
        for (let dx = -ar; dx <= ar; dx++) {
          const chunk = world.getChunk(ccx + dx, ccz + dz);
          if (!chunk) continue;
          const bx = (dx + r) * CHUNK_SIZE, bz = (dz + r) * CHUNK_SIZE;
          let dirty = false;
          for (let y = 0; y < H; y++) {
            for (let z = 0; z < CHUNK_SIZE; z++) {
              const dst = blockIndex(0, y, z);
              const src = (y * W + bz + z) * W + bx;
              for (let x = 0; x < CHUNK_SIZE; x++) {
                const v = next[src + x];
                if (chunk.data[dst + x] !== v) {
                  chunk.data[dst + x] = v;
                  dirty = true;
                }
              }
            }
          }
          if (dirty) {
            changed++;
            chunk.modified = true;
            chunk.version++;
            world.touchNeighbours(chunk.cx, chunk.cz);
          }
        }
      }
      this.lastChangedChunks = changed;
      this.ticks++;
      return true;
    } finally {
      world.locked = false;
      world.flushPendingEdits();
      this.lastTickMs = performance.now() - t0;
    }
  }
}
