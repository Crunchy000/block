import { chunkKey } from '../constants';

/**
 * - ghost:  loaded as read-only neighbour data (the halo around the active area)
 * - active: simulated by block updates and drawn
 */
export type ChunkState = 'ghost' | 'active';

/** A chunk in the halo: bookkeeping only. Its cells live in the store (GPU memory in the game), in `slot`. */
export class Chunk {
  readonly key: string;
  state: ChunkState = 'ghost';
  /** Its cells are in the store (generated, or restored after an earlier visit). */
  loaded = false;
  /** Generating or restoring. */
  loading = false;
  /** Bumped whenever its cells (or the neighbouring cells its mesh depends on) change; remeshed when it differs from meshedVersion. */
  version = 0;
  meshedVersion = -1;
  /** Where its mesh goes while it's active (-1 otherwise). */
  meshSlot = -1;
  /** Edited or simulated since generation: saved when it leaves the halo, restored when it comes back. */
  modified = false;

  constructor(readonly cx: number, readonly cz: number, readonly slot: number) {
    this.key = chunkKey(cx, cz);
  }
}
