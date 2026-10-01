import { CHUNK_VOLUME, chunkKey } from '../constants';

/**
 * - ghost:  generated, kept as read-only neighbour data (halo around the active area)
 * - active: simulated by block updates and rendered
 */
export type ChunkState = 'ghost' | 'active';

export class Chunk {
  readonly key: string;
  state: ChunkState = 'ghost';
  /** Bumped whenever the data changes; the renderer remeshes when it differs from meshedVersion. */
  version = 0;
  meshedVersion = -1;
  /** Edited or simulated since generation: keep in memory when it leaves the halo. */
  modified = false;
  /** Contains plants (grass, primed dirt, wheat), so block updates need the plant rules here. */
  plants = false;

  constructor(readonly cx: number, readonly cz: number, readonly data: Uint8Array = new Uint8Array(CHUNK_VOLUME)) {
    this.key = chunkKey(cx, cz);
  }
}
