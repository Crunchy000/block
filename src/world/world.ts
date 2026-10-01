import {
  ACTIVE_RADIUS, Block, CHUNK_HEIGHT, CHUNK_SIZE, GHOST_RADIUS, blockIndex, cellType, chunkKey,
} from '../constants';
import { Chunk } from './chunk';
import type { ChunkCoord } from '../tf/worldgen';

/** Bit flags for which chunk borders a change touched. */
export const enum Border {
  None = 0,
  West = 1, // x = 0
  East = 2, // x = CHUNK_SIZE - 1
  North = 4, // z = 0
  South = 8, // z = CHUNK_SIZE - 1
}

export interface ChunkWindow {
  /** Chunk the window is centred on. */
  cx: number;
  cz: number;
  activeRadius: number;
  ghostRadius: number;
}

/** Holds every loaded chunk and tracks which are active vs ghost around the player. */
export class World {
  readonly chunks = new Map<string, Chunk>();
  window: ChunkWindow;
  /** Set while a block-update tick is in flight; edits are queued until it lands. */
  locked = false;
  private pendingEdits: Array<[number, number, number, number]> = [];
  private pendingGen = new Set<string>();
  /**
   * Chunks whose cells may change on the next block-update tick. A chunk that
   * didn't change, with neighbours that didn't change, is a fixed point of the
   * update rules, so it sleeps until an edit or a neighbour wakes it.
   */
  private awake = new Set<string>();

  constructor(readonly activeRadius = ACTIVE_RADIUS, readonly ghostRadius = GHOST_RADIUS) {
    this.window = { cx: 0, cz: 0, activeRadius, ghostRadius };
  }

  getChunk(cx: number, cz: number): Chunk | undefined {
    return this.chunks.get(chunkKey(cx, cz));
  }

  /** Re-centre the active area + ghost halo on the chunk containing world position (x, z). */
  recenter(x: number, z: number): void {
    const cx = Math.floor(x / CHUNK_SIZE), cz = Math.floor(z / CHUNK_SIZE);
    this.window = { cx, cz, activeRadius: this.activeRadius, ghostRadius: this.ghostRadius };
    for (const chunk of this.chunks.values()) {
      const d = this.distance(chunk.cx, chunk.cz);
      if (d > this.ghostRadius) {
        // Out of the halo: drop pristine chunks (they regenerate identically), keep edited ones.
        if (!chunk.modified) this.chunks.delete(chunk.key);
        else chunk.state = 'ghost';
        continue;
      }
      const state = d <= this.activeRadius ? 'active' : 'ghost';
      if (state !== chunk.state) {
        chunk.state = state;
        // Becoming active needs a mesh; becoming ghost drops it (renderer checks state).
        chunk.meshedVersion = -1;
        // A ghost was frozen; once active it may have updates to catch up on.
        if (state === 'active') this.awake.add(chunk.key);
      }
    }
  }

  /** Chebyshev distance in chunks from the window centre. */
  distance(cx: number, cz: number): number {
    return Math.max(Math.abs(cx - this.window.cx), Math.abs(cz - this.window.cz));
  }

  /** Chunks in the halo that still need generating, nearest first. */
  missingChunks(limit: number): ChunkCoord[] {
    const out: Array<ChunkCoord & { d: number }> = [];
    const { cx, cz, ghostRadius: r } = this.window;
    for (let dz = -r; dz <= r; dz++) {
      for (let dx = -r; dx <= r; dx++) {
        const key = chunkKey(cx + dx, cz + dz);
        if (this.chunks.has(key) || this.pendingGen.has(key)) continue;
        out.push({ cx: cx + dx, cz: cz + dz, d: dx * dx + dz * dz });
      }
    }
    out.sort((a, b) => a.d - b.d);
    return out.slice(0, limit).map(({ cx, cz }) => ({ cx, cz }));
  }

  markGenerating(coords: ChunkCoord[]): void {
    for (const c of coords) this.pendingGen.add(chunkKey(c.cx, c.cz));
  }

  /** Clear the in-flight flag (e.g. after a failed batch) so the chunks are retried. */
  markGenerated(coords: ChunkCoord[]): void {
    for (const c of coords) this.pendingGen.delete(chunkKey(c.cx, c.cz));
  }

  addGenerated(coord: ChunkCoord, data: Uint8Array): void {
    const key = chunkKey(coord.cx, coord.cz);
    this.pendingGen.delete(key);
    if (this.chunks.has(key) || this.distance(coord.cx, coord.cz) > this.ghostRadius) return;
    const chunk = new Chunk(coord.cx, coord.cz, data);
    chunk.state = this.distance(coord.cx, coord.cz) <= this.activeRadius ? 'active' : 'ghost';
    this.chunks.set(key, chunk);
    // Fresh terrain may not be settled, and its neighbours now have new border data.
    this.awake.add(key);
    for (const [dx, dz] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) this.neighbourChanged(coord.cx + dx, coord.cz + dz);
  }

  /** True once every chunk in the active area and ghost halo is loaded. */
  haloReady(): boolean {
    const { cx, cz, ghostRadius: r } = this.window;
    for (let dz = -r; dz <= r; dz++)
      for (let dx = -r; dx <= r; dx++) if (!this.chunks.has(chunkKey(cx + dx, cz + dz))) return false;
    return true;
  }

  /** How many chunks of the active area + ghost halo are loaded. */
  haloProgress(): { loaded: number; total: number } {
    const { cx, cz, ghostRadius: r } = this.window;
    let loaded = 0;
    for (let dz = -r; dz <= r; dz++)
      for (let dx = -r; dx <= r; dx++) if (this.chunks.has(chunkKey(cx + dx, cz + dz))) loaded++;
    return { loaded, total: (2 * r + 1) ** 2 };
  }

  getCell(x: number, y: number, z: number): number {
    if (y < 0) return Block.Stone;
    if (y >= CHUNK_HEIGHT) return Block.Air;
    const cx = Math.floor(x / CHUNK_SIZE), cz = Math.floor(z / CHUNK_SIZE);
    const chunk = this.getChunk(cx, cz);
    if (!chunk) return Block.Air;
    return chunk.data[blockIndex(x - cx * CHUNK_SIZE, y, z - cz * CHUNK_SIZE)];
  }

  getBlock(x: number, y: number, z: number): Block {
    return cellType(this.getCell(x, y, z));
  }

  /** Set a cell (type + level). Queued if a block-update tick is in flight. */
  setCell(x: number, y: number, z: number, value: number): void {
    if (this.locked) {
      this.pendingEdits.push([x, y, z, value]);
      return;
    }
    if (y < 0 || y >= CHUNK_HEIGHT) return;
    const cx = Math.floor(x / CHUNK_SIZE), cz = Math.floor(z / CHUNK_SIZE);
    const chunk = this.getChunk(cx, cz);
    if (!chunk) return;
    const lx = x - cx * CHUNK_SIZE, lz = z - cz * CHUNK_SIZE;
    chunk.data[blockIndex(lx, y, lz)] = value;
    this.markChanged(chunk, borderOf(lx, lz));
  }

  /**
   * Record that cells in `chunk` changed: it remeshes and wakes for block updates.
   * Neighbours across the touched borders do too (their faces and fluid flow
   * depend on the cells next to them).
   */
  markChanged(chunk: Chunk, borders: number): void {
    chunk.version++;
    chunk.modified = true;
    this.awake.add(chunk.key);
    if (borders & Border.West) this.neighbourChanged(chunk.cx - 1, chunk.cz);
    if (borders & Border.East) this.neighbourChanged(chunk.cx + 1, chunk.cz);
    if (borders & Border.North) this.neighbourChanged(chunk.cx, chunk.cz - 1);
    if (borders & Border.South) this.neighbourChanged(chunk.cx, chunk.cz + 1);
  }

  private neighbourChanged(cx: number, cz: number): void {
    const c = this.getChunk(cx, cz);
    if (!c) return;
    c.version++;
    this.awake.add(c.key);
  }

  /** Active chunks that need a block-update tick. Clears the awake set. */
  takeAwake(): Chunk[] {
    const out: Chunk[] = [];
    for (const key of this.awake) {
      const c = this.chunks.get(key);
      if (c?.state === 'active') out.push(c);
    }
    this.awake.clear();
    return out;
  }

  /** Put chunks back in the awake set (e.g. after a failed tick). */
  wake(chunks: Chunk[]): void {
    for (const c of chunks) this.awake.add(c.key);
  }

  awakeCount(): number {
    let n = 0;
    for (const key of this.awake) if (this.chunks.get(key)?.state === 'active') n++;
    return n;
  }

  flushPendingEdits(): void {
    const edits = this.pendingEdits;
    this.pendingEdits = [];
    for (const [x, y, z, v] of edits) this.setCell(x, y, z, v);
  }

  activeChunks(): Chunk[] {
    return [...this.chunks.values()].filter((c) => c.state === 'active');
  }

  ghostChunks(): Chunk[] {
    return [...this.chunks.values()].filter((c) => c.state === 'ghost' && this.distance(c.cx, c.cz) <= this.ghostRadius);
  }
}

/** Which chunk borders the local column (lx, lz) lies on. */
export function borderOf(lx: number, lz: number): number {
  return (lx === 0 ? Border.West : 0) | (lx === CHUNK_SIZE - 1 ? Border.East : 0)
    | (lz === 0 ? Border.North : 0) | (lz === CHUNK_SIZE - 1 ? Border.South : 0);
}
