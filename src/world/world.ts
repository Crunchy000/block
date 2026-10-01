import {
  ACTIVE_RADIUS, Block, CHUNK_HEIGHT, CHUNK_SIZE, GHOST_RADIUS, blockIndex, cellType, chunkKey,
} from '../constants';
import { Chunk } from './chunk';
import type { ChunkCoord } from '../tf/worldgen';

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
    this.touchNeighbours(coord.cx, coord.cz);
  }

  /** True once every chunk in the active area and ghost halo is loaded. */
  haloReady(): boolean {
    const { cx, cz, ghostRadius: r } = this.window;
    for (let dz = -r; dz <= r; dz++)
      for (let dx = -r; dx <= r; dx++) if (!this.chunks.has(chunkKey(cx + dx, cz + dz))) return false;
    return true;
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
    chunk.modified = true;
    chunk.version++;
    // Faces on the chunk border belong to the neighbour's mesh too.
    if (lx === 0) this.touch(cx - 1, cz);
    if (lx === CHUNK_SIZE - 1) this.touch(cx + 1, cz);
    if (lz === 0) this.touch(cx, cz - 1);
    if (lz === CHUNK_SIZE - 1) this.touch(cx, cz + 1);
  }

  flushPendingEdits(): void {
    const edits = this.pendingEdits;
    this.pendingEdits = [];
    for (const [x, y, z, v] of edits) this.setCell(x, y, z, v);
  }

  touch(cx: number, cz: number): void {
    const c = this.getChunk(cx, cz);
    if (c) c.version++;
  }

  touchNeighbours(cx: number, cz: number): void {
    this.touch(cx - 1, cz);
    this.touch(cx + 1, cz);
    this.touch(cx, cz - 1);
    this.touch(cx, cz + 1);
  }

  activeChunks(): Chunk[] {
    return [...this.chunks.values()].filter((c) => c.state === 'active');
  }

  ghostChunks(): Chunk[] {
    return [...this.chunks.values()].filter((c) => c.state === 'ghost' && this.distance(c.cx, c.cz) <= this.ghostRadius);
  }
}
