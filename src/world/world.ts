import {
  ACTIVE_RADIUS, Block, CHUNK_HEIGHT, CHUNK_SIZE, GHOST_RADIUS, blockIndex, chunkKey,
} from '../constants';
import type { MeshPool } from '../render/meshPool';
import { AROUND, ringSize, slotOf, type CellStore } from '../sim/store';
import type { ChunkCoord } from '../tf/worldgen';
import { Chunk, type ChunkState } from './chunk';

/** Bit flags for which chunk borders a change touched. */
export const enum Border {
  None = 0,
  West = 1, // x = 0
  East = 2, // x = CHUNK_SIZE - 1
  North = 4, // z = 0
  South = 8, // z = CHUNK_SIZE - 1
  // Corner cells: grass reaches diagonally, so these also concern the diagonal neighbour.
  NorthWest = 16,
  NorthEast = 32,
  SouthWest = 64,
  SouthEast = 128,
}

export interface ChunkWindow {
  /** Chunk the window is centred on. */
  cx: number;
  cz: number;
  activeRadius: number;
  ghostRadius: number;
}

/** An active chunk with a mesh: where its mesh is, and its centre (to sort translucent water). */
export interface ChunkDraw {
  meshSlot: number;
  center: [number, number, number];
}

/** Mesh slots for an active radius: one per active chunk. */
export const meshSlotCount = (activeRadius: number) => (2 * activeRadius + 1) ** 2;

/**
 * The chunks around the player: which are active or ghost, which are loaded, which
 * are awake, which need meshing. Bookkeeping only: the cells live in the store (on
 * the GPU in the game), one chunk per slot of a ring as wide as the halo, so moving
 * reuses the slots of the chunks that drop out.
 */
export class World {
  /** Slots per side of the store's ring. */
  readonly ring: number;
  /** The chunk in each store slot. */
  readonly slots: Array<Chunk | undefined>;
  /** Every chunk in the active area and ghost halo, loaded or not. */
  readonly chunks = new Map<string, Chunk>();
  window: ChunkWindow;
  /**
   * Chunks whose cells may change on the next block-update tick. A chunk that
   * didn't change, with neighbours that didn't change, is a fixed point of the
   * update rules, so it sleeps until an edit or a neighbour wakes it.
   */
  private awake = new Set<Chunk>();
  /** Edited chunks that left the halo, read back from the store to be restored when they come back. */
  private saved = new Map<string, Promise<Uint8Array>>();
  private freeMeshSlots: number[];

  constructor(readonly store: CellStore, readonly activeRadius = ACTIVE_RADIUS, readonly ghostRadius = GHOST_RADIUS) {
    if (ghostRadius <= activeRadius) throw new Error('the ghost halo must reach past the active area');
    this.ring = ringSize(ghostRadius);
    if (store.ring !== this.ring) throw new Error(`a store with a ${store.ring}-slot ring can't hold a ${this.ring}-chunk halo`);
    this.slots = new Array<Chunk | undefined>(this.ring * this.ring).fill(undefined);
    this.freeMeshSlots = Array.from({ length: meshSlotCount(activeRadius) }, (_, i) => i).reverse();
    this.window = { cx: 0, cz: 0, activeRadius, ghostRadius };
    this.update();
  }

  getChunk(cx: number, cz: number): Chunk | undefined {
    return this.chunks.get(chunkKey(cx, cz));
  }

  /** Still in the halo (and so still in its slot)? */
  isResident(chunk: Chunk): boolean {
    return this.slots[chunk.slot] === chunk;
  }

  /** Re-centre the active area + ghost halo on the chunk containing world position (x, z). */
  recenter(x: number, z: number): void {
    const cx = Math.floor(x / CHUNK_SIZE), cz = Math.floor(z / CHUNK_SIZE);
    if (cx === this.window.cx && cz === this.window.cz) return;
    this.window = { cx, cz, activeRadius: this.activeRadius, ghostRadius: this.ghostRadius };
    this.update();
  }

  private update(): void {
    for (const chunk of [...this.chunks.values()]) {
      if (this.distance(chunk.cx, chunk.cz) > this.ghostRadius) this.unload(chunk);
    }
    const { cx, cz, ghostRadius: r } = this.window;
    for (let dz = -r; dz <= r; dz++)
      for (let dx = -r; dx <= r; dx++) if (!this.getChunk(cx + dx, cz + dz)) this.enter(cx + dx, cz + dz);
    // Demote first: chunks becoming active take the mesh slots of chunks becoming ghosts.
    const target = (c: Chunk): ChunkState => (this.distance(c.cx, c.cz) <= this.activeRadius ? 'active' : 'ghost');
    for (const chunk of this.chunks.values()) if (target(chunk) === 'ghost') this.setState(chunk, 'ghost');
    for (const chunk of this.chunks.values()) if (target(chunk) === 'active') this.setState(chunk, 'active');
  }

  private enter(cx: number, cz: number): void {
    const chunk = new Chunk(cx, cz, slotOf(cx, cz, this.ring));
    this.slots[chunk.slot] = chunk;
    this.chunks.set(chunk.key, chunk);
    this.store.setSlot(chunk.slot, cx, cz, false);
    const saved = this.saved.get(chunk.key);
    if (!saved) return; // generated (see missingChunks)
    // Been here before and edited: restore it rather than generating it again.
    this.saved.delete(chunk.key);
    chunk.loading = true;
    saved.then((cells) => {
      if (!this.isResident(chunk)) {
        this.saved.set(chunk.key, Promise.resolve(cells)); // left again before it arrived
        return;
      }
      this.store.writeChunk(chunk.slot, cells);
      chunk.modified = true;
      this.markLoaded(chunk);
    }, (e: unknown) => {
      console.warn(`lost the edits to chunk ${chunk.key}`, e);
      chunk.loading = false; // generate it afresh
    });
  }

  private unload(chunk: Chunk): void {
    // Pristine chunks regenerate identically; edited ones are read back and kept.
    // (The read is queued now, before anything can write the slot's next chunk.)
    if (chunk.loaded && chunk.modified) {
      this.saved.set(chunk.key, this.store.readChunk(chunk.slot).then((cells) => Uint8Array.from(cells)));
    }
    this.setState(chunk, 'ghost');
    chunk.loaded = chunk.loading = false;
    this.slots[chunk.slot] = undefined;
    this.chunks.delete(chunk.key);
    this.awake.delete(chunk);
  }

  private setState(chunk: Chunk, state: ChunkState): void {
    if (chunk.state === state) return;
    if (chunk.state === 'active') {
      this.freeMeshSlots.push(chunk.meshSlot);
      chunk.meshSlot = -1;
    }
    chunk.state = state;
    if (state === 'active') {
      chunk.meshSlot = this.freeMeshSlots.pop()!;
      chunk.meshedVersion = -1; // needs a mesh before it's drawn
      this.awake.add(chunk); // a ghost was frozen; once active it may have updates to catch up on
    }
  }

  /** Chebyshev distance in chunks from the window centre. */
  distance(cx: number, cz: number): number {
    return Math.max(Math.abs(cx - this.window.cx), Math.abs(cz - this.window.cz));
  }

  /** Chunks in the halo that need generating, nearest first. */
  missingChunks(limit: number): Chunk[] {
    const { cx, cz } = this.window;
    return [...this.chunks.values()]
      .filter((c) => !c.loaded && !c.loading)
      .sort((a, b) => (a.cx - cx) ** 2 + (a.cz - cz) ** 2 - ((b.cx - cx) ** 2 + (b.cz - cz) ** 2))
      .slice(0, limit);
  }

  /** Chunks are being generated: don't hand them out again. */
  markGenerating(chunks: Chunk[]): void {
    for (const c of chunks) c.loading = true;
  }

  /** Generation failed: hand them out again. */
  markGenerated(chunks: Chunk[]): void {
    for (const c of chunks) if (!c.loaded) c.loading = false;
  }

  /** The chunk at these coordinates if it's in the halo and waiting for its cells. */
  pendingChunk(coord: ChunkCoord): Chunk | undefined {
    const chunk = this.getChunk(coord.cx, coord.cz);
    return chunk && !chunk.loaded ? chunk : undefined;
  }

  /** Cells for a chunk from the CPU (generated there, or a test's). Ignored if the chunk isn't waiting for them. */
  addGenerated(coord: ChunkCoord, cells: ArrayLike<number>): void {
    const chunk = this.pendingChunk(coord);
    if (!chunk) return;
    this.store.writeChunk(chunk.slot, cells);
    this.markLoaded(chunk);
  }

  /** The chunk's cells are in its slot. */
  markLoaded(chunk: Chunk): void {
    chunk.loaded = true;
    chunk.loading = false;
    this.store.setSlot(chunk.slot, chunk.cx, chunk.cz, true);
    // Fresh terrain may not be settled, and its neighbours now have new border data.
    this.awake.add(chunk);
    for (let dz = -1; dz <= 1; dz++)
      for (let dx = -1; dx <= 1; dx++) if (dx || dz) this.neighbourChanged(chunk.cx + dx, chunk.cz + dz, !dx || !dz);
  }

  /** True once every chunk in the active area and ghost halo is loaded. */
  haloReady(): boolean {
    for (const c of this.chunks.values()) if (!c.loaded) return false;
    return true;
  }

  /** How many chunks of the active area + ghost halo are loaded. */
  haloProgress(): { loaded: number; total: number } {
    let loaded = 0;
    for (const c of this.chunks.values()) if (c.loaded) loaded++;
    return { loaded, total: this.chunks.size };
  }

  /** Edited chunks kept outside the halo. */
  savedCount(): number {
    return this.saved.size;
  }

  /** The store slots of the 3x3 chunks around a chunk, row by row (see CellStore.tick). */
  around(chunk: Chunk): number[] {
    const out: number[] = [];
    for (let dz = -1; dz <= 1; dz++)
      for (let dx = -1; dx <= 1; dx++) out.push(slotOf(chunk.cx + dx, chunk.cz + dz, this.ring));
    return out;
  }

  /** Set a cell (type + level) at a world position. Ignored outside loaded chunks. */
  setCell(x: number, y: number, z: number, value: number): void {
    if (y < 0 || y >= CHUNK_HEIGHT) return;
    const cx = Math.floor(x / CHUNK_SIZE), cz = Math.floor(z / CHUNK_SIZE);
    const chunk = this.getChunk(cx, cz);
    if (!chunk?.loaded) return;
    const lx = x - cx * CHUNK_SIZE, lz = z - cz * CHUNK_SIZE;
    this.store.writeCell(chunk.slot, blockIndex(lx, y, lz), value);
    this.markChanged(chunk, borderOf(lx, lz));
  }

  /** Read a cell back from the store (tests and debugging; the game never needs to). */
  async readCell(x: number, y: number, z: number): Promise<number> {
    if (y < 0) return Block.Stone;
    if (y >= CHUNK_HEIGHT) return Block.Air;
    const cx = Math.floor(x / CHUNK_SIZE), cz = Math.floor(z / CHUNK_SIZE);
    const chunk = this.getChunk(cx, cz);
    if (!chunk?.loaded) return Block.Air;
    const cells = await this.store.readChunk(chunk.slot);
    return cells[blockIndex(x - cx * CHUNK_SIZE, y, z - cz * CHUNK_SIZE)];
  }

  /**
   * Record that cells in `chunk` changed: it remeshes and wakes for block updates.
   * Neighbours across the touched borders do too (their faces and fluid flow
   * depend on the cells next to them).
   */
  markChanged(chunk: Chunk, borders: number): void {
    chunk.version++;
    chunk.modified = true;
    this.awake.add(chunk);
    if (borders & Border.West) this.neighbourChanged(chunk.cx - 1, chunk.cz);
    if (borders & Border.East) this.neighbourChanged(chunk.cx + 1, chunk.cz);
    if (borders & Border.North) this.neighbourChanged(chunk.cx, chunk.cz - 1);
    if (borders & Border.South) this.neighbourChanged(chunk.cx, chunk.cz + 1);
    // Diagonal neighbours share no faces (no remesh), but grass can spread into them.
    if (borders & Border.NorthWest) this.neighbourChanged(chunk.cx - 1, chunk.cz - 1, false);
    if (borders & Border.NorthEast) this.neighbourChanged(chunk.cx + 1, chunk.cz - 1, false);
    if (borders & Border.SouthWest) this.neighbourChanged(chunk.cx - 1, chunk.cz + 1, false);
    if (borders & Border.SouthEast) this.neighbourChanged(chunk.cx + 1, chunk.cz + 1, false);
  }

  private neighbourChanged(cx: number, cz: number, remesh = true): void {
    const c = this.getChunk(cx, cz);
    if (!c) return;
    if (remesh) c.version++;
    this.awake.add(c);
  }

  /** Active, loaded chunks that need a block-update tick. Clears the awake set. */
  takeAwake(): Chunk[] {
    const out = [...this.awake].filter((c) => c.state === 'active' && c.loaded);
    this.awake.clear();
    return out;
  }

  /** Put chunks back in the awake set (e.g. after a failed tick). */
  wake(chunks: Chunk[]): void {
    for (const c of chunks) if (this.isResident(c)) this.awake.add(c);
  }

  awakeCount(): number {
    let n = 0;
    for (const c of this.awake) if (c.state === 'active' && c.loaded) n++;
    return n;
  }

  /**
   * Mesh the active chunks whose cells changed, nearest first, up to the store's budget
   * per call. A chunk waits for its four side neighbours, so border faces cull correctly.
   * Returns how many were meshed.
   */
  remesh(pool: MeshPool | undefined): number {
    const { cx, cz } = this.window;
    const loaded = (x: number, z: number) => this.getChunk(x, z)?.loaded === true;
    const todo = this.activeChunks()
      .filter((c) => c.loaded && c.version !== c.meshedVersion
        && loaded(c.cx + 1, c.cz) && loaded(c.cx - 1, c.cz) && loaded(c.cx, c.cz + 1) && loaded(c.cx, c.cz - 1))
      .sort((a, b) => Math.hypot(a.cx - cx, a.cz - cz) - Math.hypot(b.cx - cx, b.cz - cz))
      .slice(0, this.store.meshBudget);
    if (todo.length === 0) return 0;
    this.store.mesh(todo.map((c) => ({ around: this.around(c), meshSlot: c.meshSlot, cx: c.cx, cz: c.cz })), pool);
    for (const c of todo) c.meshedVersion = c.version;
    return todo.length;
  }

  /** The active chunks that have a mesh to draw. */
  draws(): ChunkDraw[] {
    return this.activeChunks()
      .filter((c) => c.meshedVersion >= 0)
      .map((c) => ({ meshSlot: c.meshSlot, center: [(c.cx + 0.5) * CHUNK_SIZE, CHUNK_HEIGHT / 2, (c.cz + 0.5) * CHUNK_SIZE] }));
  }

  activeChunks(): Chunk[] {
    return [...this.chunks.values()].filter((c) => c.state === 'active');
  }

  ghostChunks(): Chunk[] {
    return [...this.chunks.values()].filter((c) => c.state === 'ghost');
  }
}

/** The AROUND slots of a list of chunks, one after another: the jobs for a block-update tick. */
export function tickJobs(world: World, chunks: Chunk[]): Uint32Array {
  const jobs = new Uint32Array(chunks.length * AROUND);
  chunks.forEach((c, n) => jobs.set(world.around(c), n * AROUND));
  return jobs;
}

/** Which chunk borders (and corners) the local column (lx, lz) lies on. */
export function borderOf(lx: number, lz: number): number {
  const w = lx === 0, e = lx === CHUNK_SIZE - 1, n = lz === 0, s = lz === CHUNK_SIZE - 1;
  return (w ? Border.West : 0) | (e ? Border.East : 0) | (n ? Border.North : 0) | (s ? Border.South : 0)
    | (n && w ? Border.NorthWest : 0) | (n && e ? Border.NorthEast : 0)
    | (s && w ? Border.SouthWest : 0) | (s && e ? Border.SouthEast : 0);
}
