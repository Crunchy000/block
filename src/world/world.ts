import {
  ACTIVE_RADIUS, Block, CHUNK_HEIGHT, CHUNK_SIZE, GHOST_RADIUS, blockIndex, chunkKey,
} from '../constants';
import { AROUND, ringSize, slotOf, type CellStore, type MeshTarget } from '../sim/store';
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
  viewRadius: number;
  ghostRadius: number;
}

/** A chunk in view with a mesh: where its mesh is, and its centre (to sort translucent water). */
export interface ChunkDraw {
  meshSlot: number;
  center: [number, number, number];
}

/** Mesh slots for a view radius: one per chunk in view. */
export const meshSlotCount = (viewRadius: number) => (2 * viewRadius + 1) ** 2;

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
  /** Chunks in view, drawn once they have a mesh. */
  private readonly inView = new Set<Chunk>();
  /** Chunks in view whose mesh is missing or out of date (version changed). */
  private readonly toMesh = new Set<Chunk>();
  /** A meshing round in flight (GPU meshing reads counts back between its passes). */
  meshing?: Promise<void>;
  /**
   * Chunks the camera sees (main.ts sets this each frame): generated and meshed before the
   * rest, so flying forward fills in what's ahead first. The simulated area and the ring
   * around it always come first, whatever the camera sees.
   */
  focus?: (cx: number, cz: number) => boolean;
  private loadedCount = 0;
  /** Offsets within the halo, nearest first (the order chunks are generated in). */
  private readonly spiral: Array<[number, number]>;

  /**
   * activeRadius: chunks simulated by block updates. viewRadius: chunks drawn (at least
   * activeRadius). ghostRadius: chunks loaded, one ring past the view, so meshes at its
   * edge and block updates at the active edge have their neighbours' cells.
   */
  constructor(
    readonly store: CellStore, readonly activeRadius = ACTIVE_RADIUS, readonly ghostRadius = GHOST_RADIUS,
    readonly viewRadius = activeRadius,
  ) {
    if (viewRadius < activeRadius || ghostRadius <= viewRadius) throw new Error('need activeRadius <= viewRadius < ghostRadius');
    this.ring = ringSize(ghostRadius);
    if (store.ring !== this.ring) throw new Error(`a store with a ${store.ring}-slot ring can't hold a ${this.ring}-chunk halo`);
    this.slots = new Array<Chunk | undefined>(this.ring * this.ring).fill(undefined);
    this.freeMeshSlots = Array.from({ length: meshSlotCount(viewRadius) }, (_, i) => i).reverse();
    this.spiral = [];
    for (let dz = -ghostRadius; dz <= ghostRadius; dz++)
      for (let dx = -ghostRadius; dx <= ghostRadius; dx++) this.spiral.push([dx, dz]);
    this.spiral.sort((a, b) => a[0] ** 2 + a[1] ** 2 - (b[0] ** 2 + b[1] ** 2));
    this.window = { cx: 0, cz: 0, activeRadius, viewRadius, ghostRadius };
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
    this.window = { ...this.window, cx, cz };
    this.update();
  }

  private update(): void {
    for (const chunk of [...this.chunks.values()]) {
      if (this.distance(chunk.cx, chunk.cz) > this.ghostRadius) this.unload(chunk);
    }
    const { cx, cz, ghostRadius: r } = this.window;
    for (let dz = -r; dz <= r; dz++)
      for (let dx = -r; dx <= r; dx++) if (!this.getChunk(cx + dx, cz + dz)) this.enter(cx + dx, cz + dz);
    for (const chunk of this.chunks.values()) {
      const d = this.distance(chunk.cx, chunk.cz);
      this.setState(chunk, d <= this.activeRadius ? 'active' : 'ghost');
      // Leaving the view first: chunks coming into view take the mesh slots of chunks leaving it.
      if (d > this.viewRadius) this.setInView(chunk, false);
    }
    for (const chunk of this.chunks.values()) if (this.distance(chunk.cx, chunk.cz) <= this.viewRadius) this.setInView(chunk, true);
  }

  private setInView(chunk: Chunk, inView: boolean): void {
    if (this.inView.has(chunk) === inView) return;
    if (inView) {
      chunk.meshSlot = this.freeMeshSlots.pop()!;
      chunk.meshedVersion = -1; // needs a mesh before it's drawn
      chunk.meshReady = false;
      this.inView.add(chunk);
      this.toMesh.add(chunk);
    } else {
      this.freeMeshSlots.push(chunk.meshSlot);
      chunk.meshSlot = -1;
      chunk.meshReady = false;
      this.inView.delete(chunk);
      this.toMesh.delete(chunk);
    }
  }

  /** The chunk's mesh is out of date. */
  private touch(chunk: Chunk): void {
    if (this.inView.has(chunk)) this.toMesh.add(chunk);
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
    this.setInView(chunk, false);
    if (chunk.loaded) this.loadedCount--;
    chunk.loaded = chunk.loading = false;
    this.slots[chunk.slot] = undefined;
    this.chunks.delete(chunk.key);
    this.awake.delete(chunk);
  }

  private setState(chunk: Chunk, state: ChunkState): void {
    if (chunk.state === state) return;
    chunk.state = state;
    // Outside the active area a chunk is frozen; once active it may have updates to catch up on
    // (if it or a neighbour was changed: generated terrain is settled).
    if (state === 'active' && this.unsettled(chunk)) this.awake.add(chunk);
  }

  /** Chebyshev distance in chunks from the window centre. */
  distance(cx: number, cz: number): number {
    return Math.max(Math.abs(cx - this.window.cx), Math.abs(cz - this.window.cz));
  }

  /** Chunks in the halo that need generating: nearest first, those in focus before the rest. */
  missingChunks(limit: number): Chunk[] {
    const out: Chunk[] = [], later: Chunk[] = [];
    if (this.loadedCount === this.chunks.size) return out;
    const { cx, cz } = this.window;
    for (const [dx, dz] of this.spiral) {
      const c = this.getChunk(cx + dx, cz + dz);
      if (!c || c.loaded || c.loading) continue;
      if (this.first(c)) out.push(c);
      else if (later.length < limit) later.push(c);
      if (out.length >= limit) break;
    }
    return out.concat(later).slice(0, limit);
  }

  /** In the simulated area or the ring around it, or in focus. */
  private first(c: Chunk): boolean {
    return this.distance(c.cx, c.cz) <= this.activeRadius + 1 || (this.focus?.(c.cx, c.cz) ?? true);
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
    if (!chunk.loaded) this.loadedCount++;
    chunk.loaded = true;
    chunk.loading = false;
    this.touch(chunk);
    this.store.setSlot(chunk.slot, chunk.cx, chunk.cz, true);
    // Its side neighbours' meshes see its border. Generated terrain is settled, so block
    // updates only wake where edits are: this chunk if it or a neighbour was changed, and
    // changed neighbours (their water may now flow in).
    if (this.unsettled(chunk)) this.awake.add(chunk);
    for (let dz = -1; dz <= 1; dz++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dz) continue;
        const n = this.getChunk(chunk.cx + dx, chunk.cz + dz);
        if (!n) continue;
        if (!dx || !dz) {
          n.version++;
          this.touch(n);
        }
        if (chunk.modified || n.modified) this.awake.add(n);
      }
    }
  }

  /** It or a neighbour was changed since generation, so it may have block updates to run. */
  private unsettled(chunk: Chunk): boolean {
    if (chunk.modified) return true;
    for (let dz = -1; dz <= 1; dz++)
      for (let dx = -1; dx <= 1; dx++) if (this.getChunk(chunk.cx + dx, chunk.cz + dz)?.modified) return true;
    return false;
  }

  /** True once every chunk in the halo is loaded. */
  haloReady(): boolean {
    return this.loadedCount === this.chunks.size;
  }

  /** True once the active area and the ring around it are loaded: block updates can run. */
  simReady(): boolean {
    const { cx, cz } = this.window, r = this.activeRadius + 1;
    for (let dz = -r; dz <= r; dz++)
      for (let dx = -r; dx <= r; dx++) if (!this.getChunk(cx + dx, cz + dz)?.loaded) return false;
    return true;
  }

  /** How many chunks of the halo are loaded. */
  haloProgress(): { loaded: number; total: number } {
    return { loaded: this.loadedCount, total: this.chunks.size };
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
    this.touch(chunk);
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
    if (remesh) {
      c.version++;
      this.touch(c);
    }
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
   * Mesh chunks in view whose cells changed, nearest first (those in focus before the rest), up to the store's budget per
   * round, one round at a time (GPU meshing reads counts back mid-way). A chunk waits for
   * its four side neighbours, so border faces cull correctly. Returns how many it started.
   */
  remesh(target: MeshTarget | undefined): number {
    if (this.meshing) return 0;
    const { cx, cz } = this.window;
    const loaded = (x: number, z: number) => this.getChunk(x, z)?.loaded === true;
    const todo: Chunk[] = [];
    for (const c of this.toMesh) {
      if (c.version === c.meshedVersion) this.toMesh.delete(c);
      else if (c.loaded && loaded(c.cx + 1, c.cz) && loaded(c.cx - 1, c.cz) && loaded(c.cx, c.cz + 1) && loaded(c.cx, c.cz - 1)) todo.push(c);
    }
    if (todo.length === 0) return 0;
    const rank = (c: Chunk) => (c.cx - cx) ** 2 + (c.cz - cz) ** 2 + (this.first(c) ? 0 : 1e9);
    todo.sort((a, b) => rank(a) - rank(b));
    todo.length = Math.min(todo.length, this.store.meshBudget);
    const jobs = todo.map((c) => {
      c.meshedVersion = c.version;
      this.toMesh.delete(c);
      const slot = c.meshSlot;
      return {
        around: this.around(c), meshSlot: slot, cx: c.cx, cz: c.cz,
        current: () => this.inView.has(c) && c.meshSlot === slot,
      };
    });
    const round = this.store.mesh(jobs, target).then(() => {
      for (const job of jobs) {
        if (!job.current()) continue;
        const chunk = this.getChunk(job.cx, job.cz)!;
        if (chunk.meshReady) continue; // a remesh (an edit): already shown
        chunk.meshReady = true;
        chunk.shownAt = performance.now();
        target?.shown?.(job.meshSlot, chunk.shownAt);
      }
    }, (e: unknown) => {
      for (const c of todo) { c.meshedVersion = -1; this.touch(c); } // try again
      throw e;
    });
    this.meshing = round.finally(() => { this.meshing = undefined; });
    this.meshing.catch(() => {});
    return todo.length;
  }

  /** Chunks in view with a mesh (that `visible` accepts, e.g. in the camera's frustum). */
  draws(visible?: (cx: number, cz: number) => boolean): ChunkDraw[] {
    const out: ChunkDraw[] = [];
    for (const c of this.inView) {
      if (!c.meshReady || (visible && !visible(c.cx, c.cz))) continue;
      out.push({ meshSlot: c.meshSlot, center: [(c.cx + 0.5) * CHUNK_SIZE, CHUNK_HEIGHT / 2, (c.cz + 0.5) * CHUNK_SIZE] });
    }
    return out;
  }

  /**
   * Which chunks in view are drawn and fully faded in (`fadeMs` after first shown), as a
   * square of bytes (255 covered, 0 not), row by row from chunk (x0, z0): where the far
   * terrain stops being drawn.
   */
  coverage(now: number, fadeMs: number): { x0: number; z0: number; size: number; data: Uint8Array<ArrayBuffer> } {
    const r = this.viewRadius, size = 2 * r + 1, x0 = this.window.cx - r, z0 = this.window.cz - r;
    const data = new Uint8Array(size * size);
    for (const c of this.inView) {
      if (!c.meshReady || now - c.shownAt < fadeMs) continue;
      const x = c.cx - x0, z = c.cz - z0;
      if (x >= 0 && z >= 0 && x < size && z < size) data[z * size + x] = 255;
    }
    return { x0, z0, size, data };
  }

  /** Chunks simulated, in view, and loaded as the halo, for the HUD. */
  counts(): { active: number; inView: number; halo: number } {
    const a = 2 * this.activeRadius + 1;
    return { active: a * a, inView: this.inView.size, halo: this.chunks.size };
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
