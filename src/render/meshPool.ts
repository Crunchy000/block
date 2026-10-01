import { CHUNK_SIZE } from '../constants';
import type { ChunkFaces } from './mesher';

/** A run of face records in the heap. */
export interface FaceRange { start: number; count: number }
/** A chunk's mesh: its opaque faces and its translucent water faces. */
export interface ChunkMesh { opaque: FaceRange; water: FaceRange }

const NONE: FaceRange = { start: 0, count: 0 };
/** Face records the heap starts with per mesh slot (grown when needed). */
const START_FACES_PER_SLOT = 2048;

/** First-fit allocator over [0, capacity) in face records, merging freed neighbours. */
class Heap {
  /** Free runs, sorted by start. */
  private free: FaceRange[];
  constructor(public capacity: number) {
    this.free = [{ start: 0, count: capacity }];
  }
  alloc(count: number): number {
    if (count === 0) return 0;
    const i = this.free.findIndex((r) => r.count >= count);
    if (i < 0) return -1;
    const r = this.free[i], start = r.start;
    if (r.count === count) this.free.splice(i, 1);
    else this.free[i] = { start: r.start + count, count: r.count - count };
    return start;
  }
  release(range: FaceRange): void {
    if (range.count === 0) return;
    let i = 0;
    while (i < this.free.length && this.free[i].start < range.start) i++;
    this.free.splice(i, 0, { ...range });
    // Merge with the next run, then with the previous one.
    const merge = (a: number) => {
      const x = this.free[a], y = this.free[a + 1];
      if (x && y && x.start + x.count === y.start) this.free.splice(a, 2, { start: x.start, count: x.count + y.count });
    };
    merge(i);
    if (i > 0) merge(i - 1);
  }
  grow(capacity: number): void {
    this.release({ start: this.capacity, count: capacity - this.capacity });
    this.capacity = capacity;
  }
  get used(): number {
    return this.capacity - this.free.reduce((n, r) => n + r.count, 0);
  }
}

/**
 * GPU memory for the meshes of the chunks in view: one buffer of face records (render/mesher.ts),
 * each chunk's opaque and water faces in their own runs, packed to their real size. The
 * CPU knows where every run is and how long, so chunks are drawn with plain draw calls
 * (draw(count * 6, 1, start * 6, slot): the vertex shader finds the face from the vertex
 * index and the chunk's origin from the instance index). The buffer grows when it fills up.
 */
export class MeshPool {
  faces: GPUBuffer;
  /** Per mesh slot: the world-space corner of its chunk, as vec4<i32>(x, 0, z, 0). */
  readonly origins: GPUBuffer;
  private readonly heap: Heap;
  private readonly meshes: Array<ChunkMesh | undefined>;
  private readonly limit: number;

  constructor(readonly device: GPUDevice, readonly slots: number) {
    this.limit = Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) / 4);
    const capacity = Math.min(this.limit, Math.max(1 << 16, slots * START_FACES_PER_SLOT));
    this.heap = new Heap(capacity);
    this.faces = this.createFaces(capacity);
    this.origins = device.createBuffer({ label: 'chunk origins', size: slots * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.meshes = new Array(slots).fill(undefined);
  }

  private createFaces(capacity: number): GPUBuffer {
    return this.device.createBuffer({
      label: 'chunk faces', size: capacity * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  /** The current mesh of a slot. */
  get(slot: number): ChunkMesh | undefined {
    return this.meshes[slot];
  }

  /** Face records in use, and room for. */
  get usage(): { used: number; capacity: number } {
    return { used: this.heap.used, capacity: this.heap.capacity };
  }

  /** Room for a mesh of these sizes (growing the buffer if needed), or undefined if the GPU can't hold it. */
  reserve(opaque: number, water: number): ChunkMesh | undefined {
    for (;;) {
      const o = this.heap.alloc(opaque);
      const w = o < 0 ? -1 : this.heap.alloc(water);
      if (o >= 0 && w >= 0) return { opaque: { start: o, count: opaque }, water: { start: w, count: water } };
      if (o >= 0) this.heap.release({ start: o, count: opaque });
      if (!this.grow(opaque + water)) return undefined;
    }
  }

  /** Double the face buffer (at least `need` more), copying what's there. False at the GPU's limit. */
  private grow(need: number): boolean {
    const old = this.heap.capacity;
    const capacity = Math.min(this.limit, Math.max(old * 2, old + need));
    if (capacity <= old) return false;
    const faces = this.createFaces(capacity);
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.faces, 0, faces, 0, old * 4);
    this.device.queue.submit([encoder.finish()]);
    this.faces.destroy(); // freed once the copy (and any draw already submitted) is done
    this.faces = faces;
    this.heap.grow(capacity);
    return true;
  }

  /** A mesh whose faces have been written becomes the slot's mesh; its old one is freed. */
  commit(slot: number, cx: number, cz: number, mesh: ChunkMesh): void {
    this.release(slot);
    this.meshes[slot] = mesh;
    this.device.queue.writeBuffer(this.origins, slot * 16, new Int32Array([cx * CHUNK_SIZE, 0, cz * CHUNK_SIZE, 0]));
  }

  /** Free a reserved mesh that never became current. */
  discard(mesh: ChunkMesh): void {
    this.heap.release(mesh.opaque);
    this.heap.release(mesh.water);
  }

  /** The slot's chunk left the view: free its mesh. */
  release(slot: number): void {
    const m = this.meshes[slot];
    if (m) this.discard(m);
    this.meshes[slot] = undefined;
  }

  /** Put a chunk meshed on the CPU into a slot. */
  upload(slot: number, cx: number, cz: number, { opaque, water }: ChunkFaces): void {
    const mesh = this.reserve(opaque.length, water.length);
    if (!mesh) throw new Error('the GPU has no room for more chunk meshes');
    if (opaque.length) this.device.queue.writeBuffer(this.faces, mesh.opaque.start * 4, Uint32Array.from(opaque));
    if (water.length) this.device.queue.writeBuffer(this.faces, mesh.water.start * 4, Uint32Array.from(water));
    this.commit(slot, cx, cz, mesh);
  }

  /** Read a slot's mesh back (checks and tests). */
  async read(slot: number): Promise<ChunkFaces> {
    const mesh = this.meshes[slot] ?? { opaque: NONE, water: NONE };
    const part = async (r: FaceRange) => {
      if (r.count === 0) return [];
      const buffer = this.device.createBuffer({ size: r.count * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.faces, r.start * 4, buffer, 0, r.count * 4);
      this.device.queue.submit([encoder.finish()]);
      try {
        await buffer.mapAsync(GPUMapMode.READ);
        return Array.from(new Uint32Array(buffer.getMappedRange()));
      } finally {
        buffer.destroy();
      }
    };
    const [opaque, water] = await Promise.all([part(mesh.opaque), part(mesh.water)]);
    return { opaque, water };
  }

  destroy(): void {
    this.faces.destroy();
    this.origins.destroy();
  }
}
