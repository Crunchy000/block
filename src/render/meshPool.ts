import { CHUNK_SIZE } from '../constants';
import { FACE_CAPACITY, type ChunkFaces } from './mesher';

/** Bytes of draw arguments per mesh slot: an indirect draw for its opaque faces, then one for its water. */
const DRAW_BYTES = 32;

/**
 * GPU memory for the meshes of the active chunks, drawn with indirect draws, so the
 * CPU never needs to know what's in them or even how many faces they have.
 *
 * Each mesh slot has room for FACE_CAPACITY face records: opaque faces fill it from
 * the start, water faces from the end. Its two indirect draws (opaque, water) count
 * 6 vertices per face and start at vertex slot * FACE_CAPACITY * 6, so the vertex
 * shader can tell from a vertex's index which slot and face it belongs to.
 */
export class MeshPool {
  /** Face records (render/mesher.ts), FACE_CAPACITY per slot. */
  readonly faces: GPUBuffer;
  /** Per slot: drawIndirect arguments for its opaque faces, then for its water faces. */
  readonly draws: GPUBuffer;
  /** Per slot: the world-space corner of its chunk, as vec4<i32>(x, 0, z, 0). */
  readonly origins: GPUBuffer;

  constructor(readonly device: GPUDevice, readonly slots: number) {
    const bytes = slots * FACE_CAPACITY * 4;
    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    if (bytes > limit) {
      throw new Error(`Meshes for ${slots} chunks need ${bytes >> 20} MB in one GPU buffer, more than this GPU allows (${limit >> 20} MB). Try a smaller ?radius.`);
    }
    this.faces = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.draws = device.createBuffer({
      size: slots * DRAW_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.origins = device.createBuffer({ size: slots * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }

  /** Offset in `draws` of a slot's opaque draw (its water draw follows, 16 bytes on). */
  drawOffset(slot: number): number {
    return slot * DRAW_BYTES;
  }

  /** Point a slot at chunk (cx, cz) and empty it, ready for the GPU mesher to add faces. */
  reset(slot: number, cx: number, cz: number): void {
    this.write(slot, cx, cz, 0, 0);
  }

  /** Put a chunk meshed on the CPU into a slot. */
  upload(slot: number, cx: number, cz: number, { opaque, water }: ChunkFaces): void {
    const base = slot * FACE_CAPACITY * 4;
    if (opaque.length) this.device.queue.writeBuffer(this.faces, base, Uint32Array.from(opaque));
    // Water fills the slot from the end: record FACE_CAPACITY - 1 - i of the slot is water face i.
    if (water.length) this.device.queue.writeBuffer(this.faces, base + (FACE_CAPACITY - water.length) * 4, Uint32Array.from(water).reverse());
    this.write(slot, cx, cz, opaque.length, water.length);
  }

  private write(slot: number, cx: number, cz: number, opaque: number, water: number): void {
    const first = slot * FACE_CAPACITY * 6;
    this.device.queue.writeBuffer(this.draws, slot * DRAW_BYTES, new Uint32Array([opaque * 6, 1, first, 0, water * 6, 1, first, 0]));
    this.device.queue.writeBuffer(this.origins, slot * 16, new Int32Array([cx * CHUNK_SIZE, 0, cz * CHUNK_SIZE, 0]));
  }

  /** Read a slot back: its faces and its draw arguments (for checks and tests). */
  async read(slot: number): Promise<ChunkFaces & { draws: number[] }> {
    const { device } = this;
    const read = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
    const draws = device.createBuffer({ size: DRAW_BYTES, usage: read });
    const faces = device.createBuffer({ size: FACE_CAPACITY * 4, usage: read });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.draws, slot * DRAW_BYTES, draws, 0, DRAW_BYTES);
    encoder.copyBufferToBuffer(this.faces, slot * FACE_CAPACITY * 4, faces, 0, FACE_CAPACITY * 4);
    device.queue.submit([encoder.finish()]);
    try {
      await Promise.all([draws.mapAsync(GPUMapMode.READ), faces.mapAsync(GPUMapMode.READ)]);
      const d = Array.from(new Uint32Array(draws.getMappedRange()));
      const f = new Uint32Array(faces.getMappedRange());
      return {
        opaque: Array.from(f.subarray(0, d[0] / 6)),
        water: Array.from(f.subarray(FACE_CAPACITY - d[4] / 6)).reverse(),
        draws: d,
      };
    } finally {
      draws.destroy();
      faces.destroy();
    }
  }

  destroy(): void {
    this.faces.destroy();
    this.draws.destroy();
    this.origins.destroy();
  }
}
