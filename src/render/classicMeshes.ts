import { faceQuad, type ChunkFaces } from './mesher';

/** Floats per vertex: position (3), normal (3), kind (1). */
export const VERTEX_FLOATS = 7;

export interface ClassicMesh { vertex: GPUBuffer; index: GPUBuffer; count: number }

/** Face records as plain vertices (VERTEX_FLOATS each, 4 a face) and indices (6 a face): safe mode's and WebGL2's meshes. */
export function classicMeshData(records: number[], cx: number, cz: number): { vertices: Float32Array<ArrayBuffer>; indices: Uint32Array<ArrayBuffer> } {
  const vertices = new Float32Array(records.length * 4 * VERTEX_FLOATS);
  const indices = new Uint32Array(records.length * 6);
  records.forEach((record, q) => {
    const { corners, normal, kind } = faceQuad(record, cx * 16, cz * 16);
    corners.forEach((c, k) => vertices.set([c[0], c[1], c[2], normal[0], normal[1], normal[2], kind], (q * 4 + k) * VERTEX_FLOATS));
    const v = q * 4;
    indices.set([v, v + 1, v + 2, v, v + 2, v + 3], q * 6);
  });
  return { vertices, indices };
}

/**
 * Safe mode's chunk meshes (?safe): face records turned into plain vertex and index
 * buffers on the CPU and drawn with ordinary indexed draws, the way the renderer worked
 * before meshing moved to the GPU. Nothing in the vertex stage reads storage buffers and
 * no draw takes its size from GPU memory.
 */
export class ClassicMeshes {
  private readonly slots: Array<{ opaque?: ClassicMesh; water?: ClassicMesh } | undefined>;

  constructor(readonly device: GPUDevice, slots: number) {
    this.slots = new Array(slots).fill(undefined);
  }

  upload(slot: number, cx: number, cz: number, faces: ChunkFaces): void {
    this.free(slot);
    this.slots[slot] = { opaque: this.build(faces.opaque, cx, cz), water: this.build(faces.water, cx, cz) };
  }

  get(slot: number) {
    return this.slots[slot];
  }

  private build(records: number[], cx: number, cz: number): ClassicMesh | undefined {
    if (records.length === 0) return undefined;
    const { vertices, indices } = classicMeshData(records, cx, cz);
    const vertex = this.device.createBuffer({ label: 'safe-mode vertices', size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(vertex, 0, vertices);
    const index = this.device.createBuffer({ label: 'safe-mode indices', size: indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(index, 0, indices);
    return { vertex, index, count: indices.length };
  }

  private free(slot: number): void {
    const s = this.slots[slot];
    for (const m of [s?.opaque, s?.water]) {
      m?.vertex.destroy();
      m?.index.destroy();
    }
    this.slots[slot] = undefined;
  }
}
