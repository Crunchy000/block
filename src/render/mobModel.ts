/**
 * A textured model for mobs (the pig): vertices with positions, normals and texture
 * coordinates, u16 indices, and its colour texture. Made by scripts/convert-pig.mjs; see
 * there for the file layout.
 */
export interface MobModel {
  /** Per vertex: position (3), normal (3), uv (2) floats. */
  vertex: GPUBuffer;
  index: GPUBuffer;
  indexCount: number;
  texture: GPUTexture;
}

/** Floats per vertex in MobModel.vertex. */
export const MOB_VERTEX_FLOATS = 8;

/** Parse a model file (vertex count, index count, positions, normals, uvs, u16 indices) into interleaved vertices. */
export function parseModel(data: ArrayBuffer): { vertices: Float32Array<ArrayBuffer>; indices: Uint16Array<ArrayBuffer> } {
  const head = new Uint32Array(data, 0, 2);
  const n = head[0], m = head[1];
  const f = new Float32Array(data, 8, n * 8);
  const vertices = new Float32Array(n * MOB_VERTEX_FLOATS);
  for (let i = 0; i < n; i++) {
    vertices.set(f.subarray(i * 3, i * 3 + 3), i * 8);
    vertices.set(f.subarray(n * 3 + i * 3, n * 3 + i * 3 + 3), i * 8 + 3);
    vertices.set(f.subarray(n * 6 + i * 2, n * 6 + i * 2 + 2), i * 8 + 6);
  }
  // Padded to a multiple of 4 bytes, as GPU buffer writes need.
  const indices = new Uint16Array(Math.ceil(m / 2) * 2);
  indices.set(new Uint16Array(data, 8 + n * 32, m));
  return { vertices, indices: indices.subarray(0, m) as Uint16Array<ArrayBuffer> };
}

/** Fetch a model and its texture (paths relative to the page) onto the GPU. */
export async function loadMobModel(device: GPUDevice, modelPath: string, texturePath: string): Promise<MobModel> {
  const url = (path: string) => new URL(path, document.baseURI).toString();
  const [data, image] = await Promise.all([
    fetch(url(modelPath)).then((r) => { if (!r.ok) throw new Error(`${modelPath}: ${r.status}`); return r.arrayBuffer(); }),
    fetch(url(texturePath)).then((r) => { if (!r.ok) throw new Error(`${texturePath}: ${r.status}`); return r.blob(); }).then((b) => createImageBitmap(b)),
  ]);
  const { vertices, indices } = parseModel(data);
  const vertex = device.createBuffer({ label: 'mob vertices', size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertex, 0, vertices);
  const indexBytes = Math.ceil(indices.byteLength / 4) * 4;
  const index = device.createBuffer({ label: 'mob indices', size: indexBytes, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(index, 0, new Uint16Array(indices.buffer, 0, indexBytes / 2));
  // Plain rgba8unorm (not sRGB): block colours are lit in the same space, so the pig matches them.
  const texture = device.createTexture({
    label: 'mob texture', size: [image.width, image.height], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: image }, { texture }, [image.width, image.height]);
  image.close();
  return { vertex, index, indexCount: indices.length, texture };
}
