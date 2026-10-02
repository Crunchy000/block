/**
 * A textured model for mobs: vertices with positions, normals, texture coordinates and a
 * colour (which replaces the texture where its alpha is 1), u16 indices, and the texture.
 * Made by scripts/convert-models.mjs; see there for the file layout.
 */
export interface MobModel {
  /** Per vertex: position (3), normal (3), uv (2), colour (4) floats. */
  vertex: GPUBuffer;
  index: GPUBuffer;
  indexCount: number;
  texture: GPUTexture;
}

/** Floats per vertex in MobModel.vertex. */
export const MOB_VERTEX_FLOATS = 12;

/** Parse a model file (vertex count, index count, interleaved vertices, u16 indices). */
export function parseModel(data: ArrayBuffer): { vertices: Float32Array<ArrayBuffer>; indices: Uint16Array<ArrayBuffer> } {
  const head = new Uint32Array(data, 0, 2);
  const n = head[0], m = head[1];
  const vertices = new Float32Array(data.slice(8, 8 + n * MOB_VERTEX_FLOATS * 4));
  // Padded to a multiple of 4 bytes, as GPU buffer writes need.
  const indices = new Uint16Array(Math.ceil(m / 2) * 2);
  indices.set(new Uint16Array(data, 8 + n * MOB_VERTEX_FLOATS * 4, m));
  return { vertices, indices: indices.subarray(0, m) as Uint16Array<ArrayBuffer> };
}

const url = (path: string) => new URL(path, document.baseURI).toString();

/** Textures by path: models sharing one (the farm animals' palette) load it once. */
const textures = new Map<string, Promise<GPUTexture>>();

function loadTexture(device: GPUDevice, path: string): Promise<GPUTexture> {
  let t = textures.get(path);
  if (!t) {
    t = fetch(url(path))
      .then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.blob(); })
      .then((b) => createImageBitmap(b))
      .then((image) => {
        // Plain rgba8unorm (not sRGB): block colours are lit in the same space, so mobs match them.
        const texture = device.createTexture({
          label: path, size: [image.width, image.height], format: 'rgba8unorm',
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.copyExternalImageToTexture({ source: image }, { texture }, [image.width, image.height]);
        image.close();
        return texture;
      });
    textures.set(path, t);
  }
  return t;
}

/** Fetch a model and its texture (paths relative to the page) onto the GPU. */
export async function loadMobModel(device: GPUDevice, modelPath: string, texturePath: string): Promise<MobModel> {
  const [data, texture] = await Promise.all([
    fetch(url(modelPath)).then((r) => { if (!r.ok) throw new Error(`${modelPath}: ${r.status}`); return r.arrayBuffer(); }),
    loadTexture(device, texturePath),
  ]);
  const { vertices, indices } = parseModel(data);
  const vertex = device.createBuffer({ label: 'mob vertices', size: vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertex, 0, vertices);
  const indexBytes = Math.ceil(indices.byteLength / 4) * 4;
  const index = device.createBuffer({ label: 'mob indices', size: indexBytes, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(index, 0, new Uint16Array(indices.buffer, 0, indexBytes / 2));
  return { vertex, index, indexCount: indices.length, texture };
}
