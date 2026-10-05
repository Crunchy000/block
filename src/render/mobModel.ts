import { assetUrl } from '../assetUrl';

/**
 * An animated, textured model for mobs (made by scripts/convert-models.mjs; see there for the
 * file layout): vertices with positions in their part's space, normals, texture coordinates
 * and a part index; u16 indices; the texture; and its poses, a matrix per part per frame of
 * each animation clip, as a float texture the vertex shader reads (3 texels a matrix).
 */
export interface MobModel {
  /** Per vertex: position (3), normal (3), uv (2), part (1) floats. */
  vertex: GPUBuffer;
  index: GPUBuffer;
  indexCount: number;
  texture: GPUTexture;
  /** rgba32float, parts × 3 wide, one row per frame. */
  poses: GPUTexture;
  clips: Record<string, Clip>;
  fps: number;
}

/** A clip's frames: rows start..start + frames - 1 of the poses (the last the same as the first, for loops). */
export interface Clip { start: number; frames: number }

/** Floats per vertex in MobModel.vertex. */
export const MOB_VERTEX_FLOATS = 9;

interface ModelHeader { vertices: number; indices: number; parts: number; frames: number; fps: number; clips: Record<string, Clip> }

/** Parse a model file: its header, vertices, indices (u16) and poses (12 floats per part per frame). */
export function parseModel(data: ArrayBuffer): ModelHeader & {
  vertexData: Float32Array<ArrayBuffer>; indexData: Uint16Array<ArrayBuffer>; poseData: Float32Array<ArrayBuffer>;
} {
  const jsonBytes = new Uint32Array(data, 0, 1)[0];
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 4, jsonBytes))) as ModelHeader;
  let at = 4 + jsonBytes;
  const vertexData = new Float32Array(data.slice(at, at += header.vertices * MOB_VERTEX_FLOATS * 4));
  // Padded to a multiple of 4 bytes, as GPU buffer writes need.
  const indexBytes = Math.ceil((header.indices * 2) / 4) * 4;
  const indexData = new Uint16Array(data.slice(at, at += indexBytes));
  const poseData = new Float32Array(data.slice(at, at + header.frames * header.parts * 48));
  return { ...header, vertexData, indexData, poseData };
}

/**
 * Where a mob `time` seconds into `clip` (looping) is: the two frames of the model's poses
 * it's between, and how far from the first to the second.
 */
export function poseFrames(model: Pick<MobModel, 'clips' | 'fps'>, clip: string, time: number): [number, number, number] {
  const c = model.clips[clip] ?? model.clips.idle;
  const loop = c.frames - 1;
  if (loop <= 0) return [c.start, c.start, 0];
  const f = ((time * model.fps) % loop + loop) % loop, k = Math.floor(f);
  return [c.start + k, c.start + k + 1, f - k];
}

const url = assetUrl;

/** A model as fetched, before a renderer uploads it: its parsed file and its texture's image. */
export type MobModelData = ReturnType<typeof parseModel> & { image: ImageBitmap };

/** Texture images by path: models sharing one (the animals' palette) fetch it once. */
const images = new Map<string, Promise<ImageBitmap>>();

function fetchImage(path: string): Promise<ImageBitmap> {
  let image = images.get(path);
  if (!image) {
    image = fetch(url(path))
      .then((r) => { if (!r.ok) throw new Error(`${path}: ${r.status}`); return r.blob(); })
      .then((b) => createImageBitmap(b));
    images.set(path, image);
  }
  return image;
}

/** Fetch a model and its texture (paths relative to the page), for a renderer to upload. */
export async function fetchMobModel(modelPath: string, texturePath: string): Promise<MobModelData> {
  const [data, image] = await Promise.all([
    fetch(url(modelPath)).then((r) => { if (!r.ok) throw new Error(`${modelPath}: ${r.status}`); return r.arrayBuffer(); }),
    fetchImage(texturePath),
  ]);
  return { ...parseModel(data), image };
}

/** Textures on the GPU by image: models sharing one upload it once. */
const textures = new WeakMap<ImageBitmap, GPUTexture>();

/** A fetched model on a WebGPU device. */
export function mobModelOnGpu(device: GPUDevice, m: MobModelData): MobModel {
  let texture = textures.get(m.image);
  if (!texture) {
    // Plain rgba8unorm (not sRGB): block colours are lit in the same space, so mobs match them.
    texture = device.createTexture({
      label: 'mob texture', size: [m.image.width, m.image.height], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture({ source: m.image }, { texture }, [m.image.width, m.image.height]);
    textures.set(m.image, texture);
  }
  const vertex = device.createBuffer({ label: 'mob vertices', size: m.vertexData.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(vertex, 0, m.vertexData);
  const index = device.createBuffer({ label: 'mob indices', size: m.indexData.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(index, 0, m.indexData);
  const poses = device.createTexture({
    label: 'mob poses', size: [m.parts * 3, m.frames], format: 'rgba32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture: poses }, m.poseData, { bytesPerRow: m.parts * 3 * 16 }, [m.parts * 3, m.frames]);
  return { vertex, index, indexCount: m.indices, texture, poses, clips: m.clips, fps: m.fps };
}
