import { assetUrl } from '../assetUrl';

/**
 * Block textures: "Baunilha" by Mirtilo (CC BY-SA 4.0), a texture pack for Luanti, as layers
 * of one 16 x 16 texture array with mipmaps. scripts/convert-textures.mjs builds the file,
 * public/textures/blocks.bin; the layers are in this order (keep the two in step).
 */
export const TEXTURE_SIZE = 16;
export const Layer = {
  GrassTop: 0,
  GrassSide: 1,
  Dirt: 2,
  Stone: 3,
  Diamond: 4,
  /** 16 frames, animated. */
  Water: 5,
  WaterFrames: 16,
  /** 8 frames, animated. */
  Lava: 21,
  LavaFrames: 8,
  /** Growth stages 0..7. */
  Wheat: 29,
  Sand: 37,
  Gravel: 38,
} as const;

/**
 * The mip levels of `count` square RGBA layers of `size` texels (level 0 being `data`), each
 * half the last down to 1 x 1: 2 x 2 texel averages, colour weighted by alpha (so see-through
 * texels, wheat's background, don't darken the stalks).
 */
export function mipLevels(data: Uint8Array<ArrayBuffer>, size: number, count: number): Uint8Array<ArrayBuffer>[] {
  const levels = [data];
  for (let s = size; s > 1; s /= 2) {
    const src = levels[levels.length - 1], h = s / 2;
    const out = new Uint8Array(h * h * 4 * count);
    for (let l = 0; l < count; l++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < h; x++) {
          let r = 0, g = 0, b = 0, a = 0;
          for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
            const i = ((l * s + 2 * y + dy) * s + 2 * x + dx) * 4, w = src[i + 3];
            r += src[i] * w; g += src[i + 1] * w; b += src[i + 2] * w; a += w;
          }
          const o = ((l * h + y) * h + x) * 4;
          if (a > 0) out.set([Math.round(r / a), Math.round(g / a), Math.round(b / a), Math.round(a / 4)], o);
        }
      }
    }
    levels.push(out);
  }
  return levels;
}

/** The block textures as loaded: `count` layers, and their mip levels (TEXTURE_SIZE down to 1, RGBA, every layer). */
export interface BlockTextureData { count: number; levels: Uint8Array<ArrayBuffer>[] }

/** Fetch the block textures and make their mipmaps (for a renderer to upload as a texture array). */
export async function fetchBlockTextures(): Promise<BlockTextureData> {
  const r = await fetch(assetUrl('textures/blocks.bin'));
  if (!r.ok) throw new Error(`textures/blocks.bin: ${r.status}`);
  const file = await r.arrayBuffer();
  const count = new Uint32Array(file, 0, 1)[0];
  return { count, levels: mipLevels(new Uint8Array(file, 4, count * TEXTURE_SIZE * TEXTURE_SIZE * 4), TEXTURE_SIZE, count) };
}

/** The block textures on a WebGPU device, as a texture array with mipmaps. */
export function blockTexturesOnGpu(device: GPUDevice, { count, levels }: BlockTextureData): GPUTexture {
  const texture = device.createTexture({
    label: 'block textures', size: [TEXTURE_SIZE, TEXTURE_SIZE, count], mipLevelCount: levels.length,
    // Plain rgba8unorm (not sRGB), as the mobs' texture: lit in the same space as before.
    format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  levels.forEach((data, mipLevel) => {
    const s = TEXTURE_SIZE >> mipLevel;
    device.queue.writeTexture({ texture, mipLevel }, data, { bytesPerRow: s * 4, rowsPerImage: s }, [s, s, count]);
  });
  return texture;
}
