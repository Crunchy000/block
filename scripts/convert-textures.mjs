// Builds the block textures the game loads, public/textures/blocks.bin, from the Baunilha
// texture pack (assets/baunilha/: "Baunilha" by Mirtilo, CC BY-SA 4.0, for Luanti). Run after
// changing them: node scripts/convert-textures.mjs   (needs ffmpeg, to decode the PNGs)
//
// blocks.bin: u32 layer count, then each layer as 16 x 16 RGBA bytes. Overlays are composited
// here (the grass side over dirt, the diamond over stone), animations split into frames. The
// layer order is the one src/render/blockTextures.ts names: keep the two in step.
// (These composited textures are an adaptation of Baunilha, so CC BY-SA 4.0 too.)
import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';

const SRC = new URL('../assets/baunilha/', import.meta.url);
const SIZE = 16;

/** A PNG as RGBA bytes (width SIZE; any height: animation frames stacked downwards). */
function rgba(name) {
  return execFileSync('ffmpeg', ['-v', 'error', '-i', new URL(name, SRC).pathname, '-f', 'rawvideo', '-pix_fmt', 'rgba', '-']);
}
/** The frames of a vertical strip. */
const frames = (strip) => Array.from({ length: strip.length / (SIZE * SIZE * 4) }, (_, k) => strip.subarray(k * SIZE * SIZE * 4, (k + 1) * SIZE * SIZE * 4));
/** `top` over `base` (alpha blending), opaque. */
function over(base, top) {
  const out = Buffer.from(base);
  for (let i = 0; i < out.length; i += 4) {
    const a = top[i + 3] / 255;
    for (let k = 0; k < 3; k++) out[i + k] = Math.round(top[i + k] * a + base[i + k] * (1 - a));
    out[i + 3] = 255;
  }
  return out;
}

const dirt = rgba('default_dirt.png'), stone = rgba('default_stone.png');
const layers = [
  ['grass top', rgba('default_grass.png')],
  ['grass side', over(dirt, rgba('default_grass_side.png'))],
  ['dirt', dirt],
  ['stone', stone],
  ['diamond ore', over(stone, rgba('default_mineral_diamond.png'))],
  ...frames(rgba('default_water_source_animated.png')).map((f, k) => [`water ${k}`, f]),
  ...frames(rgba('default_lava_source_animated.png')).map((f, k) => [`lava ${k}`, f]),
  ...[1, 2, 3, 4, 5, 6, 7, 8].map((s) => [`wheat ${s}`, rgba(`farming_wheat_${s}.png`)]),
  ['sand', rgba('default_sand.png')],
  ['gravel', rgba('default_gravel.png')],
];
for (const [name, data] of layers) if (data.length !== SIZE * SIZE * 4) throw new Error(`${name}: not ${SIZE} x ${SIZE}`);

const head = Buffer.alloc(4);
head.writeUInt32LE(layers.length);
writeFileSync(new URL('../public/textures/blocks.bin', import.meta.url), Buffer.concat([head, ...layers.map(([, d]) => d)]));
console.log(`${layers.length} layers: ${layers.map(([n]) => n).join(', ')}`);
