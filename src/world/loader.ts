import { GEN_BATCH, generateChunksTensor } from '../tf/worldgen';
import type { World } from './world';

/** Chunks per batch for the worldgen shader (TF.js batches stay at GEN_BATCH). */
export const SHADER_BATCH = 64;

/**
 * Generate the nearest missing chunks (up to SHADER_BATCH or GEN_BATCH) and write them into their store
 * slots: with the GPU store's worldgen shader when the world lives on the GPU, else as one
 * TF.js batch (`viaTensorflow` forces that; the GPU tests compare the two). Resolves to
 * how many chunks it generated.
 */
export async function generateMissing(world: World, seed?: number, viaTensorflow = false): Promise<number> {
  // The worldgen shader is cheap enough for bigger batches (a quarter of a ms a chunk or so).
  const generate = viaTensorflow ? undefined : world.store.generate?.bind(world.store);
  const chunks = world.missingChunks(generate ? SHADER_BATCH : GEN_BATCH);
  if (chunks.length === 0) return 0;
  world.markGenerating(chunks);
  // The GPU store generates straight into its slots, in one compute shader.
  if (generate) {
    try {
      generate(chunks.map(({ slot, cx, cz }) => ({ slot, cx, cz })), seed);
    } catch (e) {
      world.markGenerated(chunks);
      throw e;
    }
    for (const chunk of chunks) world.markLoaded(chunk);
    return chunks.length;
  }
  try {
    // Always a full batch, so every batch runs the same compiled kernels.
    const coords = chunks.map(({ cx, cz }) => ({ cx, cz }));
    while (coords.length < GEN_BATCH) coords.push(coords[0]);
    const cells = generateChunksTensor(coords, seed);
    try {
      const staged = await world.store.stage(cells);
      try {
        chunks.forEach((chunk, k) => {
          if (!world.isResident(chunk) || chunk.loaded) return; // left the halo meanwhile
          staged.write(chunk.slot, k);
          world.markLoaded(chunk);
        });
      } finally {
        staged.release();
      }
    } finally {
      cells.dispose();
    }
  } catch (e) {
    world.markGenerated(chunks);
    throw e;
  }
  return chunks.length;
}

/** Generate until every chunk in the halo is loaded. */
export async function generateAll(world: World, seed?: number, viaTensorflow = false): Promise<void> {
  while (await generateMissing(world, seed, viaTensorflow));
}
