import { GEN_BATCH, generateChunksTensor } from '../tf/worldgen';
import type { World } from './world';

/**
 * Generate the nearest missing chunks (up to GEN_BATCH, as one TF.js batch) and write
 * them into their store slots: on the GPU straight from TF.js's output buffer when the
 * world lives there too. Resolves to how many chunks it generated.
 */
export async function generateMissing(world: World, seed?: number): Promise<number> {
  const chunks = world.missingChunks(GEN_BATCH);
  if (chunks.length === 0) return 0;
  world.markGenerating(chunks);
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
export async function generateAll(world: World, seed?: number): Promise<void> {
  while (await generateMissing(world, seed));
}
