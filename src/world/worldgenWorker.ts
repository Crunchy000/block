/// <reference lib="webworker" />
import { generateChunkJs } from './jsWorldgen';
import type { ChunkCoord } from './worldgenParams';

/**
 * World generation off the main thread (safe mode): generates the chunks it's sent with the
 * plain-JS generator and sends their cells back (the buffers handed over, not copied).
 */
self.onmessage = (e: MessageEvent<{ id: number; coords: ChunkCoord[]; seed?: number }>) => {
  const { id, coords, seed } = e.data;
  const chunks = coords.map((c) => generateChunkJs(c, seed));
  (self as unknown as DedicatedWorkerGlobalScope).postMessage({ id, chunks }, chunks.map((c) => c.buffer));
};
