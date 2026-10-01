import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, SOURCE_LEVEL, blockIndex, cell, cellType } from '../src/constants';
import { Simulation } from '../src/tf/simulation';
import { World } from '../src/world/world';

beforeAll(async () => {
  await tf.setBackend('cpu');
});

/** Fill the whole halo with flat stone (y < 4). */
function fill(world: World): void {
  for (const c of world.missingChunks(1000)) {
    const data = new Uint8Array(CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT);
    for (let y = 0; y < 4; y++)
      for (let z = 0; z < CHUNK_SIZE; z++)
        for (let x = 0; x < CHUNK_SIZE; x++) data[blockIndex(x, y, z)] = cell(Block.Stone);
    world.addGenerated(c, data);
  }
}

describe('active area + ghost halo', () => {
  it('marks the inner square active and the surrounding ring ghost', () => {
    const world = new World(1, 2);
    world.recenter(CHUNK_SIZE * 10 + 3, -5); // chunk (10, -1)
    fill(world);
    expect(world.haloReady()).toBe(true);
    expect(world.activeChunks()).toHaveLength(9);
    expect(world.ghostChunks()).toHaveLength(16);
    expect(world.getChunk(10, -1)!.state).toBe('active');
    expect(world.getChunk(12, -1)!.state).toBe('ghost');
  });

  it('moving drops pristine chunks outside the halo but keeps modified ones', () => {
    const world = new World(1, 2);
    world.recenter(0, 0);
    fill(world);
    world.setCell(-CHUNK_SIZE * 2, 5, 0, cell(Block.Dirt)); // edit a ghost chunk at (-2, 0)
    world.recenter(CHUNK_SIZE * 3, 0); // now centred on (3, 0)
    expect(world.getChunk(-2, 0)).toBeDefined();
    expect(world.getChunk(-2, 1)).toBeUndefined();
    expect(world.getChunk(1, 0)!.state).toBe('ghost');
    expect(world.getChunk(2, 0)!.state).toBe('active');
  });

  it('block updates flow from ghost into active chunks but never write ghost chunks', async () => {
    const world = new World(0, 1);
    world.recenter(8, 8);
    fill(world);
    // Water source in the ghost chunk to the east, right at the border.
    world.setCell(CHUNK_SIZE, 4, 8, cell(Block.Water, SOURCE_LEVEL));
    const ghostBefore = Uint8Array.from(world.getChunk(1, 0)!.data);

    const sim = new Simulation(world);
    for (let i = 0; i < 3; i++) expect(await sim.tick()).toBe(true);

    expect(world.getChunk(1, 0)!.data).toEqual(ghostBefore);
    expect(cellType(world.getCell(CHUNK_SIZE - 1, 4, 8))).toBe(Block.Water);
    expect(cellType(world.getCell(CHUNK_SIZE - 3, 4, 8))).toBe(Block.Water);
  });

  it('queues edits made while a tick is in flight', async () => {
    const world = new World(0, 1);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    const pending = sim.tick();
    world.setCell(3, 6, 3, cell(Block.Dirt));
    expect(world.getBlock(3, 6, 3)).toBe(Block.Air);
    await pending;
    expect(world.getBlock(3, 6, 3)).toBe(Block.Dirt);
  });

  it('sleeps once nothing is changing, so idle ticks cost nothing', async () => {
    const world = new World(1, 2);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    expect(await sim.tick()).toBe(true); // freshly loaded chunks are awake
    expect(sim.lastRegion).toEqual([5, 5]); // 3x3 active + ghost border
    expect(sim.lastChangedChunks).toBe(0);
    expect(await sim.tick()).toBe(false);
    expect(sim.asleep).toBe(true);
  });

  it('an edit only simulates the chunk around it, then sleeps again once settled', async () => {
    const world = new World(1, 2);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    await sim.tick();
    world.setCell(CHUNK_SIZE + 8, 4, 8, cell(Block.Water, SOURCE_LEVEL)); // middle of chunk (1, 0)
    expect(await sim.tick()).toBe(true);
    expect(sim.lastRegion).toEqual([3, 3]);
    let ticks = 1;
    while (await sim.tick()) ticks++;
    expect(ticks).toBeLessThan(12);
    expect(sim.asleep).toBe(true);
    expect(cellType(world.getCell(CHUNK_SIZE + 2, 4, 8))).toBe(Block.Water); // spread 6 blocks
  });

  it('flow across a chunk border wakes the neighbouring chunk', async () => {
    const world = new World(1, 2);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    await sim.tick();
    world.setCell(CHUNK_SIZE - 2, 4, 8, cell(Block.Water, SOURCE_LEVEL)); // near the east edge of chunk (0, 0)
    await sim.tick();
    expect(sim.lastRegion).toEqual([3, 3]);
    await sim.tick(); // water reached x = 15, the border with chunk (1, 0)
    await sim.tick();
    expect(sim.lastRegion).toEqual([4, 3]);
    expect(cellType(world.getCell(CHUNK_SIZE, 4, 8))).toBe(Block.Water);
  });
});
