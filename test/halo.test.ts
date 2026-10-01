import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import {
  Block, CHUNK_HEIGHT, CHUNK_SIZE, DEFAULT_RATES, PRIMED_DIRT, SOURCE_LEVEL, WHEAT_RIPE, blockIndex, cell, cellType,
} from '../src/constants';
import { Simulation } from '../src/tf/simulation';
import { World } from '../src/world/world';

beforeAll(async () => {
  await tf.setBackend('cpu');
});

/** Fill the whole halo with flat stone (y < 4), optionally topped with a layer at y = 4. */
function fill(world: World, top?: Block): void {
  for (const c of world.missingChunks(1000)) {
    const data = new Uint8Array(CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT);
    for (let y = 0; y < (top === undefined ? 4 : 5); y++)
      for (let z = 0; z < CHUNK_SIZE; z++)
        for (let x = 0; x < CHUNK_SIZE; x++) data[blockIndex(x, y, z)] = cell(y === 4 ? top! : Block.Stone);
    world.addGenerated(c, data);
  }
}

/** Run ticks until the simulation sleeps; returns how many ran. */
async function settle(sim: Simulation, max = 100): Promise<number> {
  let ticks = 0;
  while (await sim.tick()) if (++ticks >= max) throw new Error(`still awake after ${max} ticks`);
  return ticks;
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
    expect(sim.lastBatch).toBe(9); // all 3x3 active chunks
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
    expect(sim.lastBatch).toBe(1);
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
    expect(sim.lastBatch).toBe(1);
    await sim.tick(); // water reached x = 15, the border with chunk (1, 0)
    await sim.tick();
    expect(sim.lastBatch).toBe(2);
    expect(cellType(world.getCell(CHUNK_SIZE, 4, 8))).toBe(Block.Water);
  });

  it('grass keeps its chunk awake while it spreads, then the chunk sleeps', async () => {
    const world = new World(0, 1);
    world.recenter(8, 8);
    fill(world, Block.Dirt);
    const sim = new Simulation(world, { ...DEFAULT_RATES, grassSpread: 1 }); // spread every tick: deterministic
    await settle(sim);
    world.setCell(8, 4, 8, cell(Block.Grass));
    const ticks = await settle(sim);
    expect(ticks).toBeGreaterThanOrEqual(8); // ~1 block per tick to the far corner
    expect(sim.growingChunks).toBe(0);
    for (let z = 0; z < CHUNK_SIZE; z++)
      for (let x = 0; x < CHUNK_SIZE; x++) expect(world.getBlock(x, 4, z)).toBe(Block.Grass);
    // Ghost chunks are read (border cells) but never written.
    expect(world.getBlock(CHUNK_SIZE, 4, 8)).toBe(Block.Dirt);
    expect(world.getChunk(1, 0)!.data.every((c) => cellType(c) !== Block.Grass)).toBe(true);
  });

  it('a chance roll that changes nothing does not put spreading grass to sleep', async () => {
    const world = new World(0, 1);
    world.recenter(8, 8);
    fill(world, Block.Dirt);
    const sim = new Simulation(world, { ...DEFAULT_RATES, grassSpread: 1e-9 }); // effectively never sprouts
    await settle(sim);
    world.setCell(8, 4, 8, cell(Block.Grass));
    for (let i = 0; i < 3; i++) expect(await sim.tick()).toBe(true);
    expect(sim.lastChangedChunks).toBe(0); // nothing visible happened...
    expect(sim.growingChunks).toBe(1); // ...but primed dirt keeps the chunk awake
    expect(world.getCell(9, 4, 9)).toBe(PRIMED_DIRT);
  });

  it('grass spreads diagonally across a chunk corner through the ghost border', async () => {
    const world = new World(1, 2);
    world.recenter(8, 8);
    fill(world, Block.Dirt);
    const sim = new Simulation(world, { ...DEFAULT_RATES, grassSpread: 1 });
    await settle(sim);
    world.setCell(CHUNK_SIZE - 1, 4, CHUNK_SIZE - 1, cell(Block.Grass)); // corner of chunk (0, 0)
    expect(await sim.tick()).toBe(true);
    expect(sim.lastBatch).toBe(4); // (0,0), its east and south neighbours, and the diagonal (1,1)
    expect(world.getBlock(CHUNK_SIZE, 4, CHUNK_SIZE)).toBe(Block.Grass);
  });

  it('chunks without plants nearby skip the plant rules', async () => {
    const world = new World(1, 2);
    world.recenter(8, 8);
    fill(world, Block.Dirt);
    const sim = new Simulation(world);
    await sim.tick();
    expect(sim.lastPlants).toBe(false);
    world.setCell(8, 4, 8, cell(Block.Grass));
    await sim.tick();
    expect(sim.lastPlants).toBe(true);
    expect(sim.lastBatch).toBe(1);
  });

  it('growing wheat keeps its chunk awake until it is ripe, then the chunk sleeps', async () => {
    const world = new World(0, 1);
    world.recenter(8, 8);
    fill(world, Block.Dirt);
    const sim = new Simulation(world, { ...DEFAULT_RATES, wheatGrow: 1, wheatGrowWet: 1 }); // a stage per tick
    await settle(sim);
    world.setCell(8, 5, 8, cell(Block.Wheat, 0));
    const ticks = await settle(sim);
    expect(world.getCell(8, 5, 8)).toBe(cell(Block.Wheat, WHEAT_RIPE));
    expect(ticks).toBe(WHEAT_RIPE + 1); // 7 growing ticks, then one that finds it ripe
    expect(sim.growingChunks).toBe(0);
  });

  it('wheat planted on stone pops off', async () => {
    const world = new World(0, 1);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    await settle(sim);
    world.setCell(8, 4, 8, cell(Block.Wheat, 0));
    await sim.tick();
    expect(world.getBlock(8, 4, 8)).toBe(Block.Air);
  });
});
