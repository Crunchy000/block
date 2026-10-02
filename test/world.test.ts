import { describe, expect, it } from 'vitest';
import {
  Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, DEFAULT_RATES, PRIMED_DIRT, SOURCE_LEVEL, WHEAT_RIPE, blockIndex, cell,
  cellType, concrete,
} from '../src/constants';
import { CpuStore } from '../src/sim/cpuStore';
import { Simulation } from '../src/sim/simulation';
import { ringSize, slotOf } from '../src/sim/store';
import { blockUpdateReference, cellRandom } from '../src/tf/blockUpdateReference';
import { mulberry32, randomCells } from '../src/tf/kernelCheck';
import { World } from '../src/world/world';

const newWorld = (activeRadius: number, ghostRadius: number) =>
  new World(new CpuStore(ringSize(ghostRadius)), activeRadius, ghostRadius);

/** Flat stone (y < 4), optionally topped with a layer at y = 4. */
function flatChunk(top?: Block): Int32Array {
  const data = new Int32Array(CHUNK_VOLUME);
  for (let y = 0; y < (top === undefined ? 4 : 5); y++)
    for (let z = 0; z < CHUNK_SIZE; z++)
      for (let x = 0; x < CHUNK_SIZE; x++) data[blockIndex(x, y, z)] = cell(y === 4 ? top! : Block.Stone);
  return data;
}

/** Load every missing chunk in the halo with flat terrain. */
function fill(world: World, top?: Block): void {
  for (const c of world.missingChunks(1000)) world.addGenerated(c, flatChunk(top));
}

const blockAt = async (world: World, x: number, y: number, z: number) => cellType(await world.readCell(x, y, z));
const chunkCells = (world: World, cx: number, cz: number) => world.store.readChunk(world.getChunk(cx, cz)!.slot);

/** Run ticks until the simulation sleeps; returns how many ran. */
async function settle(sim: Simulation, max = 100): Promise<number> {
  let ticks = 0;
  while (await sim.tick()) if (++ticks >= max) throw new Error(`still awake after ${max} ticks`);
  return ticks;
}

describe('active area + ghost halo', () => {
  it('marks the inner square active and the surrounding ring ghost', () => {
    const world = newWorld(1, 2);
    world.recenter(CHUNK_SIZE * 10 + 3, -5); // chunk (10, -1)
    expect(world.haloReady()).toBe(false);
    fill(world);
    expect(world.haloReady()).toBe(true);
    expect(world.activeChunks()).toHaveLength(9);
    expect(world.ghostChunks()).toHaveLength(16);
    expect(world.getChunk(10, -1)!.state).toBe('active');
    expect(world.getChunk(12, -1)!.state).toBe('ghost');
  });

  it('keeps each chunk in the ring slot its coordinates pick, reusing the slots of chunks that leave', () => {
    const world = newWorld(1, 2);
    fill(world);
    const before = new Set(world.chunks.values());
    world.recenter(CHUNK_SIZE * 2 + 1, CHUNK_SIZE); // two east, one south
    for (const c of world.chunks.values()) {
      expect(c.slot).toBe(slotOf(c.cx, c.cz, world.ring));
      expect(world.slots[c.slot]).toBe(c);
    }
    const stayed = [...world.chunks.values()].filter((c) => before.has(c));
    expect(stayed).toHaveLength(3 * 4); // the overlap of the two 5x5 halos
    expect(stayed.every((c) => c.loaded)).toBe(true);
    expect(world.missingChunks(100)).toHaveLength(25 - 12);
  });

  it('gives each active chunk its own mesh slot, and ghosts none', () => {
    const world = newWorld(1, 2);
    fill(world);
    for (const step of [0, 1, 2, 3]) {
      world.recenter(CHUNK_SIZE * step, CHUNK_SIZE * step * 2);
      const slots = world.activeChunks().map((c) => c.meshSlot);
      expect(new Set(slots).size).toBe(9);
      expect(slots.every((m) => m >= 0 && m < 9)).toBe(true);
      expect(world.ghostChunks().every((c) => c.meshSlot === -1)).toBe(true);
    }
  });

  it('saves edited chunks that leave the halo and restores them when they come back', async () => {
    const world = newWorld(1, 2);
    fill(world);
    world.setCell(-CHUNK_SIZE * 2 + 3, 5, 7, cell(Block.Dirt)); // ghost chunk (-2, 0)
    world.recenter(CHUNK_SIZE * 3, 0); // now centred on (3, 0): (-2, 0) is gone
    expect(world.getChunk(-2, 0)).toBeUndefined();
    expect(world.savedCount()).toBe(1);
    fill(world);
    world.recenter(8, 8); // back
    // Pristine chunks are generated again; the edited one is restored instead.
    expect(world.missingChunks(100).some((c) => c.cx === -2 && c.cz === 0)).toBe(false);
    fill(world);
    await new Promise((resolve) => setTimeout(resolve, 0)); // the read-back cells arrive
    expect(world.getChunk(-2, 0)!.loaded).toBe(true);
    expect(world.getChunk(-2, 0)!.modified).toBe(true);
    expect(world.savedCount()).toBe(0);
    expect(await blockAt(world, -CHUNK_SIZE * 2 + 3, 5, 7)).toBe(Block.Dirt);
    expect(world.haloReady()).toBe(true);
  });
});

describe('block updates', () => {
  it('flow from ghost into active chunks but never write ghost chunks', async () => {
    const world = newWorld(0, 1);
    world.recenter(8, 8);
    fill(world);
    // Water source in the ghost chunk to the east, right at the border.
    world.setCell(CHUNK_SIZE, 4, 8, cell(Block.Water, SOURCE_LEVEL));
    const ghostBefore = await chunkCells(world, 1, 0);

    const sim = new Simulation(world);
    for (let i = 0; i < 3; i++) expect(await sim.tick()).toBe(true);

    expect(await chunkCells(world, 1, 0)).toEqual(ghostBefore);
    expect(await blockAt(world, CHUNK_SIZE - 1, 4, 8)).toBe(Block.Water);
    expect(await blockAt(world, CHUNK_SIZE - 3, 4, 8)).toBe(Block.Water);
  });

  it('match the reference rules run on the whole area at once', async () => {
    // A tick gives each chunk a one-cell border from its neighbours; running the rules
    // over all 3x3 chunks as one region must give the same middle chunk.
    const world = newWorld(0, 1);
    const rand = mulberry32(7);
    for (const c of world.missingChunks(100)) world.addGenerated(c, randomCells(CHUNK_VOLUME, rand));
    const W = 3 * CHUNK_SIZE, region = new Int32Array(CHUNK_HEIGHT * W * W);
    for (let cz = -1; cz <= 1; cz++)
      for (let cx = -1; cx <= 1; cx++) {
        const cells = await chunkCells(world, cx, cz);
        for (let i = 0; i < CHUNK_VOLUME; i++) {
          const x = i & 15, z = (i >> 4) & 15, y = i >> 8;
          region[(y * W + (cz + 1) * CHUNK_SIZE + z) * W + (cx + 1) * CHUNK_SIZE + x] = cells[i];
        }
      }
    const seed = 99, self = world.getChunk(0, 0)!.slot;
    const random = new Float32Array(region.length);
    for (let i = 0; i < CHUNK_VOLUME; i++) {
      const x = i & 15, z = (i >> 4) & 15, y = i >> 8;
      random[(y * W + CHUNK_SIZE + z) * W + CHUNK_SIZE + x] = cellRandom(self * CHUNK_VOLUME + i, seed);
    }
    const rates = { grassSpread: 0.3, wheatGrow: 0.2, wheatGrowWet: 0.6 };
    const expected = blockUpdateReference(region, CHUNK_HEIGHT, W, W, random, rates);
    await world.store.tick(Uint32Array.from(world.around(world.getChunk(0, 0)!)), seed, rates);
    const got = await chunkCells(world, 0, 0);
    let mismatches = 0;
    for (let i = 0; i < CHUNK_VOLUME; i++) {
      const x = i & 15, z = (i >> 4) & 15, y = i >> 8;
      if (got[i] !== expected[(y * W + CHUNK_SIZE + z) * W + CHUNK_SIZE + x]) mismatches++;
    }
    expect(mismatches).toBe(0);
  });

  it('land edits made while a tick is in flight after it', async () => {
    const world = newWorld(0, 1);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    const pending = sim.tick();
    world.setCell(3, 6, 3, cell(Block.Dirt));
    await pending;
    expect(await blockAt(world, 3, 6, 3)).toBe(Block.Dirt);
    expect(await sim.tick()).toBe(true); // the edit woke its chunk
    expect(sim.lastBatch).toBe(1);
  });

  it('sleep once nothing is changing, so idle ticks cost nothing', async () => {
    const world = newWorld(1, 2);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    // Generated terrain is settled, so fresh chunks don't even wake.
    expect(await sim.tick()).toBe(false);
    expect(sim.asleep).toBe(true);
    // Woken anyway, a tick changes nothing and they go back to sleep.
    world.wake([...world.chunks.values()].filter((c) => c.state === 'active'));
    expect(await sim.tick()).toBe(true);
    expect(sim.lastBatch).toBe(9); // all 3x3 active chunks
    expect(sim.lastChangedChunks).toBe(0);
    expect(await sim.tick()).toBe(false);
  });

  it('generate the simulated area first, then chunks in focus, then the rest, nearest first', () => {
    const world = newWorld(1, 4);
    world.recenter(8, 8);
    world.focus = (cx) => cx >= 3; // the camera looks east
    const order = world.missingChunks(1000);
    const near = (c: { cx: number; cz: number }) => Math.max(Math.abs(c.cx), Math.abs(c.cz)) <= 2;
    const firstFar = order.findIndex((c) => !near(c));
    expect(order.slice(0, firstFar).every(near)).toBe(true);
    const rest = order.slice(firstFar);
    const lastFocused = rest.map((c) => c.cx >= 3).lastIndexOf(true);
    expect(rest.slice(0, lastFocused + 1).every((c) => c.cx >= 3)).toBe(true);
    expect(rest.slice(lastFocused + 1).some((c) => c.cx >= 3)).toBe(false);
    expect(order.length).toBe(81);
  });

  it('leave pastel concrete as it is (solid, keeps its colour, holds up water)', async () => {
    const world = newWorld(1, 2);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    world.setCell(8, 10, 8, concrete(5));
    world.setCell(8, 11, 8, cell(Block.Water, SOURCE_LEVEL));
    await settle(sim);
    expect(await world.readCell(8, 10, 8)).toBe(concrete(5));
    expect(await blockAt(world, 8, 11, 8)).toBe(Block.Water);
  });

  it('report chunks shown once (not again on a remesh), covered once faded in', async () => {
    const world = newWorld(1, 2);
    world.recenter(8, 8);
    fill(world);
    const shown: number[] = [];
    const target = { upload: () => {}, shown: (slot: number) => { shown.push(slot); } };
    while (world.remesh(target) > 0) await world.meshing;
    const inView = [...world.chunks.values()].filter((c) => c.meshReady);
    expect(inView).toHaveLength(9); // view radius 1 (the halo's edge has no neighbours to mesh against)
    expect(shown.sort()).toEqual(inView.map((c) => c.meshSlot).sort());
    // Covered only once fully faded in; no fade (safe mode) covers at once.
    const now = performance.now();
    expect(world.coverage(now, 600).data.every((v) => v === 0)).toBe(true);
    const later = world.coverage(now + 1000, 600);
    expect(later.size).toBe(3);
    expect(Array.from(later.data)).toEqual(Array(9).fill(255));
    expect(Array.from(world.coverage(now, 0).data)).toEqual(Array(9).fill(255));
    // An edit remeshes a chunk without showing (fading) it again.
    world.setCell(8, 5, 8, cell(Block.Air));
    while (world.remesh(target) > 0) await world.meshing;
    expect(shown).toHaveLength(9);
  });

  it('wake chunks that come into the active area next to an edited one', async () => {
    const world = newWorld(1, 2);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    world.setCell(CHUNK_SIZE + 15, 4, 8, cell(Block.Stone)); // chunk (1, 0), at its east border
    await settle(sim);
    // Move one chunk east: chunk (2, 0) becomes active beside the edited chunk, so it wakes;
    // the pristine chunks coming in elsewhere don't.
    world.recenter(CHUNK_SIZE + 8, 8);
    fill(world);
    expect(world.getChunk(2, 0)!.state).toBe('active');
    const awake = world.takeAwake().map((c) => c.key).sort();
    expect(awake).toContain(world.getChunk(2, 0)!.key);
    expect(awake.every((k) => {
      const c = world.chunks.get(k)!;
      return Math.abs(c.cx - 1) <= 1 && Math.abs(c.cz) <= 1;
    })).toBe(true);
  });

  it('only simulate the chunk around an edit, then sleep again once settled', async () => {
    const world = newWorld(1, 2);
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
    expect(await blockAt(world, CHUNK_SIZE + 2, 4, 8)).toBe(Block.Water); // spread 6 blocks
  });

  it('wake the neighbouring chunk when flow crosses a chunk border, and remesh both', async () => {
    const world = newWorld(1, 2);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    await sim.tick();
    const east = world.getChunk(1, 0)!, eastVersion = east.version;
    world.setCell(CHUNK_SIZE - 2, 4, 8, cell(Block.Water, SOURCE_LEVEL)); // near the east edge of chunk (0, 0)
    expect(east.version).toBe(eastVersion); // not on the border
    await sim.tick(); // the water spreads to x = 15, the border with chunk (1, 0)
    expect(sim.lastBatch).toBe(1);
    expect(east.version).toBeGreaterThan(eastVersion); // its west faces depend on that border
    await sim.tick(); // and on into chunk (1, 0), now awake
    expect(sim.lastBatch).toBe(2);
    expect(await blockAt(world, CHUNK_SIZE, 4, 8)).toBe(Block.Water);
  });
});

describe('plants', () => {
  it('grass keeps its chunk awake while it spreads, then the chunk sleeps', async () => {
    const world = newWorld(0, 1);
    world.recenter(8, 8);
    fill(world, Block.Dirt);
    const sim = new Simulation(world, { ...DEFAULT_RATES, grassSpread: 1 }); // spread every tick: deterministic
    await settle(sim);
    world.setCell(8, 4, 8, cell(Block.Grass));
    const ticks = await settle(sim);
    expect(ticks).toBeGreaterThanOrEqual(8); // ~1 block per tick to the far corner
    expect(sim.growingChunks).toBe(0);
    const cells = await chunkCells(world, 0, 0);
    for (let z = 0; z < CHUNK_SIZE; z++)
      for (let x = 0; x < CHUNK_SIZE; x++) expect(cellType(cells[blockIndex(x, 4, z)])).toBe(Block.Grass);
    // Ghost chunks are read (border cells) but never written.
    expect((await chunkCells(world, 1, 0)).every((c) => cellType(c) !== Block.Grass)).toBe(true);
  });

  it('a chance roll that changes nothing does not put spreading grass to sleep', async () => {
    const world = newWorld(0, 1);
    world.recenter(8, 8);
    fill(world, Block.Dirt);
    const sim = new Simulation(world, { ...DEFAULT_RATES, grassSpread: 1e-9 }); // effectively never sprouts
    await settle(sim);
    world.setCell(8, 4, 8, cell(Block.Grass));
    const chunk = world.getChunk(0, 0)!;
    await sim.tick(); // the edit
    const version = chunk.version;
    for (let i = 0; i < 3; i++) expect(await sim.tick()).toBe(true);
    expect(sim.lastChangedChunks).toBe(0); // nothing visible happened...
    expect(chunk.version).toBe(version); // ...so nothing to remesh...
    expect(sim.growingChunks).toBe(1); // ...but primed dirt keeps the chunk awake
    expect(await world.readCell(9, 4, 9)).toBe(PRIMED_DIRT);
  });

  it('grass spreads diagonally across a chunk corner through the ghost border', async () => {
    const world = newWorld(1, 2);
    world.recenter(8, 8);
    fill(world, Block.Dirt);
    const sim = new Simulation(world, { ...DEFAULT_RATES, grassSpread: 1 });
    await settle(sim);
    world.setCell(CHUNK_SIZE - 1, 4, CHUNK_SIZE - 1, cell(Block.Grass)); // corner of chunk (0, 0)
    expect(await sim.tick()).toBe(true);
    expect(sim.lastBatch).toBe(4); // (0,0), its east and south neighbours, and the diagonal (1,1)
    expect(await blockAt(world, CHUNK_SIZE, 4, CHUNK_SIZE)).toBe(Block.Grass);
  });

  it('growing wheat keeps its chunk awake until it is ripe, then the chunk sleeps', async () => {
    const world = newWorld(0, 1);
    world.recenter(8, 8);
    fill(world, Block.Dirt);
    const sim = new Simulation(world, { ...DEFAULT_RATES, wheatGrow: 1, wheatGrowWet: 1 }); // a stage per tick
    await settle(sim);
    world.setCell(8, 5, 8, cell(Block.Wheat, 0));
    const ticks = await settle(sim);
    expect(await world.readCell(8, 5, 8)).toBe(cell(Block.Wheat, WHEAT_RIPE));
    expect(ticks).toBe(WHEAT_RIPE + 1); // 7 growing ticks, then one that finds it ripe
    expect(sim.growingChunks).toBe(0);
  });

  it('wheat planted on stone pops off', async () => {
    const world = newWorld(0, 1);
    world.recenter(8, 8);
    fill(world);
    const sim = new Simulation(world);
    await settle(sim);
    world.setCell(8, 4, 8, cell(Block.Wheat, 0));
    await sim.tick();
    expect(await blockAt(world, 8, 4, 8)).toBe(Block.Air);
  });
});

describe('picking', () => {
  it('finds the first solid block across chunk borders, and sees unloaded chunks as air', async () => {
    const world = newWorld(1, 2);
    fill(world);
    world.setCell(CHUNK_SIZE + 2, 5, 3, cell(Block.Dirt)); // in chunk (1, 0)
    world.setCell(CHUNK_SIZE + 1, 5, 3, cell(Block.Water, SOURCE_LEVEL)); // fluids are passed through
    const hit = await world.store.raycast([CHUNK_SIZE - 2.5, 5.5, 3.5], [1, 0, 0], 8);
    expect(hit).toEqual({ block: [CHUNK_SIZE + 2, 5, 3], before: [CHUNK_SIZE + 1, 5, 3], type: Block.Dirt });
    // Straight down onto the stone floor.
    expect((await world.store.raycast([4.5, 9.5, 4.5], [0, -1, 0], 8))?.block).toEqual([4, 3, 4]);
    // The ring slot of chunk (3, 0) holds chunk (-2, 0): picking must not see it.
    world.setCell(-CHUNK_SIZE * 2 + 1, 6, 1, cell(Block.Stone));
    expect(await world.store.raycast([CHUNK_SIZE * 3 + 1.5, 6.5, -0.5], [0, 0, 1], 4)).toBeNull();
  });
});
