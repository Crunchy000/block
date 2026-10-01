import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, blockIndex, cell } from '../constants';
import { MeshPool } from '../render/meshPool';
import { FACE_CAPACITY } from '../render/mesher';
import { mulberry32, randomCells } from '../tf/kernelCheck';
import { borderOf } from '../world/world';
import { CpuStore } from './cpuStore';
import { GpuStore } from './gpuStore';
import { AROUND, TickFlag, slotOf } from './store';

export interface StoreCheck {
  ok: boolean;
  /** What was compared. */
  summary: string;
  /** The first difference, if any. */
  detail: string;
}

const RING = 3;

/**
 * Run the GPU store and the reference (CpuStore) side by side on random cells of every
 * kind and compare everything they produce, exactly: cells and flags after each of a
 * few block-update ticks (random chunks, neighbours wrapping around the ring), meshes
 * (as sets of faces) and picking. Small enough to run at startup, so a GPU or driver
 * that computes something different can fall back to the reference.
 */
export async function checkGpuStore(device: GPUDevice, seed = 1, ticks = 2, rays = 64): Promise<StoreCheck> {
  const rand = mulberry32(seed);
  const gpu = await GpuStore.create(device, RING), cpu = new CpuStore(RING);
  const pool = new MeshPool(device, 3);
  const slots = RING * RING;
  let detail = '', cellsCompared = 0, facesCompared = 0;
  const fail = (text: string) => { detail ||= text; };
  try {
    // Slot (sx, sz) holds chunk (sx, sz). One chunk isn't loaded: picking must see air there.
    const unloaded = Math.floor(rand() * slots);
    for (let slot = 0; slot < slots; slot++) {
      const cells = randomCells(CHUNK_VOLUME, rand);
      for (const store of [gpu, cpu]) {
        store.writeChunk(slot, cells);
        store.setSlot(slot, slot % RING, Math.floor(slot / RING), slot !== unloaded);
      }
    }
    const around = (slot: number) => {
      const out: number[] = [];
      for (let dz = -1; dz <= 1; dz++)
        for (let dx = -1; dx <= 1; dx++) out.push(slotOf(slot % RING + dx, Math.floor(slot / RING) + dz, RING));
      return out;
    };
    const compareCells = async (what: string) => {
      for (let slot = 0; slot < slots; slot++) {
        const got = await gpu.readChunk(slot), want = cpu.cells.subarray(slot * CHUNK_VOLUME, (slot + 1) * CHUNK_VOLUME);
        cellsCompared += CHUNK_VOLUME;
        const i = got.findIndex((v, k) => v !== want[k]);
        if (i >= 0) fail(`${what}: slot ${slot} cell ${i} (x${i & 15} y${i >> 8} z${(i >> 4) & 15}) is ${got[i]}, expected ${want[i]}`);
      }
    };

    // Block updates.
    const rates = { grassSpread: 0.25, wheatGrow: 0.125, wheatGrowWet: 0.5 };
    for (let tick = 0; tick < ticks && !detail; tick++) {
      const chosen = [...Array(slots).keys()].filter(() => rand() < 0.5);
      if (chosen.length === 0) chosen.push(0);
      const jobs = Uint32Array.from(chosen.flatMap(around));
      const tickSeed = (rand() * 2 ** 32) >>> 0;
      const [got, want] = await Promise.all([gpu.tick(jobs, tickSeed, rates), cpu.tick(jobs, tickSeed, rates)]);
      for (let k = 0; k < chosen.length; k++) {
        if (got[k] !== want[k]) fail(`tick ${tick}: flags of slot ${jobs[k * AROUND + 4]} are ${got[k].toString(2)}, expected ${want[k].toString(2)}`);
      }
      await compareCells(`after tick ${tick}`);
    }

    // Flags, with each chunk changing in one known place (random cells change everywhere):
    // wheat on stone pops off at a border, corner or the middle; unripe wheat that can't
    // grow (rates 0) changes nothing but keeps its chunk growing.
    const spots = [[0, 0], [15, 0], [0, 15], [15, 15], [0, 7], [15, 7], [7, 0], [7, 15], [7, 7]];
    for (let slot = 0; slot < slots; slot++) {
      const cells = new Int32Array(CHUNK_VOLUME);
      const [x, z] = spots[slot];
      for (let i = 0; i < CHUNK_SIZE * CHUNK_SIZE; i++) cells[i] = cell(Block.Stone);
      if (slot === slots - 1) cells[blockIndex(x, 0, z)] = cell(Block.Dirt);
      cells[blockIndex(x, 1, z)] = cell(Block.Wheat, 3);
      gpu.writeChunk(slot, cells);
      cpu.writeChunk(slot, cells);
    }
    const all = Uint32Array.from([...Array(slots).keys()].flatMap(around));
    const still = { grassSpread: 0, wheatGrow: 0, wheatGrowWet: 0 };
    const [gotFlags, wantFlags] = await Promise.all([gpu.tick(all, 7, still), cpu.tick(all, 7, still)]);
    spots.forEach(([x, z], slot) => {
      const expected = slot === slots - 1 ? TickFlag.Growing : TickFlag.Changed | borderOf(x, z);
      if (wantFlags[slot] !== expected) fail(`reference flags for a change at ${x},${z}: ${wantFlags[slot].toString(2)}, expected ${expected.toString(2)}`);
      if (gotFlags[slot] !== expected) fail(`flags for a change at ${x},${z}: ${gotFlags[slot].toString(2)}, expected ${expected.toString(2)}`);
    });

    // Meshing (of random cells again).
    for (let slot = 0; slot < slots; slot++) {
      const cells = randomCells(CHUNK_VOLUME, rand);
      gpu.writeChunk(slot, cells);
      cpu.writeChunk(slot, cells);
    }
    const meshed = [0, 4, 8].map((slot, meshSlot) => ({ around: around(slot), meshSlot, cx: slot % RING, cz: Math.floor(slot / RING) }));
    gpu.mesh(meshed, pool);
    for (const job of meshed) {
      const got = await pool.read(job.meshSlot), want = cpu.meshChunk(job.around);
      const first = job.meshSlot * FACE_CAPACITY * 6;
      if (got.draws.join() !== [want.opaque.length * 6, 1, first, 0, want.water.length * 6, 1, first, 0].join()) {
        fail(`mesh of chunk ${job.cx},${job.cz}: draws ${got.draws.join()}, expected ${want.opaque.length} opaque and ${want.water.length} water faces`);
      }
      for (const kind of ['opaque', 'water'] as const) {
        const a = [...got[kind]].sort(), b = [...want[kind]].sort();
        facesCompared += b.length;
        const i = b.findIndex((v, k) => v !== a[k]);
        if (a.length !== b.length || i >= 0) fail(`mesh of chunk ${job.cx},${job.cz}: ${kind} faces differ (${a.length} vs ${b.length}; first at ${i})`);
      }
    }

    // Picking, in a sparser world so rays travel.
    for (let slot = 0; slot < slots; slot++) {
      const cells = randomCells(CHUNK_VOLUME, rand).map((c) => (rand() < 0.9 ? Block.Air : c));
      gpu.writeChunk(slot, cells);
      cpu.writeChunk(slot, cells);
    }
    for (let r = 0; r < rays; r++) {
      const span = RING * CHUNK_SIZE;
      const origin = [rand() * (span + 8) - 4, rand() * (CHUNK_HEIGHT + 8) - 4, rand() * (span + 8) - 4];
      const dir = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
      if (r % 8 === 0) dir[r % 3] = 0; // along a plane
      const len = Math.hypot(...dir) || 1;
      const unit = dir.map((d) => d / len), maxDist = 4 + rand() * 16;
      const [got, want] = await Promise.all([gpu.raycast(origin, unit, maxDist), cpu.raycast(origin, unit, maxDist)]);
      if (JSON.stringify(got) !== JSON.stringify(want)) {
        fail(`ray from ${origin.map((v) => v.toFixed(2))} along ${unit.map((v) => v.toFixed(3))}: ${JSON.stringify(got)}, expected ${JSON.stringify(want)}`);
      }
    }
  } catch (e) {
    fail(`failed to run: ${e instanceof Error ? e.message : String(e)}`);
  } finally {
    gpu.destroy();
    pool.destroy();
  }
  return {
    ok: detail === '',
    summary: `${ticks} ticks (${cellsCompared.toLocaleString()} cells), ${facesCompared.toLocaleString()} mesh faces, ${rays} rays`,
    detail,
  };
}
