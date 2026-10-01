import * as tf from '@tensorflow/tfjs';
import { expect, it } from 'vitest';
import { DEFAULT_RATES } from '../src/constants';
import { CpuStore } from '../src/sim/cpuStore';
import { Simulation } from '../src/sim/simulation';
import { ringSize } from '../src/sim/store';
import { generateAll } from '../src/world/loader';
import { World } from '../src/world/world';

// Generated terrain is already settled (grass on top, ripe wheat, no lava by water, still
// water), so the first tick over a fresh world changes nothing and the world goes to sleep.
it('a freshly generated world goes to sleep after one tick', async () => {
  await tf.setBackend('cpu');
  const world = new World(new CpuStore(ringSize(4)), 3, 4);
  await generateAll(world);
  const store = world.store as CpuStore;
  const before = Int32Array.from(store.cells);
  const sim = new Simulation(world, DEFAULT_RATES);
  expect(await sim.tick()).toBe(true);
  expect(sim.lastChangedChunks).toBe(0);
  expect(sim.growingChunks).toBe(0);
  expect(await sim.tick()).toBe(false);
  expect(sim.asleep).toBe(true);
  expect(store.cells).toEqual(before);
}, 600000);
