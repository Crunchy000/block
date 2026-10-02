import * as tf from '@tensorflow/tfjs';
import { expect, it } from 'vitest';
import { DEFAULT_RATES } from '../src/constants';
import { CpuStore } from '../src/sim/cpuStore';
import { Simulation } from '../src/sim/simulation';
import { ringSize } from '../src/sim/store';
import { generateAll } from '../src/world/loader';
import { World } from '../src/world/world';

// Generated terrain is already settled (grass on top, ripe wheat, no lava by water, still
// water): fresh chunks don't wake for block updates (world.ts relies on this), and a tick
// over all of them anyway changes nothing.
it('a freshly generated world is settled', async () => {
  await tf.setBackend('cpu');
  const world = new World(new CpuStore(ringSize(4)), 3, 4);
  await generateAll(world);
  const store = world.store as CpuStore;
  const before = Int32Array.from(store.cells);
  const sim = new Simulation(world, DEFAULT_RATES);
  expect(await sim.tick()).toBe(false); // nothing woke
  world.wake([...world.chunks.values()].filter((c) => c.state === 'active'));
  expect(await sim.tick()).toBe(true);
  expect(sim.lastBatch).toBe(49);
  expect(sim.lastChangedChunks).toBe(0);
  expect(sim.growingChunks).toBe(0);
  expect(await sim.tick()).toBe(false);
  expect(sim.asleep).toBe(true);
  expect(store.cells).toEqual(before);
}, 600000);
