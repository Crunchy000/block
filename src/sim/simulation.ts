import { DEFAULT_RATES, type PlantRates } from '../constants';
import type { Chunk } from '../world/chunk';
import { tickJobs, type World } from '../world/world';
import { randomSeed } from './rules';
import { TickFlag } from './store';

/**
 * Runs block updates for the active area.
 *
 * Only chunks that can change are simulated: the world keeps an "awake" set (chunks
 * that were edited, freshly loaded, next to a change, or with plants still growing).
 * A tick hands the awake chunks to the store, which updates them where the cells live
 * (on the GPU in the game) and returns one flags word per chunk: whether it changed in
 * a way that matters, which of its borders that touched, and whether it has plants
 * still growing. The flags are all that comes back; they decide what remeshes and
 * what stays awake. When nothing is awake the tick is skipped entirely.
 */
export class Simulation {
  ticks = 0;
  /** Wall time of the last tick, including waiting for the GPU. */
  lastTickMs = 0;
  /** Main-thread time of the last tick (issuing it, applying its flags). */
  lastCpuMs = 0;
  lastChangedChunks = 0;
  /** Chunks simulated in the last tick. */
  lastBatch = 0;
  /** Chunks where plants could still change by chance after the last tick. */
  growingChunks = 0;
  asleep = false;
  /** A tick is in flight. */
  busy = false;
  /** Each tick's seed for the plant rules' random numbers (tests make it deterministic). */
  seed: () => number = randomSeed;

  constructor(private readonly world: World, public rates: PlantRates = DEFAULT_RATES) {}

  /** Returns true if a tick ran (false when not ready, busy, or nothing is awake). */
  async tick(): Promise<boolean> {
    const world = this.world;
    if (this.busy || !world.simReady()) return false;
    const awake = world.takeAwake();
    this.asleep = awake.length === 0;
    if (this.asleep) return false;

    const t0 = performance.now();
    this.busy = true;
    let cpuMs = 0;
    try {
      this.lastBatch = awake.length;
      const pending = world.store.tick(tickJobs(world, awake), this.seed(), this.rates);
      cpuMs += performance.now() - t0;
      const flags = await pending;
      const t1 = performance.now();
      this.lastChangedChunks = this.apply(awake, flags);
      cpuMs += performance.now() - t1;
      this.ticks++;
      return true;
    } catch (e) {
      world.wake(awake); // retry these next time
      throw e;
    } finally {
      this.busy = false;
      this.lastTickMs = performance.now() - t0;
      this.lastCpuMs = cpuMs;
    }
  }

  /** Act on each chunk's flags. Returns how many chunks changed in a way that matters. */
  private apply(chunks: Chunk[], flags: Uint32Array): number {
    let changed = 0, growing = 0;
    chunks.forEach((chunk, n) => {
      if (!this.world.isResident(chunk)) return; // left the halo while the tick ran
      const f = flags[n];
      if (f & TickFlag.Changed) {
        changed++;
        this.world.markChanged(chunk, f & TickFlag.Borders);
      }
      // Primed dirt and unripe wheat change by chance, so this chunk isn't settled even if
      // nothing changed this tick.
      if (f & TickFlag.Growing) {
        growing++;
        this.world.wake([chunk]);
      }
    });
    this.growingChunks = growing;
    return changed;
  }
}
