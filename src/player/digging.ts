import { Block } from '../constants';

/** Seconds of digging to break each block (as with a pickaxe): wheat at once, diamond ore slowest. */
export const DIG_SECONDS: Partial<Record<Block, number>> = {
  [Block.Wheat]: 0,
  [Block.Dirt]: 0.35,
  [Block.Sand]: 0.3,
  [Block.Gravel]: 0.4,
  [Block.Grass]: 0.4,
  [Block.Stone]: 0.75,
  [Block.Diamond]: 1.1,
};
/** A short pause after breaking a block before the next one starts, so holding doesn't tunnel. */
export const DIG_COOLDOWN = 0.15;

/** Holding to dig: progress on the block under the crosshair, which starts over if the target changes. */
export class Digging {
  private target?: string;
  private cooldown = 0;
  /** How far the current block is dug, 0 to 1. */
  progress = 0;

  /**
   * Dig for `dt` seconds at `block` (a block of `type`), or stop if `block` is undefined or
   * the button isn't held. Returns true when the block breaks.
   */
  step(dt: number, held: boolean, block: readonly number[] | undefined, type: Block | undefined): boolean {
    this.cooldown = Math.max(0, this.cooldown - dt);
    if (!held || !block || type === undefined) {
      this.target = undefined;
      this.progress = 0;
      return false;
    }
    const key = block.join();
    if (key !== this.target) {
      this.target = key;
      this.progress = 0;
    }
    if (this.cooldown > 0) return false;
    const seconds = DIG_SECONDS[type] ?? 0.5;
    this.progress = seconds === 0 ? 1 : this.progress + dt / seconds;
    if (this.progress < 1) return false;
    this.target = undefined;
    this.progress = 0;
    this.cooldown = DIG_COOLDOWN;
    return true;
  }
}
