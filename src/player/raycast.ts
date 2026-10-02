import { Block, isSolid } from '../constants';

export interface RayHit {
  /** Block that was hit. */
  block: [number, number, number];
  /** Empty cell in front of the hit face (where a placed block goes). */
  before: [number, number, number];
  /** The block type hit. */
  type: Block;
}

/** Voxel DDA (Amanatides & Woo). Fluids are passed through; solid blocks and plants (wheat) are hit. */
export function raycast(
  getBlock: (x: number, y: number, z: number) => Block,
  origin: readonly number[], dir: readonly number[], maxDist: number,
): RayHit | null {
  const pos = [Math.floor(origin[0]), Math.floor(origin[1]), Math.floor(origin[2])];
  const step = dir.map((d) => (d > 0 ? 1 : d < 0 ? -1 : 0));
  const tDelta = dir.map((d) => (d !== 0 ? Math.abs(1 / d) : Infinity));
  const tMax = dir.map((d, i) => {
    if (d === 0) return Infinity;
    const boundary = d > 0 ? pos[i] + 1 : pos[i];
    return (boundary - origin[i]) / d;
  });
  let prev = [...pos];
  let t = 0;
  while (t <= maxDist) {
    const b = getBlock(pos[0], pos[1], pos[2]);
    if (isSolid(b) || b === Block.Wheat) {
      return { block: [pos[0], pos[1], pos[2]], before: [prev[0], prev[1], prev[2]], type: b };
    }
    prev = [...pos];
    const axis = tMax[0] < tMax[1] ? (tMax[0] < tMax[2] ? 0 : 2) : tMax[1] < tMax[2] ? 1 : 2;
    pos[axis] += step[axis];
    t = tMax[axis];
    tMax[axis] += tDelta[axis];
  }
  return null;
}
