import * as tf from '@tensorflow/tfjs';
import { webgpu_util, type WebGPUBackend, type WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import {
  Block, DEFAULT_RATES, FALLING_LEVEL, LAVA_DECAY, LEVEL_MUL, PRIMED_DIRT, SOURCE_LEVEL, WATER_DECAY, WHEAT_RIPE,
  type PlantRates,
} from '../constants';
import { GRASS_REACH, PCG, WHEAT_WATER_REACH } from './blockUpdateReference';

// Every block-update rule fused into one WebGPU compute shader, run as a TF.js custom
// kernel. The tensor-op version (blockUpdate.ts) makes about a hundred full passes over
// the cells per tick, each its own GPU dispatch issued from the main thread; this makes
// one. Same rules as blockUpdateReference.ts (the spec) and blockUpdate.ts (used on
// backends other than WebGPU); the GPU tests check that all three agree.
//
// Random numbers for the plant rules come from cellRandom() in the spec: an integer
// hash of the cell's input index and a per-tick seed, so they're exact on any GPU.

const KERNEL = 'FusedBlockUpdate';

const SHADER = /* wgsl */ `
const AIR: i32 = ${Block.Air};
const STONE: i32 = ${Block.Stone};
const DIRT: i32 = ${Block.Dirt};
const WATER: i32 = ${Block.Water};
const LAVA: i32 = ${Block.Lava};
const GRASS: i32 = ${Block.Grass};
const WHEAT: i32 = ${Block.Wheat};
const LEVEL_MUL: i32 = ${LEVEL_MUL};
const SOURCE: i32 = ${SOURCE_LEVEL};
const FALLING: i32 = ${FALLING_LEVEL};
const RIPE: i32 = ${WHEAT_RIPE};
const PRIMED: i32 = ${PRIMED_DIRT};

// The batch item being updated: [H, D, W] cells starting at itemBase in the input.
var<private> dimH: i32;
var<private> dimD: i32;
var<private> dimW: i32;
var<private> itemBase: i32;

// Cells past the item's sides or top read as air, below y = 0 as bedrock.
fn cellAt(y: i32, z: i32, x: i32) -> i32 {
  if (x < 0 || x >= dimW || z < 0 || z >= dimD || y >= dimH) { return AIR; }
  if (y < 0) { return STONE; }
  return cells[itemBase + (y * dimD + z) * dimW + x];
}
fn inItem(y: i32, z: i32, x: i32) -> bool {
  return x >= 0 && x < dimW && z >= 0 && z < dimD && y >= 0 && y < dimH;
}
fn blockOf(c: i32) -> i32 { return c & 7; }
fn levelOf(c: i32) -> i32 { return c >> 3u; }
fn fluidLevel(c: i32, fluid: i32) -> i32 { return select(0, levelOf(c), blockOf(c) == fluid); }
fn isSolidCell(c: i32) -> bool {
  let t = blockOf(c);
  return t == STONE || t == DIRT || t == GRASS;
}
// Holds up fluid resting on it, and smothers grass under it.
fn supportsCell(c: i32) -> bool { return isSolidCell(c) || levelOf(c) == SOURCE; }

// What a fluid cell at (y, z, x) passes sideways: level - decay, if it rests on something.
fn emitFrom(y: i32, z: i32, x: i32, fluid: i32, decay: i32) -> i32 {
  if (x < 0 || x >= dimW || z < 0 || z >= dimD) { return 0; }
  if (!supportsCell(cellAt(y - 1, z, x))) { return 0; }
  return max(fluidLevel(cellAt(y, z, x), fluid) - decay, 0);
}
// The level a fluid wants to put in (y, z, x): from the sides, or falling from above.
fn wantOf(y: i32, z: i32, x: i32, fluid: i32, decay: i32) -> i32 {
  var w = max(max(emitFrom(y, z, x - 1, fluid, decay), emitFrom(y, z, x + 1, fluid, decay)),
              max(emitFrom(y, z - 1, x, fluid, decay), emitFrom(y, z + 1, x, fluid, decay)));
  if (fluidLevel(cellAt(y + 1, z, x), fluid) > 0) { w = max(w, FALLING); }
  return w;
}
// A cell fluid flows into: stone where water and lava meet.
fn flowed(ww: i32, wl: i32) -> i32 {
  if (ww > 0 && wl > 0) { return STONE; }
  if (ww > 0) { return WATER + LEVEL_MUL * ww; }
  if (wl > 0) { return LAVA + LEVEL_MUL * wl; }
  return AIR;
}
fn aliveGrassAt(y: i32, z: i32, x: i32) -> bool {
  return inItem(y, z, x) && blockOf(cellAt(y, z, x)) == GRASS && !supportsCell(cellAt(y + 1, z, x));
}
fn grassInReach(y: i32, z: i32, x: i32) -> bool {
  for (var dy = ${GRASS_REACH.dy[0]}; dy <= ${GRASS_REACH.dy[1]}; dy++) {
    for (var dz = ${GRASS_REACH.dz[0]}; dz <= ${GRASS_REACH.dz[1]}; dz++) {
      for (var dx = ${GRASS_REACH.dx[0]}; dx <= ${GRASS_REACH.dx[1]}; dx++) {
        if (aliveGrassAt(y + dy, z + dz, x + dx)) { return true; }
      }
    }
  }
  return false;
}
fn wetNear(y: i32, z: i32, x: i32) -> bool {
  for (var dy = ${WHEAT_WATER_REACH.dy[0]}; dy <= ${WHEAT_WATER_REACH.dy[1]}; dy++) {
    for (var dz = ${WHEAT_WATER_REACH.dz[0]}; dz <= ${WHEAT_WATER_REACH.dz[1]}; dz++) {
      for (var dx = ${WHEAT_WATER_REACH.dx[0]}; dx <= ${WHEAT_WATER_REACH.dx[1]}; dx++) {
        if (inItem(y + dy, z + dz, x + dx) && fluidLevel(cellAt(y + dy, z + dz, x + dx), WATER) > 0) { return true; }
      }
    }
  }
  return false;
}
fn pcgHash(v: u32) -> u32 {
  let state = v * ${PCG.mul}u + ${PCG.inc}u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * ${PCG.out}u;
  return (word >> 22u) ^ word;
}
fn cellRandom(index: i32) -> f32 {
  return f32(pcgHash(u32(index) ^ uniforms.seed) >> 8u) / 16777216.0;
}

fn main(index : i32) {
  if (index >= uniforms.size) { return; }
  dimH = uniforms.cellsShape.y;
  dimD = uniforms.cellsShape.z;
  dimW = uniforms.cellsShape.w;
  // The output may leave out a ghost border of this width on each side.
  let halo = (dimD - uniforms.outShape.z) / 2;
  let o = getCoordsFromIndex(index);
  itemBase = o.x * dimH * dimD * dimW;
  let y = o.y;
  let z = o.z + halo;
  let x = o.w + halo;
  let cellIndex = itemBase + (y * dimD + z) * dimW + x;
  let c = cells[cellIndex];
  let t = blockOf(c);
  var next = c;

  // Air and flowing fluid are recomputed from their neighbours.
  if (t == AIR || ((t == WATER || t == LAVA) && levelOf(c) < SOURCE)) {
    next = flowed(wantOf(y, z, x, WATER, ${WATER_DECAY}), wantOf(y, z, x, LAVA, ${LAVA_DECAY}));
  }
  // Lava with water beside or above it turns to stone.
  if (t == LAVA && levelOf(c) > 0 && (
      fluidLevel(cellAt(y, z, x - 1), WATER) > 0 || fluidLevel(cellAt(y, z, x + 1), WATER) > 0 ||
      fluidLevel(cellAt(y, z - 1, x), WATER) > 0 || fluidLevel(cellAt(y, z + 1, x), WATER) > 0 ||
      fluidLevel(cellAt(y + 1, z, x), WATER) > 0)) {
    next = STONE;
  }

  if (t == WHEAT) {
    // Flowing fluid washes wheat away; otherwise it pops off without soil, or may grow.
    let ww = wantOf(y, z, x, WATER, ${WATER_DECAY});
    let wl = wantOf(y, z, x, LAVA, ${LAVA_DECAY});
    if (ww > 0 || wl > 0) {
      next = flowed(ww, wl);
    } else if (uniforms.plants != 0) {
      let soil = blockOf(cellAt(y - 1, z, x));
      if (soil != DIRT && soil != GRASS) {
        next = AIR;
      } else {
        let stage = levelOf(c);
        let chance = select(uniforms.wheatGrow, uniforms.wheatGrowWet, wetNear(y, z, x));
        if (stage < RIPE && cellRandom(cellIndex) < chance) { next = WHEAT + LEVEL_MUL * (stage + 1); }
      }
    }
  } else if (uniforms.plants != 0 && t == GRASS) {
    // Covered grass dies back to dirt.
    next = select(DIRT, GRASS, aliveGrassAt(y, z, x));
  } else if (uniforms.plants != 0 && t == DIRT) {
    // Exposed dirt near living grass is primed, and sprouts with the spread chance.
    let primed = uniforms.grassSpread > 0.0 && blockOf(cellAt(y + 1, z, x)) == AIR && grassInReach(y, z, x);
    if (primed && cellRandom(cellIndex) < uniforms.grassSpread) {
      next = GRASS;
    } else {
      next = select(DIRT, PRIMED, primed);
    }
  }
  result[index] = next;
}
`;

class FusedBlockUpdateProgram implements WebGPUProgram {
  variableNames = ['cells'];
  outputShape: number[];
  shaderKey = 'fusedBlockUpdate';
  dispatchLayout: { x: number[] };
  dispatch: [number, number, number];
  workgroupSize: [number, number, number] = [64, 1, 1];
  size = true;
  uniforms = 'seed : u32, grassSpread : f32, wheatGrow : f32, wheatGrowWet : f32, plants : i32,';

  constructor(outputShape: number[]) {
    this.outputShape = outputShape;
    this.dispatchLayout = webgpu_util.flatDispatchLayout(outputShape);
    this.dispatch = webgpu_util.computeDispatch(this.dispatchLayout, outputShape, this.workgroupSize);
  }

  getUserCode(): string {
    return SHADER;
  }
}

interface FusedAttrs {
  seed: number;
  plants: boolean;
  grassSpread: number;
  wheatGrow: number;
  wheatGrowWet: number;
  halo: number;
}

tf.registerKernel({
  kernelName: KERNEL,
  backendName: 'webgpu',
  kernelFunc: ({ inputs, backend, attrs }) => {
    const { cells } = inputs as { cells: tf.TensorInfo };
    const a = attrs as unknown as FusedAttrs;
    const [n, h, d, w] = cells.shape;
    const program = new FusedBlockUpdateProgram([n, h, d - 2 * a.halo, w - 2 * a.halo]);
    return (backend as WebGPUBackend).runWebGPUProgram(program, [cells], 'int32', [
      { type: 'uint32', data: [a.seed] },
      { type: 'float32', data: [a.grassSpread] },
      { type: 'float32', data: [a.wheatGrow] },
      { type: 'float32', data: [a.wheatGrowWet] },
      { type: 'int32', data: [a.plants ? 1 : 0] },
    ]);
  },
});

export interface FusedOptions {
  /** Per-tick seed for the plant rules' random numbers (see cellRandom in the spec). */
  seed: number;
  /** Run the plant rules (grass, wheat). Without them grass, dirt and wheat stay as they are. */
  plants: boolean;
  rates?: PlantRates;
  /** Width of a ghost border to leave out of the output (0: output has the input's shape). */
  halo?: number;
}

/** The fused kernel only exists for the WebGPU backend. */
export const fusedAvailable = () => tf.getBackend() === 'webgpu';

/** One block-update tick over a batch of cells [N, H, D, W] with the fused WebGPU kernel. */
export function blockUpdateFused(cells: tf.Tensor4D, options: FusedOptions): tf.Tensor4D {
  const rates = options.rates ?? DEFAULT_RATES;
  const attrs: FusedAttrs = {
    seed: options.seed >>> 0,
    plants: options.plants,
    grassSpread: rates.grassSpread,
    wheatGrow: rates.wheatGrow,
    wheatGrowWet: rates.wheatGrowWet,
    halo: options.halo ?? 0,
  };
  return tf.engine().runKernel(KERNEL, { cells }, attrs as unknown as tf.NamedAttrMap) as tf.Tensor4D;
}

/** A fresh per-tick seed. */
export const randomSeed = () => (Math.random() * 2 ** 32) >>> 0;
