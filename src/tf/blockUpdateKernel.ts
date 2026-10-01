import * as tf from '@tensorflow/tfjs';
import { webgpu_util, type WebGPUBackend, type WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { DEFAULT_RATES, type PlantRates } from '../constants';
import { RULES_WGSL } from '../sim/rules';

// Every block-update rule fused into one WebGPU compute shader (the WGSL rules of
// sim/rules.ts), run as a TF.js custom kernel on a batch of chunks packed with their
// ghost borders. The tensor-op version (blockUpdate.ts) makes about a hundred full
// passes over the cells per tick; this makes one. It was the game's tick until the world
// moved into GPU memory (sim/gpuStore.ts runs the same WGSL where the cells live); the
// benchmark still times it for comparison. The GPU tests check it against
// blockUpdateReference.ts (the spec).
//
// Random numbers for the plant rules come from cellRandom() in the spec: an integer
// hash of the cell's input index and a per-tick seed, so they're exact on any GPU.

const KERNEL = 'FusedBlockUpdate';

const SHADER = /* wgsl */ `
${RULES_WGSL}

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
  // The random number for the plant rules: cellRandom(index of the cell in the input, seed).
  result[index] = nextCell(y, z, x, u32(itemBase + (y * dimD + z) * dimW + x));
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

