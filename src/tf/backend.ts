import * as tf from '@tensorflow/tfjs';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { CHUNK_HEIGHT, CHUNK_SIZE } from '../constants';
import { blockUpdateStep } from './blockUpdate';
import { GEN_BATCH, generateChunksTensor } from './worldgen';

/**
 * Pick the fastest available TF.js backend.
 *
 * If the renderer's GPUDevice is passed, TF.js's WebGPU backend runs on that
 * same device (one device for compute + rendering, and a path to sharing
 * buffers later). Falls back to WebGL, then CPU.
 */
export async function initTensorflow(device?: GPUDevice, adapterInfo?: GPUAdapterInfo): Promise<string> {
  // WebGL bakes tensor shapes into shaders unless told to pass them as uniforms;
  // with uniforms, differently sized regions reuse the same compiled programs.
  tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true);
  if (device) {
    // Kernels are registered under the name 'webgpu', so replace that backend's factory.
    tf.removeBackend('webgpu');
    tf.registerBackend('webgpu', () => new WebGPUBackend(device, adapterInfo), 3);
  }
  for (const name of ['webgpu', 'webgl', 'cpu']) {
    if (await trySetBackend(name)) return name;
  }
  throw new Error('No TensorFlow.js backend available');
}

/** Switch away from a backend that started failing at runtime. Returns the new backend name. */
export async function fallbackBackend(): Promise<string> {
  const order = ['webgpu', 'webgl', 'cpu'];
  for (const name of order.slice(order.indexOf(tf.getBackend()) + 1)) {
    if (await trySetBackend(name)) return name;
  }
  return tf.getBackend();
}

async function trySetBackend(name: string): Promise<boolean> {
  try {
    if (!(await tf.setBackend(name))) return false;
    await tf.ready();
    await selfTest();
    return true;
  } catch (e) {
    console.warn(`tfjs backend ${name} failed`, e);
    return false;
  }
}

/** Run a tiny int32 op round-trip so a broken backend is caught at startup. */
async function selfTest(): Promise<void> {
  const t = tf.tensor1d([1, 2, 3], 'int32').mod(2).add(1);
  const v = await t.data();
  t.dispose();
  if (v[0] !== 2 || v[1] !== 1) throw new Error('tfjs self-test returned wrong result');
}

/**
 * Compile every GPU kernel that world generation and block updates use before
 * play starts. Otherwise each kernel compiles the first time it runs, which
 * blocks the page (badly so on some drivers) in the middle of loading.
 *
 * Kernels are cached by tensor rank and dtype rather than exact size, so
 * compiling with representative shapes covers every later run. In compile-only
 * mode the GPU backends build their pipelines in parallel without executing
 * anything or blocking the main thread.
 */
export async function warmUpKernels(): Promise<void> {
  const backend = tf.getBackend();
  const flag = backend === 'webgpu' ? 'WEBGPU_ENGINE_COMPILE_ONLY' : backend === 'webgl' ? 'ENGINE_COMPILE_ONLY' : null;
  if (!flag) return; // CPU has nothing to compile
  const coords = Array.from({ length: GEN_BATCH }, (_, i) => ({ cx: i, cz: 0 }));
  // A 3x3-chunk region: the smallest a tick ever simulates (one chunk + ghost border).
  const region = [CHUNK_HEIGHT, 3 * CHUNK_SIZE, 3 * CHUNK_SIZE] as [number, number, number];
  tf.env().set(flag, true);
  try {
    tf.tidy(() => {
      generateChunksTensor(coords);
      blockUpdateStep(tf.zeros(region, 'int32')).slice([0, CHUNK_SIZE, CHUNK_SIZE], [CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_SIZE]);
    });
    await (tf.backend() as unknown as { checkCompileCompletionAsync(): Promise<unknown> }).checkCompileCompletionAsync();
  } finally {
    tf.env().set(flag, false);
  }
}
