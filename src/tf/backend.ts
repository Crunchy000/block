import * as tf from '@tensorflow/tfjs';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

/**
 * Pick the fastest available TF.js backend.
 *
 * If the renderer's GPUDevice is passed, TF.js's WebGPU backend runs on that
 * same device (one device for compute + rendering, and a path to sharing
 * buffers later). Falls back to WebGL, then CPU.
 */
export async function initTensorflow(device?: GPUDevice, adapterInfo?: GPUAdapterInfo): Promise<string> {
  if (device) {
    // Kernels are registered under the name 'webgpu', so replace that backend's factory.
    tf.removeBackend('webgpu');
    tf.registerBackend('webgpu', () => new WebGPUBackend(device, adapterInfo), 3);
  }
  for (const name of ['webgpu', 'webgl', 'cpu']) {
    try {
      if (!(await tf.setBackend(name))) continue;
      await tf.ready();
      await selfTest();
      return tf.getBackend();
    } catch (e) {
      console.warn(`tfjs backend ${name} failed`, e);
    }
  }
  throw new Error('No TensorFlow.js backend available');
}

/** Switch away from a backend that started failing at runtime. Returns the new backend name. */
export async function fallbackBackend(): Promise<string> {
  const order = ['webgpu', 'webgl', 'cpu'];
  for (const name of order.slice(order.indexOf(tf.getBackend()) + 1)) {
    try {
      if (!(await tf.setBackend(name))) continue;
      await tf.ready();
      await selfTest();
      return name;
    } catch (e) {
      console.warn(`tfjs backend ${name} failed`, e);
    }
  }
  return tf.getBackend();
}

/** Run a tiny int32 op round-trip so a broken backend is caught at startup. */
async function selfTest(): Promise<void> {
  const t = tf.tensor1d([1, 2, 3], 'int32').mod(2).add(1);
  const v = await t.data();
  t.dispose();
  if (v[0] !== 2 || v[1] !== 1) throw new Error('tfjs self-test returned wrong result');
}
