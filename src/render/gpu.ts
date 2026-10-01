/**
 * Request the WebGPU device used for both rendering and TF.js compute. TF.js's
 * kernels want the adapter's full compute limits, so ask for them up front.
 */
export async function requestGpu(): Promise<{ device: GPUDevice; info: GPUAdapterInfo }> {
  if (!navigator.gpu) throw new Error('WebGPU is not supported in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');
  const l = adapter.limits;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize,
      maxComputeWorkgroupsPerDimension: l.maxComputeWorkgroupsPerDimension,
      maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
      maxBufferSize: l.maxBufferSize,
      maxComputeWorkgroupSizeX: l.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup,
    },
  });
  device.lost.then((info) => console.error('WebGPU device lost:', info.message));
  return { device, info: adapter.info };
}
