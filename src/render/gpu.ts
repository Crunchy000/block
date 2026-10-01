import { log } from '../log';

/**
 * Request the WebGPU device used for both rendering and TF.js compute. TF.js's
 * kernels want the adapter's full compute limits, so ask for them up front.
 * What the browser offers, and anything that later goes wrong with the device,
 * goes into the page log.
 */
export async function requestGpu(): Promise<{ device: GPUDevice; info: GPUAdapterInfo }> {
  if (!navigator.gpu) {
    throw new Error('This browser has no WebGPU (navigator.gpu is missing). It needs a recent Chrome or Edge, '
      + 'Safari 26 or Firefox 141 or newer, and some phones still have it switched off.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Error('WebGPU is switched off right now: the browser offered no GPU adapter. Browsers do this '
      + 'for a while after the GPU crashes or hangs. Fully close and reopen the browser, then try again.');
  }
  log.info('GPU adapter', describeAdapter(adapter));
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
  watchDevice(device);
  return { device, info: adapter.info };
}

/** Everything the browser says about the adapter: names, features and limits. */
function describeAdapter(adapter: GPUAdapter): string {
  const info = adapter.info as GPUAdapterInfo | undefined;
  const names = info
    ? ['vendor', 'architecture', 'device', 'description'].map((k) => `${k}=${(info as unknown as Record<string, string>)[k] || '?'}`).join(' ')
    : 'no adapter info';
  const fallback = (info as { isFallbackAdapter?: boolean } | undefined)?.isFallbackAdapter ? ' (fallback adapter)' : '';
  const limits: string[] = [];
  for (const key in adapter.limits) {
    const value = (adapter.limits as unknown as Record<string, unknown>)[key];
    if (typeof value === 'number') limits.push(`${key}=${value}`);
  }
  return `${names}${fallback}\nfeatures: ${[...adapter.features].join(', ') || 'none'}\nlimits: ${limits.join(', ')}`;
}

/**
 * Log what goes wrong with the device: losing it (a GPU crash or hang, a driver update, the
 * browser reclaiming it), errors nothing caught (validation, out of memory), and shader
 * compile errors and warnings with their line, for every shader including TF.js's.
 */
function watchDevice(device: GPUDevice): void {
  device.lost.then((info) => log.error(`WebGPU device lost (reason: ${info.reason})`, info.message || 'no message'));
  device.addEventListener('uncapturederror', (event) => {
    const error = (event as GPUUncapturedErrorEvent).error;
    log.error(`WebGPU ${error.constructor.name}`, error.message);
  });
  const createShaderModule = device.createShaderModule.bind(device);
  device.createShaderModule = (descriptor: GPUShaderModuleDescriptor) => {
    const module = createShaderModule(descriptor);
    module.getCompilationInfo?.().then((info) => {
      for (const m of info.messages) {
        if (m.type === 'info') continue;
        const line = descriptor.code.split('\n')[m.lineNum - 1] ?? '';
        const where = `shader "${descriptor.label || descriptor.code.trim().split('\n')[0].slice(0, 60)}", line ${m.lineNum}:${m.linePos}`;
        (m.type === 'error' ? log.error : log.warn)(`WGSL ${m.type} in ${where}`, `${m.message}\n> ${line.trim()}`);
      }
    }).catch(() => {});
    return module;
  };
}
