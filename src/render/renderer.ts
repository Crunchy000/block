import type { ChunkDraw } from '../world/world';
import { requestGpu } from './gpu';
import type { MeshPool } from './meshPool';
import { blockShader, lineShader } from './shaders';

export const SKY: [number, number, number] = [0.55, 0.75, 0.95];

export interface RendererOptions {
  /**
   * Render into an offscreen texture and copy it to a 2D canvas instead of
   * presenting through a WebGPU canvas context. Slower; for headless
   * environments where canvas presentation is unavailable (screenshots, CI).
   */
  offscreen?: boolean;
}

export class Renderer {
  private context?: GPUCanvasContext;
  private context2d?: CanvasRenderingContext2D;
  private colorTarget?: GPUTexture;
  private readback?: GPUBuffer;
  private reading = false;
  /** Offscreen mode only: a submitted frame the GPU hasn't finished yet. */
  private frameInFlight = false;
  private format!: GPUTextureFormat;
  private depth?: GPUTexture;
  private uniformBuffer!: GPUBuffer;
  private layout!: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private bindGroupPool?: MeshPool;
  private opaquePipeline!: GPURenderPipeline;
  private waterPipeline!: GPURenderPipeline;
  private linePipeline!: GPURenderPipeline;
  private lineBuffer?: GPUBuffer;

  private constructor(
    readonly device: GPUDevice,
    readonly adapterInfo: GPUAdapterInfo,
    readonly canvas: HTMLCanvasElement,
    readonly options: RendererOptions,
  ) {}

  static async create(canvas: HTMLCanvasElement, options: RendererOptions = {}): Promise<Renderer> {
    const { device, info } = await requestGpu();
    const r = new Renderer(device, info, canvas, options);
    r.init();
    return r;
  }

  private init(): void {
    const { device } = this;
    if (this.options.offscreen) {
      this.context2d = this.canvas.getContext('2d')!;
      this.format = 'rgba8unorm';
    } else {
      this.context = this.canvas.getContext('webgpu')!;
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
    }

    this.uniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Uniforms, then the meshes' face records and chunk origins (read by the vertex shader).
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });

    const blockModule = device.createShaderModule({ code: blockShader });
    this.opaquePipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: blockModule, entryPoint: 'vsOpaque' },
      fragment: { module: blockModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.waterPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: blockModule, entryPoint: 'vsWater' },
      fragment: {
        module: blockModule,
        entryPoint: 'fs',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });
    const lineModule = device.createShaderModule({ code: lineShader });
    this.linePipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: lineModule,
        entryPoint: 'vs',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: lineModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
    });
  }

  private groupFor(pool: MeshPool): GPUBindGroup {
    if (this.bindGroup && this.bindGroupPool === pool) return this.bindGroup;
    this.bindGroupPool = pool;
    return this.bindGroup = this.device.createBindGroup({
      layout: this.layout,
      entries: [this.uniformBuffer, pool.faces, pool.origins].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h || !this.depth) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.depth?.destroy();
      this.depth = this.device.createTexture({ size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
      if (this.options.offscreen) {
        this.colorTarget?.destroy();
        this.colorTarget = this.device.createTexture({
          size: [w, h], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
      }
    }
  }

  /** Offscreen mode: copy the rendered frame into the 2D canvas (skipped while a copy is in flight). */
  private present(encoder: GPUCommandEncoder): (() => void) | undefined {
    if (!this.colorTarget || this.reading) return undefined;
    const { width: w, height: h } = this.colorTarget;
    const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
    if (!this.readback || this.readback.size !== bytesPerRow * h) {
      this.readback?.destroy();
      this.readback = this.device.createBuffer({ size: bytesPerRow * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    const buffer = this.readback;
    encoder.copyTextureToBuffer({ texture: this.colorTarget }, { buffer, bytesPerRow }, [w, h]);
    this.reading = true;
    return () => {
      buffer.mapAsync(GPUMapMode.READ).then(() => {
        const src = new Uint8Array(buffer.getMappedRange());
        const img = new ImageData(w, h);
        for (let y = 0; y < h; y++) img.data.set(src.subarray(y * bytesPerRow, y * bytesPerRow + w * 4), y * w * 4);
        buffer.unmap();
        this.context2d!.putImageData(img, 0, 0);
      }).catch(() => {}).finally(() => { this.reading = false; });
    };
  }

  get aspect(): number {
    return this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
  }

  /**
   * Draw the chunks in `draws` from their meshes in `pool` (indirect draws: the counts
   * stay on the GPU), plus lines: interleaved [x, y, z, r, g, b] pairs for a line list.
   */
  render(
    viewProj: Float32Array, cam: readonly number[], time: number, fogDistance: number, lines: Float32Array<ArrayBuffer>,
    pool: MeshPool, draws: ChunkDraw[],
  ): void {
    // A canvas throttles us to what the GPU can present; offscreen nothing does, and on a
    // slow GPU frames would pile up in the queue ahead of the block-update work. Skip
    // frames until the previous one is done.
    if (this.options.offscreen && this.frameInFlight) return;
    this.resize();
    const { device } = this;
    const u = new Float32Array(24);
    u.set(viewProj, 0);
    u.set([cam[0], cam[1], cam[2], time], 16);
    u.set([...SKY, fogDistance], 20);
    device.queue.writeBuffer(this.uniformBuffer, 0, u);

    if (lines.length > 0 && (!this.lineBuffer || this.lineBuffer.size < lines.byteLength)) {
      this.lineBuffer?.destroy();
      this.lineBuffer = device.createBuffer({ size: Math.max(lines.byteLength, 1 << 16), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    if (lines.length > 0) device.queue.writeBuffer(this.lineBuffer!, 0, lines);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: (this.colorTarget ?? this.context!.getCurrentTexture()).createView(),
        clearValue: { r: SKY[0], g: SKY[1], b: SKY[2], a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: { view: this.depth!.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setBindGroup(0, this.groupFor(pool));

    pass.setPipeline(this.opaquePipeline);
    for (const d of draws) pass.drawIndirect(pool.draws, pool.drawOffset(d.meshSlot));

    if (lines.length > 0) {
      pass.setPipeline(this.linePipeline);
      pass.setVertexBuffer(0, this.lineBuffer!);
      pass.draw(lines.length / 6);
    }

    // Translucent water, far chunks first.
    const d2 = (p: number[]) => (p[0] - cam[0]) ** 2 + (p[1] - cam[1]) ** 2 + (p[2] - cam[2]) ** 2;
    pass.setPipeline(this.waterPipeline);
    for (const d of [...draws].sort((a, b) => d2(b.center) - d2(a.center))) {
      pass.drawIndirect(pool.draws, pool.drawOffset(d.meshSlot) + 16);
    }

    pass.end();
    const afterSubmit = this.present(encoder);
    device.queue.submit([encoder.finish()]);
    afterSubmit?.();
    if (this.options.offscreen) {
      this.frameInFlight = true;
      device.queue.onSubmittedWorkDone().finally(() => { this.frameInFlight = false; });
    }
  }
}
