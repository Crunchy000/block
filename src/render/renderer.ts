import type { MeshTarget } from '../sim/store';
import type { ChunkDraw } from '../world/world';
import { blockTexturesOnGpu, type BlockTextureData } from './blockTextures';
import { requestGpu } from './gpu';
import { MeshPool } from './meshPool';
import { MOB_VERTEX_FLOATS, mobModelOnGpu, type MobModel, type MobModelData } from './mobModel';
import { blockShader, farShader, lineShader, mobShader } from './shaders';
import { MIST_FOG, SKY, type FarDraw, type GameRenderer, type MobDraw } from './types';

/** Reversed depth (math.ts perspective): float depth keeps precision at any view distance. */
const DEPTH: GPUTextureFormat = 'depth32float';

export { SKY, MIST_FOG } from './types';
export type { FarDraw, MobDraw } from './types';

export interface RendererOptions {
  /**
   * Render into an offscreen texture and copy it to a 2D canvas instead of
   * presenting through a WebGPU canvas context. Slower; for headless
   * environments where canvas presentation is unavailable (screenshots, CI).
   */
  offscreen?: boolean;
  /**
   * Multisample antialiasing: 4 samples a pixel smooth the edges of blocks (most of all far
   * away, where they'd otherwise shimmer as the camera turns). On unless false.
   */
  msaa?: boolean;
}

export class Renderer implements GameRenderer {
  readonly api = 'WebGPU';
  private context?: GPUCanvasContext;
  private context2d?: CanvasRenderingContext2D;
  private colorTarget?: GPUTexture;
  private readback?: GPUBuffer;
  private reading = false;
  /** Offscreen mode only: a submitted frame the GPU hasn't finished yet. */
  private frameInFlight = false;
  private format!: GPUTextureFormat;
  private depth?: GPUTexture;
  /** Samples per pixel (4 with MSAA), and the multisampled colour target resolved into the frame. */
  private samples = 1;
  private msaaTarget?: GPUTexture;
  private uniformBuffer!: GPUBuffer;
  private layout!: GPUBindGroupLayout;
  private bindGroup?: GPUBindGroup;
  private bindGroupKey?: GPUBuffer;
  private blockTextures!: GPUTexture;
  private blockSampler!: GPUSampler;
  private textured = false;
  private opaquePipeline!: GPURenderPipeline;
  private waterPipeline!: GPURenderPipeline;
  private linePipeline!: GPURenderPipeline;
  private lineBuffer?: GPUBuffer;
  private farPipeline!: GPURenderPipeline;
  private farUniforms!: GPUBuffer;
  private farGroup?: GPUBindGroup;
  private farLayout!: GPUBindGroupLayout;
  private coverage?: GPUTexture;
  /** The far terrain's buffers, and which version of its mesh they hold. */
  private farVertex?: GPUBuffer;
  private farIndex?: GPUBuffer;
  private farVersion = -1;
  private mobPipeline!: GPURenderPipeline;
  private mobLayout!: GPUBindGroupLayout;
  private readonly mobGroups = new Map<MobModel, GPUBindGroup>();
  private mobInstances?: GPUBuffer;
  private mobSampler!: GPUSampler;

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
      const context2d = this.canvas.getContext('2d');
      if (!context2d) throw new Error("The browser gave no 2D canvas context (canvas.getContext('2d') returned null).");
      this.context2d = context2d;
      this.format = 'rgba8unorm';
    } else {
      // Browsers can hand out a GPU device but no canvas to show it on (null here),
      // for instance once they've switched WebGPU off after a GPU crash.
      const context = this.canvas.getContext('webgpu');
      if (!context) {
        throw new Error("The browser gave no WebGPU canvas (canvas.getContext('webgpu') returned null), although it gave a "
          + 'GPU device. Fully close and reopen the browser, then try again.');
      }
      this.context = context;
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
    }

    this.samples = this.options.msaa === false ? 1 : 4;
    this.uniformBuffer = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // Uniforms, the meshes' face records and chunk origins (read by the vertex shader), then the
    // block textures and their sampler (for the fragment shader).
    const uniformEntry: GPUBindGroupLayoutEntry = { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} };
    this.layout = device.createBindGroupLayout({
      entries: [
        uniformEntry,
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '2d-array' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    // Until the textures load (or if they don't), a blank layer, and blocks keep their procedural look.
    this.blockTextures = device.createTexture({ label: 'no block textures', size: [1, 1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
    // Pixel-art up close (nearest), smoothly mipmapped further away; repeating, as faces tile by world position.
    this.blockSampler = device.createSampler({
      magFilter: 'nearest', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat',
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });

    const blockModule = device.createShaderModule({ label: 'blocks', code: blockShader });
    this.opaquePipeline = device.createRenderPipeline({
      label: 'blocks (opaque)',
      layout: pipelineLayout,
      vertex: { module: blockModule, entryPoint: 'vsFace' },
      fragment: { module: blockModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'back', frontFace: 'ccw' },
      depthStencil: { format: DEPTH, depthWriteEnabled: true, depthCompare: 'greater' },
      multisample: { count: this.samples },
    });
    this.waterPipeline = device.createRenderPipeline({
      label: 'blocks (water)',
      layout: pipelineLayout,
      vertex: { module: blockModule, entryPoint: 'vsFace' },
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
      depthStencil: { format: DEPTH, depthWriteEnabled: false, depthCompare: 'greater' },
      multisample: { count: this.samples },
    });
    // Far terrain: its own small uniform block beside the shared one; plain vertex buffers.
    this.farUniforms = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const farLayout = this.farLayout = device.createBindGroupLayout({
      entries: [
        uniformEntry,
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    const farModule = device.createShaderModule({ label: 'far terrain', code: farShader });
    this.farPipeline = device.createRenderPipeline({
      label: 'far terrain',
      layout: device.createPipelineLayout({ bindGroupLayouts: [farLayout] }),
      vertex: { module: farModule, entryPoint: 'vs', buffers: [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
      fragment: { module: farModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: DEPTH, depthWriteEnabled: true, depthCompare: 'greater' },
      multisample: { count: this.samples },
    });

    // Mobs: textured, animated models, one instance per mob (plain vertex buffers and textures).
    this.mobLayout = device.createBindGroupLayout({
      entries: [
        uniformEntry,
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    this.mobSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat' });
    const mobModule = device.createShaderModule({ label: 'mobs', code: mobShader });
    this.mobPipeline = device.createRenderPipeline({
      label: 'mobs',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.mobLayout] }),
      vertex: {
        module: mobModule,
        entryPoint: 'vs',
        buffers: [
          {
            arrayStride: MOB_VERTEX_FLOATS * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x2' },
              { shaderLocation: 5, offset: 32, format: 'float32' },
            ],
          },
          {
            arrayStride: 32,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 3, offset: 0, format: 'float32x4' },
              { shaderLocation: 4, offset: 16, format: 'float32x4' },
            ],
          },
        ],
      },
      fragment: { module: mobModule, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: DEPTH, depthWriteEnabled: true, depthCompare: 'greater' },
      multisample: { count: this.samples },
    });

    const lineModule = device.createShaderModule({ label: 'lines', code: lineShader });
    this.linePipeline = device.createRenderPipeline({
      label: 'lines',
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
      depthStencil: { format: DEPTH, depthWriteEnabled: false, depthCompare: 'greater-equal' },
      multisample: { count: this.samples },
    });
  }

  private groupFor(meshes: MeshPool): GPUBindGroup {
    // (A MeshPool's face buffer is replaced when it grows.)
    if (this.bindGroup && this.bindGroupKey === meshes.faces) return this.bindGroup;
    this.bindGroupKey = meshes.faces;
    const buffers = [this.uniformBuffer, meshes.faces, meshes.origins];
    return this.bindGroup = this.device.createBindGroup({
      layout: this.layout,
      entries: [
        ...buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        { binding: buffers.length, resource: this.blockTextures.createView({ dimension: '2d-array' }) },
        { binding: buffers.length + 1, resource: this.blockSampler },
      ],
    });
  }

  get lost(): Promise<{ reason: string; message: string }> {
    return this.device.lost.then((info) => ({ reason: info.reason, message: info.message }));
  }

  get gpuName(): string {
    const i = this.adapterInfo;
    return [i.vendor, i.architecture || i.device || i.description].filter(Boolean).join(' ') || 'unknown';
  }

  /** Chunk meshes: face records in GPU memory, built there (or uploaded, with the world on the CPU). */
  createMeshes(slots: number): MeshPool {
    return new MeshPool(this.device, slots);
  }

  createMobModel(data: MobModelData): MobModel {
    return mobModelOnGpu(this.device, data);
  }

  /** Texture the blocks with these layers (render/blockTextures.ts) from now on. */
  setBlockTextures(data: BlockTextureData): void {
    this.blockTextures.destroy();
    this.blockTextures = blockTexturesOnGpu(this.device, data);
    this.textured = true;
    this.bindGroup = undefined;
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h || !this.depth) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.depth?.destroy();
      this.depth = this.device.createTexture({ size: [w, h], format: DEPTH, usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: this.samples });
      this.msaaTarget?.destroy();
      this.msaaTarget = this.samples > 1
        ? this.device.createTexture({ size: [w, h], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: this.samples })
        : undefined;
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
   * stay on the GPU), plus lines: interleaved [x, y, z, r, g, b] pairs for a line list,
   * and the far terrain beyond them if given.
   */
  render(
    viewProj: Float32Array, cam: readonly number[], time: number, fogDistance: number, lines: Float32Array<ArrayBuffer>,
    meshes: MeshTarget, draws: ChunkDraw[], far?: FarDraw, mobDraws: MobDraw[] = [],
  ): void {
    const pool = meshes as MeshPool;
    const mobs = mobDraws as Array<{ model: MobModel; instances: Float32Array<ArrayBuffer> }>;
    // A canvas throttles us to what the GPU can present; offscreen nothing does, and on a
    // slow GPU frames would pile up in the queue ahead of the block-update work. Skip
    // frames until the previous one is done.
    if (this.options.offscreen && this.frameInFlight) return;
    this.resize();
    const { device } = this;
    const u = new Float32Array(28);
    u.set(viewProj, 0);
    u.set([cam[0], cam[1], cam[2], time], 16);
    u.set([...SKY, fogDistance], 20);
    u.set([far?.look === 'mist' ? MIST_FOG : 1, this.textured ? 1 : 0, 0, 0], 24);
    device.queue.writeBuffer(this.uniformBuffer, 0, u);

    if (lines.length > 0 && (!this.lineBuffer || this.lineBuffer.size < lines.byteLength)) {
      this.lineBuffer?.destroy();
      this.lineBuffer = device.createBuffer({ size: Math.max(lines.byteLength, 1 << 16), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    if (lines.length > 0) device.queue.writeBuffer(this.lineBuffer!, 0, lines);
    // All mobs' instances in one buffer, each kind a run of it.
    const mobFloats = mobs.reduce((n, m) => n + m.instances.length, 0);
    if (mobFloats > 0) {
      if (!this.mobInstances || this.mobInstances.size < mobFloats * 4) {
        this.mobInstances?.destroy();
        this.mobInstances = device.createBuffer({ size: Math.max(mobFloats * 4, 32 * 32), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      }
      let at = 0;
      for (const m of mobs) {
        device.queue.writeBuffer(this.mobInstances, at * 4, m.instances);
        at += m.instances.length;
        if (!this.mobGroups.has(m.model)) {
          this.mobGroups.set(m.model, device.createBindGroup({
            layout: this.mobLayout,
            entries: [
              { binding: 0, resource: { buffer: this.uniformBuffer } },
              { binding: 1, resource: m.model.texture.createView() },
              { binding: 2, resource: this.mobSampler },
              { binding: 3, resource: m.model.poses.createView() },
            ],
          }));
        }
      }
    }
    if (far) {
      if (!this.farVertex || this.farVertex.size !== far.vertices.byteLength) {
        this.farVertex?.destroy();
        this.farIndex?.destroy();
        this.farVertex = device.createBuffer({ label: 'far terrain vertices', size: far.vertices.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        this.farIndex = device.createBuffer({ label: 'far terrain indices', size: far.indices.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.farIndex, 0, far.indices);
        this.farVersion = -1;
      }
      if (far.version !== this.farVersion) {
        device.queue.writeBuffer(this.farVertex, 0, far.vertices);
        this.farVersion = far.version;
      }
      const { x0, z0, size, data } = far.coverage;
      device.queue.writeBuffer(this.farUniforms, 0, new Float32Array([x0 * 16, z0 * 16, size, 0, far.seaY, ['colour', 'silhouette', 'mist'].indexOf(far.look), far.extent, 0]));
      // The coverage map: one byte a chunk, re-made when its size changes.
      if (!this.coverage || this.coverage.width !== size) {
        this.coverage?.destroy();
        this.coverage = device.createTexture({ label: 'chunk coverage', size: [size, size], format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        this.farGroup = device.createBindGroup({
          layout: this.farLayout,
          entries: [
            { binding: 0, resource: { buffer: this.uniformBuffer } },
            { binding: 1, resource: { buffer: this.farUniforms } },
            { binding: 2, resource: this.coverage.createView() },
          ],
        });
      }
      device.queue.writeTexture({ texture: this.coverage }, data, { bytesPerRow: size }, [size, size]);
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        // With MSAA: draw into the multisampled target and resolve it into the frame.
        ...(this.msaaTarget
          ? { view: this.msaaTarget.createView(), resolveTarget: (this.colorTarget ?? this.context!.getCurrentTexture()).createView(), storeOp: 'discard' as const }
          : { view: (this.colorTarget ?? this.context!.getCurrentTexture()).createView(), storeOp: 'store' as const }),
        clearValue: { r: SKY[0], g: SKY[1], b: SKY[2], a: 1 },
        loadOp: 'clear',
      }],
      depthStencilAttachment: { view: this.depth!.createView(), depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setBindGroup(0, this.groupFor(pool));

    // Each chunk's run of face records.
    const drawMesh = (d: ChunkDraw, water: boolean) => {
      const m = pool.get(d.meshSlot);
      const run = water ? m?.water : m?.opaque;
      if (run?.count) pass.draw(run.count * 6, 1, run.start * 6, d.meshSlot);
    };
    pass.setPipeline(this.opaquePipeline);
    for (const d of draws) drawMesh(d, false);

    if (far) {
      pass.setPipeline(this.farPipeline);
      pass.setBindGroup(0, this.farGroup!);
      pass.setVertexBuffer(0, this.farVertex!);
      pass.setIndexBuffer(this.farIndex!, 'uint32');
      pass.drawIndexed(far.indices.length);
      pass.setBindGroup(0, this.groupFor(pool));
    }

    if (mobFloats > 0) {
      pass.setPipeline(this.mobPipeline);
      pass.setVertexBuffer(1, this.mobInstances!);
      let first = 0;
      for (const m of mobs) {
        const count = m.instances.length / 8;
        pass.setBindGroup(0, this.mobGroups.get(m.model)!);
        pass.setVertexBuffer(0, m.model.vertex);
        pass.setIndexBuffer(m.model.index, 'uint16');
        pass.drawIndexed(m.model.indexCount, count, 0, 0, first);
        first += count;
      }
      pass.setBindGroup(0, this.groupFor(pool));
    }

    if (lines.length > 0) {
      pass.setPipeline(this.linePipeline);
      pass.setVertexBuffer(0, this.lineBuffer!);
      pass.draw(lines.length / 6);
    }

    // Translucent water, far chunks first.
    const d2 = (p: number[]) => (p[0] - cam[0]) ** 2 + (p[1] - cam[1]) ** 2 + (p[2] - cam[2]) ** 2;
    pass.setPipeline(this.waterPipeline);
    for (const d of [...draws].sort((a, b) => d2(b.center) - d2(a.center))) drawMesh(d, true);

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
