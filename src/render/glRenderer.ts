import { log } from '../log';
import type { ChunkFaces } from './mesher';
import type { MeshTarget } from '../sim/store';
import type { ChunkDraw } from '../world/world';
import { TEXTURE_SIZE, type BlockTextureData } from './blockTextures';
import { VERTEX_FLOATS, classicMeshData } from './classicMeshes';
import {
  glBlockFragment, glBlockVertex, glFarFragment, glFarVertex, glLineFragment, glLineVertex, glMobFragment, glMobVertex,
} from './glShaders';
import { MOB_VERTEX_FLOATS, type MobModelData } from './mobModel';
import { MIST_FOG, SKY, type FarDraw, type GameRenderer, type MobDraw, type MobModelHandle } from './types';

/** A chunk's mesh in WebGL: its vertex array (vertices and indices bound) and how many indices. */
interface GlMesh { vao: WebGLVertexArrayObject; buffers: WebGLBuffer[]; count: number }

/** Chunk meshes for WebGL2: face records built into plain vertices on the CPU, as safe mode does. */
export class GlMeshes implements MeshTarget {
  private readonly slots: Array<{ opaque?: GlMesh; water?: GlMesh } | undefined>;

  constructor(private readonly gl: WebGL2RenderingContext, slots: number) {
    this.slots = new Array(slots).fill(undefined);
  }

  upload(slot: number, cx: number, cz: number, faces: ChunkFaces): void {
    this.free(slot);
    this.slots[slot] = { opaque: this.build(faces.opaque, cx, cz), water: this.build(faces.water, cx, cz) };
  }

  get(slot: number) {
    return this.slots[slot];
  }

  private build(records: number[], cx: number, cz: number): GlMesh | undefined {
    if (records.length === 0) return undefined;
    const { gl } = this;
    const { vertices, indices } = classicMeshData(records, cx, cz);
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const vertex = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, vertex);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    const stride = VERTEX_FLOATS * 4;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 12);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 1, gl.FLOAT, false, stride, 24);
    const index = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, index);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    return { vao, buffers: [vertex, index], count: indices.length };
  }

  private free(slot: number): void {
    const s = this.slots[slot];
    for (const m of [s?.opaque, s?.water]) {
      if (!m) continue;
      this.gl.deleteVertexArray(m.vao);
      for (const b of m.buffers) this.gl.deleteBuffer(b);
    }
    this.slots[slot] = undefined;
  }
}

/** A mob model in WebGL: its vertex array (vertices, indices, and the instance attributes' buffer), texture and poses. */
interface GlMobModel extends MobModelHandle { vao: WebGLVertexArrayObject; indexCount: number; skin: WebGLTexture; poses: WebGLTexture }

interface Program { program: WebGLProgram; uniform: (name: string) => WebGLUniformLocation | null }

/**
 * The renderer for browsers without WebGPU: the same picture as render/renderer.ts in safe
 * mode (chunk meshes built on the CPU), drawn with WebGL2. Into a multisampled framebuffer
 * (4x MSAA, with float depth) that's resolved onto the canvas. Reversed depth as with
 * WebGPU where the browser can set the depth range to 0..1 (EXT_clip_control); otherwise the
 * same projection still works, with less depth precision far away.
 */
export class GlRenderer implements GameRenderer {
  readonly api = 'WebGL2';
  readonly gpuName: string;
  /** Resolves if the browser takes the WebGL context away (a GPU crash or reset). */
  readonly lost: Promise<{ reason: string; message: string }>;
  private readonly samples: number;
  private readonly block: Program;
  private readonly far: Program;
  private readonly mob: Program;
  private readonly line: Program;
  private framebuffer?: { fb: WebGLFramebuffer; colour: WebGLRenderbuffer; depth: WebGLRenderbuffer; width: number; height: number };
  private blockTextures?: WebGLTexture;
  private readonly emptyArray: WebGLTexture;
  private farVao?: WebGLVertexArrayObject;
  private farVertex?: WebGLBuffer;
  private farIndex?: WebGLBuffer;
  private farVersion = -1;
  private farSize = 0;
  private coverage?: WebGLTexture;
  private coverageSize = 0;
  private readonly lineVao: WebGLVertexArrayObject;
  private readonly lineBuffer: WebGLBuffer;
  private readonly instanceBuffer: WebGLBuffer;
  private readonly skins = new WeakMap<ImageBitmap, WebGLTexture>();

  private constructor(readonly gl: WebGL2RenderingContext, readonly canvas: HTMLCanvasElement, msaa: boolean) {
    const debug = gl.getExtension('WEBGL_debug_renderer_info');
    this.gpuName = String(debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
    this.lost = new Promise((resolve) => canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      resolve({ reason: 'context lost', message: (e as WebGLContextEvent).statusMessage || '' });
    }));
    this.samples = msaa ? Math.min(4, gl.getParameter(gl.MAX_SAMPLES) as number) : 0;
    const clip = gl.getExtension('EXT_clip_control');
    if (clip) clip.clipControlEXT(clip.LOWER_LEFT_EXT, clip.ZERO_TO_ONE_EXT);
    log.info(`WebGL2: ${this.gpuName}; ${this.samples}x MSAA; depth range ${clip ? '0..1 (EXT_clip_control)' : '-1..1'}`);

    this.block = this.program('blocks', glBlockVertex, glBlockFragment);
    this.far = this.program('far terrain', glFarVertex, glFarFragment);
    this.mob = this.program('mobs', glMobVertex, glMobFragment);
    this.line = this.program('lines', glLineVertex, glLineFragment);

    // A blank layer until the textures load (the shader shows plain colours meanwhile).
    this.emptyArray = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.emptyArray);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA8, 1, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));

    this.lineVao = gl.createVertexArray()!;
    this.lineBuffer = gl.createBuffer()!;
    gl.bindVertexArray(this.lineVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    gl.bindVertexArray(null);
    this.instanceBuffer = gl.createBuffer()!;
  }

  /** A WebGL2 renderer on `canvas`, or an error saying why there can't be one. */
  static create(canvas: HTMLCanvasElement, options: { msaa?: boolean; offscreen?: boolean } = {}): GlRenderer {
    // (?offscreen, for screenshots: keep each frame in the canvas after it's shown.)
    const gl = canvas.getContext('webgl2', {
      // (alpha: an RGBA canvas, as the multisampled target is: resolving needs the same format. Everything drawn is opaque.)
      alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false, powerPreference: 'high-performance',
      preserveDrawingBuffer: options.offscreen === true,
    });
    if (!gl) throw new Error('This browser has neither WebGPU nor WebGL2, so the game can\'t draw anything.');
    return new GlRenderer(gl, canvas, options.msaa !== false);
  }

  private program(label: string, vertex: string, fragment: string): Program {
    const { gl } = this;
    const shader = (type: number, source: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, source);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`GLSL error in ${label} (${type === gl.VERTEX_SHADER ? 'vertex' : 'fragment'}): ${gl.getShaderInfoLog(s)}`);
      return s;
    };
    const program = gl.createProgram()!;
    gl.attachShader(program, shader(gl.VERTEX_SHADER, vertex));
    gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragment));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`GLSL link error in ${label}: ${gl.getProgramInfoLog(program)}`);
    const locations = new Map<string, WebGLUniformLocation | null>();
    const uniform = (name: string) => {
      if (!locations.has(name)) locations.set(name, gl.getUniformLocation(program, name));
      return locations.get(name)!;
    };
    return { program, uniform };
  }

  get aspect(): number {
    return this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight);
  }

  createMeshes(slots: number): GlMeshes {
    return new GlMeshes(this.gl, slots);
  }

  setBlockTextures({ count, levels }: BlockTextureData): void {
    const { gl } = this;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    levels.forEach((data, level) => {
      const s = TEXTURE_SIZE >> level;
      gl.texImage3D(gl.TEXTURE_2D_ARRAY, level, gl.RGBA8, s, s, count, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    });
    // Pixel-art up close (nearest), smoothly mipmapped further away; repeating, as faces tile by world position.
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAX_LEVEL, levels.length - 1);
    if (this.blockTextures) gl.deleteTexture(this.blockTextures);
    this.blockTextures = texture;
  }

  createMobModel(m: MobModelData): GlMobModel {
    const { gl } = this;
    let skin = this.skins.get(m.image);
    if (!skin) {
      skin = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, skin);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, m.image);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      this.skins.set(m.image, skin);
    }
    const poses = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, poses);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, m.parts * 3, m.frames, 0, gl.RGBA, gl.FLOAT, m.poseData);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, m.vertexData, gl.STATIC_DRAW);
    const stride = MOB_VERTEX_FLOATS * 4;
    [[0, 3, 0], [1, 3, 12], [2, 2, 24], [3, 1, 32]].forEach(([loc, n, offset]) => {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, n, gl.FLOAT, false, stride, offset);
    });
    // Per mob: placed and posed (pointed into the instance buffer at each draw).
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    for (const loc of [4, 5]) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.indexData, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    return { vao, indexCount: m.indices, skin, poses, clips: m.clips, fps: m.fps };
  }

  /** The multisampled colour and float depth target, the canvas's size. */
  private target(): WebGLFramebuffer {
    const { gl, canvas } = this;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    if (this.framebuffer?.width === w && this.framebuffer.height === h) return this.framebuffer.fb;
    if (this.framebuffer) {
      gl.deleteFramebuffer(this.framebuffer.fb);
      gl.deleteRenderbuffer(this.framebuffer.colour);
      gl.deleteRenderbuffer(this.framebuffer.depth);
    }
    const fb = gl.createFramebuffer()!, colour = gl.createRenderbuffer()!, depth = gl.createRenderbuffer()!;
    gl.bindRenderbuffer(gl.RENDERBUFFER, colour);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, gl.RGBA8, w, h);
    gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, this.samples, gl.DEPTH_COMPONENT32F, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, colour);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    this.framebuffer = { fb, colour, depth, width: w, height: h };
    return fb;
  }

  /** The uniforms every program shares. */
  private shared(p: Program, viewProj: Float32Array, cam: readonly number[], time: number, fogDistance: number, fogCap: number): void {
    const { gl } = this;
    gl.useProgram(p.program);
    gl.uniformMatrix4fv(p.uniform('u_viewProj'), false, viewProj);
    gl.uniform4f(p.uniform('u_camPos'), cam[0], cam[1], cam[2], time);
    gl.uniform4f(p.uniform('u_sky'), SKY[0], SKY[1], SKY[2], fogDistance);
    gl.uniform4f(p.uniform('u_fogCap'), fogCap, this.blockTextures ? 1 : 0, 0, 0);
  }

  render(
    viewProj: Float32Array, cam: readonly number[], time: number, fogDistance: number, lines: Float32Array<ArrayBuffer>,
    meshes: MeshTarget, draws: ChunkDraw[], far?: FarDraw, mobDraws: MobDraw[] = [],
  ): void {
    const { gl } = this;
    if (gl.isContextLost()) return;
    const pool = meshes as GlMeshes;
    const mobs = mobDraws as Array<{ model: GlMobModel; instances: Float32Array<ArrayBuffer> }>;
    const fogCap = far?.look === 'mist' ? MIST_FOG : 1;
    const fb = this.target();
    const { width, height } = this.framebuffer!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, width, height);
    gl.clearColor(SKY[0], SKY[1], SKY[2], 1);
    gl.clearDepth(0); // reversed depth: far is 0
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.GREATER);
    gl.disable(gl.BLEND);

    // Opaque chunks (back faces culled).
    this.shared(this.block, viewProj, cam, time, fogDistance, fogCap);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.blockTextures ?? this.emptyArray);
    gl.uniform1i(this.block.uniform('u_blocks'), 0);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
    for (const d of draws) {
      const m = pool.get(d.meshSlot)?.opaque;
      if (!m) continue;
      gl.bindVertexArray(m.vao);
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    }
    gl.disable(gl.CULL_FACE);

    if (far) this.drawFar(far, viewProj, cam, time, fogDistance, fogCap);

    // Mobs: one instanced draw per kind.
    if (mobs.length > 0) {
      this.shared(this.mob, viewProj, cam, time, fogDistance, fogCap);
      gl.uniform1i(this.mob.uniform('u_skin'), 0);
      gl.uniform1i(this.mob.uniform('u_poses'), 1);
      const all = new Float32Array(mobs.reduce((n, m) => n + m.instances.length, 0));
      let at = 0;
      for (const m of mobs) { all.set(m.instances, at); at += m.instances.length; }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, all, gl.STREAM_DRAW);
      let first = 0;
      for (const m of mobs) {
        const count = m.instances.length / 8;
        gl.bindVertexArray(m.model.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
        gl.vertexAttribPointer(4, 4, gl.FLOAT, false, 32, first * 32);
        gl.vertexAttribPointer(5, 4, gl.FLOAT, false, 32, first * 32 + 16);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, m.model.skin);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, m.model.poses);
        gl.drawElementsInstanced(gl.TRIANGLES, m.model.indexCount, gl.UNSIGNED_SHORT, 0, count);
        first += count;
      }
      gl.activeTexture(gl.TEXTURE0);
    }

    if (lines.length > 0) {
      this.shared(this.line, viewProj, cam, time, fogDistance, fogCap);
      gl.bindVertexArray(this.lineVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, lines, gl.STREAM_DRAW);
      gl.depthMask(false);
      gl.depthFunc(gl.GEQUAL);
      gl.drawArrays(gl.LINES, 0, lines.length / 6);
      gl.depthFunc(gl.GREATER);
    }

    // Translucent water, far chunks first, without writing depth.
    this.shared(this.block, viewProj, cam, time, fogDistance, fogCap);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.blockTextures ?? this.emptyArray);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    const d2 = (p: number[]) => (p[0] - cam[0]) ** 2 + (p[1] - cam[1]) ** 2 + (p[2] - cam[2]) ** 2;
    for (const d of [...draws].sort((a, b) => d2(b.center) - d2(a.center))) {
      const m = pool.get(d.meshSlot)?.water;
      if (!m) continue;
      gl.bindVertexArray(m.vao);
      gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_INT, 0);
    }
    gl.disable(gl.BLEND);
    gl.depthMask(true);
    gl.bindVertexArray(null);

    // Resolve onto the canvas.
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, fb);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private drawFar(far: FarDraw, viewProj: Float32Array, cam: readonly number[], time: number, fogDistance: number, fogCap: number): void {
    const { gl } = this;
    if (!this.farVao || this.farSize !== far.vertices.length) {
      if (this.farVao) gl.deleteVertexArray(this.farVao);
      for (const b of [this.farVertex, this.farIndex]) if (b) gl.deleteBuffer(b);
      this.farVao = gl.createVertexArray()!;
      gl.bindVertexArray(this.farVao);
      this.farVertex = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.farVertex);
      gl.bufferData(gl.ARRAY_BUFFER, far.vertices.byteLength, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
      this.farIndex = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.farIndex);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, far.indices, gl.STATIC_DRAW);
      this.farSize = far.vertices.length;
      this.farVersion = -1;
    }
    gl.bindVertexArray(this.farVao);
    if (far.version !== this.farVersion) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.farVertex!);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, far.vertices);
      this.farVersion = far.version;
    }
    // The coverage map: one byte a chunk.
    const { x0, z0, size, data } = far.coverage;
    gl.activeTexture(gl.TEXTURE2);
    if (!this.coverage || this.coverageSize !== size) {
      if (this.coverage) gl.deleteTexture(this.coverage);
      this.coverage = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, this.coverage);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R8, size, size);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      this.coverageSize = size;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.coverage);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, size, size, gl.RED, gl.UNSIGNED_BYTE, data);
    this.shared(this.far, viewProj, cam, time, fogDistance, fogCap);
    gl.uniform1i(this.far.uniform('u_coverage'), 2);
    gl.uniform4f(this.far.uniform('u_near'), x0 * 16, z0 * 16, size, 0);
    gl.uniform4f(this.far.uniform('u_sea'), far.seaY, ['colour', 'silhouette', 'mist'].indexOf(far.look), far.extent, 0);
    gl.drawElements(gl.TRIANGLES, far.indices.length, gl.UNSIGNED_INT, 0);
    gl.activeTexture(gl.TEXTURE0);
  }
}
