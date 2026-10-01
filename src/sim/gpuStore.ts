import * as tf from '@tensorflow/tfjs';
import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, type PlantRates } from '../constants';
import type { RayHit } from '../player/raycast';
import { GpuMesher } from '../render/gpuMesher';
import { MeshPool } from '../render/meshPool';
import { GpuWorldgen, type GenJob } from './gpuWorldgen';
import { RULES_WGSL } from './rules';
import { AROUND, SELF, TickFlag, type CellStore, type MeshJob, type MeshTarget, type StagedChunks } from './store';

const CHUNK_BYTES = CHUNK_VOLUME * 4;
const WORKGROUP = 64;

// One block-update tick for a list of chunks, each read where it lives in the world
// buffer, its neighbours' cells read straight from their slots. Results go to a scratch
// buffer (every chunk steps from the state before the tick) and are copied back after.
// Each cell ORs its flags into its chunk's flags word, skipping the atomic when the
// bits are already set. (No workgroup memory or barriers: measured faster without.)
const SIM_WGSL = /* wgsl */ `
${RULES_WGSL}

struct Uniforms {
  seed: u32,
  grassSpread: f32,
  wheatGrow: f32,
  wheatGrowWet: f32,
  plants: i32,
};
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> cells: array<i32>;
@group(0) @binding(2) var<storage, read_write> stepped: array<i32>;
// Per chunk: the slots of the 3x3 chunks around it, row by row.
@group(0) @binding(3) var<storage, read> jobs: array<u32>;
@group(0) @binding(4) var<storage, read_write> flags: array<atomic<u32>>;

const S: i32 = ${CHUNK_SIZE};
const H: i32 = ${CHUNK_HEIGHT};
const VOLUME: i32 = ${CHUNK_VOLUME};

// Where this chunk's job starts in jobs.
var<private> job: u32;

// Chunk-local cell; x and z may be a block into the neighbouring chunks.
fn cellAt(y: i32, z: i32, x: i32) -> i32 {
  if (y >= H) { return AIR; }
  if (y < 0) { return STONE; }
  let gx = select(select(1, 2, x >= S), 0, x < 0);
  let gz = select(select(1, 2, z >= S), 0, z < 0);
  let slot = i32(jobs[job + u32(gz * 3 + gx)]);
  return cells[slot * VOLUME + (y * S + z - (gz - 1) * S) * S + x - (gx - 1) * S];
}
fn unprimed(c: i32) -> i32 { return select(c, DIRT, c == PRIMED); }
fn isGrowing(c: i32) -> bool { return c == PRIMED || (blockOf(c) == WHEAT && levelOf(c) < RIPE); }
// The Border bits (world.ts) of a column: its chunk sides and corners.
fn borderBits(x: i32, z: i32) -> u32 {
  let w = x == 0;
  let e = x == S - 1;
  let n = z == 0;
  let s = z == S - 1;
  return select(0u, 1u, w) | select(0u, 2u, e) | select(0u, 4u, n) | select(0u, 8u, s)
    | select(0u, 16u, n && w) | select(0u, 32u, n && e) | select(0u, 64u, s && w) | select(0u, 128u, s && e);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  job = wg.y * ${AROUND}u;
  let i = i32(wg.x * ${WORKGROUP}u + li);
  let x = i & 15;
  let z = (i >> 4u) & 15;
  let y = i >> 8u;
  let home = i32(jobs[job + ${SELF}u]) * VOLUME;
  let old = cells[home + i];
  // The cell's random number: cellRandom(its index in the world buffer, seed).
  let next = nextCell(y, z, x, u32(home + i));
  stepped[i32(wg.y) * VOLUME + i] = next;

  var f = 0u;
  if (unprimed(next) != unprimed(old)) { f = ${TickFlag.Changed}u | borderBits(x, z); }
  if (isGrowing(next)) { f = f | ${TickFlag.Growing}u; }
  if (f != 0u && (atomicLoad(&flags[wg.y]) & f) != f) { atomicOr(&flags[wg.y], f); }
}
`;

// Picking: the voxel walk of player/raycast.ts, through the world buffer. Chunks that
// aren't in their slot (not loaded, or the slot holds another chunk) read as air.
const RAY_WGSL = /* wgsl */ `
const AIR: i32 = ${Block.Air};
const STONE: i32 = ${Block.Stone};
const DIRT: i32 = ${Block.Dirt};
const GRASS: i32 = ${Block.Grass};
const WHEAT: i32 = ${Block.Wheat};
const S: i32 = ${CHUNK_SIZE};
const H: i32 = ${CHUNK_HEIGHT};
const VOLUME: i32 = ${CHUNK_VOLUME};
const FAR: f32 = 1e30;

struct Ray {
  origin: vec3f,
  maxDist: f32,
  dir: vec3f,
  ring: i32,
};
@group(0) @binding(0) var<uniform> ray: Ray;
@group(0) @binding(1) var<storage, read> cells: array<i32>;
// Per slot: the chunk it holds (x, z) and whether its cells are loaded.
@group(0) @binding(2) var<storage, read> slots: array<vec4<i32>>;
// Hit (1 or 0), the block hit, the cell before it.
@group(0) @binding(3) var<storage, read_write> result: array<i32, 8>;

fn ringIndex(c: i32) -> i32 { return ((c % ray.ring) + ray.ring) % ray.ring; }

fn blockAt(p: vec3<i32>) -> i32 {
  if (p.y < 0) { return STONE; }
  if (p.y >= H) { return AIR; }
  let cx = p.x >> 4u;
  let cz = p.z >> 4u;
  let slot = ringIndex(cz) * ray.ring + ringIndex(cx);
  let info = slots[slot];
  if (info.x != cx || info.y != cz || info.z == 0) { return AIR; }
  return cells[slot * VOLUME + (p.y * S + (p.z & 15)) * S + (p.x & 15)] & 7;
}

@compute @workgroup_size(1)
fn main() {
  let o = ray.origin;
  let d = ray.dir;
  var pos = vec3<i32>(floor(o));
  let dirStep = vec3<i32>(sign(d));
  var tDelta = vec3f(FAR);
  var tMax = vec3f(FAR);
  for (var a = 0; a < 3; a++) {
    if (d[a] != 0.0) {
      tDelta[a] = abs(1.0 / d[a]);
      tMax[a] = (select(f32(pos[a]), f32(pos[a] + 1), d[a] > 0.0) - o[a]) / d[a];
    }
  }
  var prev = pos;
  var t = 0.0;
  result[0] = 0;
  for (var k = 0; k < 4096 && t <= ray.maxDist; k++) {
    let b = blockAt(pos);
    if (b == STONE || b == DIRT || b == GRASS || b == WHEAT) {
      result[0] = 1;
      result[1] = pos.x; result[2] = pos.y; result[3] = pos.z;
      result[4] = prev.x; result[5] = prev.y; result[6] = prev.z;
      return;
    }
    prev = pos;
    var axis = 2;
    if (tMax.x < tMax.y) {
      if (tMax.x < tMax.z) { axis = 0; }
    } else if (tMax.y < tMax.z) {
      axis = 1;
    }
    pos[axis] += dirStep[axis];
    t = tMax[axis];
    tMax[axis] += tDelta[axis];
  }
}
`;

/** A tensor's data in a GPU buffer (a copy TF.js makes on the GPU), if it's on the GPU. */
function gpuData(t: tf.Tensor): { buffer: GPUBuffer; tensorRef: tf.Tensor } | undefined {
  try {
    const { buffer, tensorRef } = t.dataToGPU();
    if (buffer) return { buffer, tensorRef };
    tensorRef.dispose();
  } catch {
    // e.g. TF.js computed it on the CPU: use the CPU path
  }
  return undefined;
}

/** Mappable buffers for reading results back, reused once they're unmapped. */
class Readbacks {
  private free: GPUBuffer[] = [];
  constructor(private readonly device: GPUDevice) {}

  /** A buffer of at least `size` bytes to copy into, and a reader for after the copy is submitted. */
  take(size: number): { buffer: GPUBuffer; read: () => Promise<ArrayBuffer> } {
    const i = this.free.findIndex((b) => b.size >= size);
    const buffer = i >= 0 ? this.free.splice(i, 1)[0]
      : this.device.createBuffer({ size: Math.max(256, Math.ceil(size / 256) * 256), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const read = async () => {
      try {
        await buffer.mapAsync(GPUMapMode.READ, 0, size);
        const out = buffer.getMappedRange(0, size).slice(0);
        buffer.unmap();
        this.free.push(buffer);
        return out;
      } catch (e) {
        buffer.destroy();
        throw e;
      }
    };
    return { buffer, read };
  }
}

/**
 * The world's cells in GPU memory: one storage buffer holding every ring slot. Block
 * updates, meshing and picking all run there; per tick the CPU gets back one flags word
 * per simulated chunk, and otherwise only an edited chunk's cells when it leaves the
 * halo (to be restored when it comes back).
 */
export class GpuStore implements CellStore {
  /** Chunks per meshing round (two GPU passes with a read back of 8 bytes a chunk between). */
  readonly meshBudget = 32;
  /** Every slot's cells (i32), CHUNK_VOLUME per slot in chunk layout. */
  readonly cells: GPUBuffer;
  private readonly slotInfo: GPUBuffer;
  private readonly uniforms: GPUBuffer;
  private readonly ray: GPUBuffer;
  private readonly rayResult: GPUBuffer;
  private readonly rayGroup: GPUBindGroup;
  private readonly readbacks: Readbacks;
  /** Per-tick buffers, grown to the biggest batch so far. */
  private capacity = 0;
  private stepped?: GPUBuffer;
  private jobs?: GPUBuffer;
  private flags?: GPUBuffer;
  private simGroup?: GPUBindGroup;

  private constructor(
    readonly device: GPUDevice, readonly ring: number,
    private readonly simPipeline: GPUComputePipeline, private readonly rayPipeline: GPUComputePipeline,
    private readonly mesher: GpuMesher, worldgen: GPUComputePipeline,
  ) {
    const slots = ring * ring;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.cells = device.createBuffer({ size: slots * CHUNK_BYTES, usage: storage });
    this.slotInfo = device.createBuffer({ size: slots * 16, usage: storage });
    this.uniforms = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.ray = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.rayResult = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.rayGroup = device.createBindGroup({
      layout: rayPipeline.getBindGroupLayout(0),
      entries: [this.ray, this.cells, this.slotInfo, this.rayResult].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    this.readbacks = new Readbacks(device);
    this.generator = new GpuWorldgen(device, this.cells, worldgen);
  }

  private readonly generator: GpuWorldgen;

  generate(chunks: GenJob[], seed?: number): void {
    this.generator.generate(chunks, seed);
  }

  /** Compiles its shaders without blocking the page. */
  static async create(device: GPUDevice, ring: number): Promise<GpuStore> {
    const pipeline = (label: string, code: string) => device.createComputePipelineAsync({
      label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
    });
    const [sim, ray, mesher, worldgen] = await Promise.all([
      pipeline('block updates', SIM_WGSL), pipeline('picking', RAY_WGSL), GpuMesher.create(device), GpuWorldgen.compile(device),
    ]);
    return new GpuStore(device, ring, sim, ray, mesher, worldgen);
  }

  /** Bytes of GPU memory the world's cells take. */
  get bytes(): number {
    return this.cells.size;
  }

  setSlot(slot: number, cx: number, cz: number, loaded: boolean): void {
    this.device.queue.writeBuffer(this.slotInfo, slot * 16, new Int32Array([cx, cz, loaded ? 1 : 0, 0]));
  }

  writeChunk(slot: number, cells: ArrayLike<number>): void {
    const data = cells instanceof Int32Array ? cells : Int32Array.from(cells);
    this.device.queue.writeBuffer(this.cells, slot * CHUNK_BYTES, data as Int32Array<ArrayBuffer>);
  }

  writeCell(slot: number, index: number, value: number): void {
    this.device.queue.writeBuffer(this.cells, (slot * CHUNK_VOLUME + index) * 4, new Int32Array([value]));
  }

  readChunk(slot: number): Promise<Int32Array> {
    const { buffer, read } = this.readbacks.take(CHUNK_BYTES);
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.cells, slot * CHUNK_BYTES, buffer, 0, CHUNK_BYTES);
    this.device.queue.submit([encoder.finish()]); // now, before anything else can write the slot
    return read().then((b) => new Int32Array(b));
  }

  async stage(cells: tf.Tensor): Promise<StagedChunks> {
    // TF.js on this same GPU device: copy its output buffer into the slots, GPU to GPU.
    const backend = tf.getBackend() === 'webgpu' ? (tf.backend() as WebGPUBackend) : undefined;
    const gpu = backend?.device === this.device ? gpuData(cells) : undefined;
    if (gpu) {
      const { buffer, tensorRef } = gpu;
      const encoder = this.device.createCommandEncoder();
      return {
        write: (slot, index) => encoder.copyBufferToBuffer(buffer!, index * CHUNK_BYTES, this.cells, slot * CHUNK_BYTES, CHUNK_BYTES),
        release: () => {
          this.device.queue.submit([encoder.finish()]);
          tensorRef.dispose();
        },
      };
    }
    // Otherwise (TF.js fell back to WebGL or the CPU) through the CPU.
    const data = (await cells.data()) as Int32Array;
    return {
      write: (slot, index) => this.writeChunk(slot, data.subarray(index * CHUNK_VOLUME, (index + 1) * CHUNK_VOLUME)),
      release: () => {},
    };
  }

  tick(jobs: Uint32Array, seed: number, rates: PlantRates): Promise<Uint32Array> {
    return this.step(jobs, seed, rates, true)!;
  }

  /**
   * Issue a tick. With `readFlags` false nothing is read back and nothing is returned
   * (benchmarks chain ticks this way); `plants` false skips the grass and wheat rules.
   */
  step(jobs: Uint32Array, seed: number, rates: PlantRates, readFlags: boolean, plants = true): Promise<Uint32Array> | undefined {
    const n = jobs.length / AROUND;
    if (n === 0) return readFlags ? Promise.resolve(new Uint32Array(0)) : undefined;
    this.reserve(n);
    const { device } = this;
    device.queue.writeBuffer(this.jobs!, 0, jobs as Uint32Array<ArrayBuffer>);
    const u = new ArrayBuffer(32);
    new Uint32Array(u, 0, 1)[0] = seed >>> 0;
    new Float32Array(u, 4, 3).set([rates.grassSpread, rates.wheatGrow, rates.wheatGrowWet]);
    new Int32Array(u, 16, 1)[0] = plants ? 1 : 0; // the game leaves them on: they cost little here
    device.queue.writeBuffer(this.uniforms, 0, u);

    const encoder = device.createCommandEncoder();
    encoder.clearBuffer(this.flags!, 0, n * 4);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.simPipeline);
    pass.setBindGroup(0, this.simGroup!);
    pass.dispatchWorkgroups(CHUNK_VOLUME / WORKGROUP, n);
    pass.end();
    for (let k = 0; k < n; k++) {
      encoder.copyBufferToBuffer(this.stepped!, k * CHUNK_BYTES, this.cells, jobs[k * AROUND + SELF] * CHUNK_BYTES, CHUNK_BYTES);
    }
    let readback: ReturnType<Readbacks['take']> | undefined;
    if (readFlags) {
      readback = this.readbacks.take(n * 4);
      encoder.copyBufferToBuffer(this.flags!, 0, readback.buffer, 0, n * 4);
    }
    device.queue.submit([encoder.finish()]);
    return readback?.read().then((b) => new Uint32Array(b));
  }

  private reserve(n: number): void {
    if (n <= this.capacity) return;
    const { device } = this;
    this.capacity = Math.max(n, this.capacity * 2, 16);
    for (const b of [this.stepped, this.jobs, this.flags]) b?.destroy();
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.stepped = device.createBuffer({ size: this.capacity * CHUNK_BYTES, usage: storage });
    this.jobs = device.createBuffer({ size: this.capacity * AROUND * 4, usage: storage });
    this.flags = device.createBuffer({ size: this.capacity * 4, usage: storage });
    this.simGroup = device.createBindGroup({
      layout: this.simPipeline.getBindGroupLayout(0),
      entries: [this.uniforms, this.cells, this.stepped, this.jobs, this.flags].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
  }

  async raycast(origin: readonly number[], dir: readonly number[], maxDist: number): Promise<RayHit | null> {
    const u = new ArrayBuffer(32);
    new Float32Array(u, 0, 7).set([origin[0], origin[1], origin[2], maxDist, dir[0], dir[1], dir[2]]);
    new Int32Array(u, 28, 1)[0] = this.ring;
    this.device.queue.writeBuffer(this.ray, 0, u);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.rayPipeline);
    pass.setBindGroup(0, this.rayGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    const { buffer, read } = this.readbacks.take(32);
    encoder.copyBufferToBuffer(this.rayResult, 0, buffer, 0, 32);
    this.device.queue.submit([encoder.finish()]);
    const r = new Int32Array(await read());
    return r[0] ? { block: [r[1], r[2], r[3]], before: [r[4], r[5], r[6]] } : null;
  }

  mesh(jobs: MeshJob[], target: MeshTarget | undefined): Promise<void> {
    if (!target) return Promise.resolve();
    if (!(target instanceof MeshPool)) throw new Error('the GPU world meshes into a MeshPool');
    return this.mesher.mesh(this.cells, jobs, target);
  }

  destroy(): void {
    for (const b of [this.cells, this.slotInfo, this.uniforms, this.ray, this.rayResult, this.stepped, this.jobs, this.flags]) b?.destroy();
    this.generator.destroy();
  }
}
