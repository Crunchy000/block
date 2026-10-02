import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, SOURCE_LEVEL } from '../constants';
import { AROUND, type MeshJob } from '../sim/store';
import { FULL_HEIGHT, Face } from './mesher';
import type { ChunkMesh, MeshPool } from './meshPool';

/** u32s per mesh job: the AROUND slots around the chunk, the starts of its opaque and water runs, and their sizes. */
const JOB_STRIDE = AROUND + 4;
const WORKGROUP = 64;

// The mesher of render/mesher.ts as a compute shader, in two passes over the same per-cell
// face finding. countFaces adds up each chunk's opaque and water faces; the CPU reads those
// counts back (8 bytes a chunk) and reserves exactly that much room in the mesh pool;
// writeFaces then puts every face in its chunk's run, its place in the run taken with an
// atomic add. Faces land in no particular order; the GPU tests compare them with the CPU
// mesher's as sets. (No workgroup memory or barriers: measured faster without.)
const MESH_WGSL = /* wgsl */ `
const AIR: i32 = ${Block.Air};
const STONE: i32 = ${Block.Stone};
const DIRT: i32 = ${Block.Dirt};
const WATER: i32 = ${Block.Water};
const LAVA: i32 = ${Block.Lava};
const GRASS: i32 = ${Block.Grass};
const WHEAT: i32 = ${Block.Wheat};
const DIAMOND: i32 = ${Block.Diamond};
const S: i32 = ${CHUNK_SIZE};
const H: i32 = ${CHUNK_HEIGHT};
const VOLUME: i32 = ${CHUNK_VOLUME};
const FULL_HEIGHT: i32 = ${FULL_HEIGHT};
const SOURCE: i32 = ${SOURCE_LEVEL};
const PLANT_FIRST: i32 = ${Face.PlantA};
const PLANT_LAST: i32 = ${Face.PlantBBack};

@group(0) @binding(0) var<storage, read> cells: array<i32>;
// Per chunk: its ${AROUND} slots around (3x3, row by row), the starts of its opaque and water runs, their sizes.
@group(0) @binding(1) var<storage, read> jobs: array<u32>;
@group(0) @binding(2) var<storage, read_write> faces: array<u32>;
// Per chunk: opaque and water face counts (countFaces), or how much of each run is filled (writeFaces).
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;

// Where this chunk's job starts in jobs.
var<private> job: u32;

// Chunk-local cell; x and z may be a block into the neighbouring chunks.
fn cellAt(y: i32, z: i32, x: i32) -> i32 {
  if (y >= H) { return AIR; }
  if (y < 0) { return STONE; }
  let gx = select(select(1, 2, x >= S), 0, x < 0);
  let gz = select(select(1, 2, z >= S), 0, z < 0);
  return cells[i32(jobs[job + u32(gz * 3 + gx)]) * VOLUME + (y * S + z - (gz - 1) * S) * S + x - (gx - 1) * S];
}
fn typeOf(c: i32) -> i32 { return c & 7; }
fn isSolid(t: i32) -> bool { return t == STONE || t == DIRT || t == GRASS || t == DIAMOND; }
fn heightCode(c: i32, above: i32) -> i32 {
  if (typeOf(above) == typeOf(c)) { return FULL_HEIGHT; }
  return clamp(c >> 3u, 1, SOURCE);
}
fn faceRecord(x: i32, y: i32, z: i32, face: i32, t: i32, aux: i32) -> u32 {
  return u32(x | (z << 4u) | (y << 8u) | (face << 14u) | (t << 18u) | (aux << 21u));
}

var<private> mine: array<u32, 6>;
var<private> n: u32;
var<private> water: bool;

// The faces of one cell of chunk wg.y, into mine[0..n].
fn cellFaces(wg: vec3u, li: u32) {
  job = wg.y * ${JOB_STRIDE}u;
  let i = i32(wg.x * ${WORKGROUP}u + li);
  let x = i & 15;
  let z = (i >> 4u) & 15;
  let y = i >> 8u;

  n = 0u;
  let c = cellAt(y, z, x);
  let t = typeOf(c);
  water = t == WATER;
  if (t == WHEAT) {
    let stage = min(c >> 3u, 15);
    for (var f = PLANT_FIRST; f <= PLANT_LAST; f++) {
      mine[n] = faceRecord(x, y, z, f, t, stage);
      n++;
    }
  } else if (t != AIR) {
    let fluid = water || t == LAVA;
    let h = select(0, heightCode(c, cellAt(y + 1, z, x)), fluid);
    for (var f = 0; f < 6; f++) {
      // The normal of side f: +x, -x, +y, -y, +z, -z.
      let s = 1 - 2 * (f & 1);
      let nx = select(0, s, f / 2 == 0);
      let ny = select(0, s, f / 2 == 1);
      let nz = select(0, s, f / 2 == 2);
      let nc = cellAt(y + ny, z + nz, x + nx);
      let nt = typeOf(nc);
      // Only full solid blocks hide a neighbour's face. Fluid shows against air or the other
      // fluid, and against the same fluid only where the neighbour's surface is lower.
      var visible = !isSolid(nt);
      if (fluid) {
        visible = (nt != t && !isSolid(nt)) || (nt == t && ny == 0 && heightCode(nc, cellAt(y + 1, z + nz, x + nx)) < h);
      }
      if (visible) {
        mine[n] = faceRecord(x, y, z, f, t, h);
        n++;
      }
    }
  }

}

@compute @workgroup_size(${WORKGROUP})
fn countFaces(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  cellFaces(wg, li);
  if (n > 0u) { atomicAdd(&counts[wg.y * 2u + select(0u, 1u, water)], n); }
}

@compute @workgroup_size(${WORKGROUP})
fn writeFaces(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  cellFaces(wg, li);
  if (n == 0u) { return; }
  let w = select(0u, 1u, water);
  // A block update between the two passes can add faces: never write past the run (the
  // chunk changed, so it's meshed again anyway).
  let at = atomicAdd(&counts[wg.y * 2u + w], n);
  let size = jobs[job + ${AROUND}u + 2u + w];
  let start = jobs[job + ${AROUND}u + w];
  for (var k = 0u; k < n; k++) {
    if (at + k < size) { faces[start + at + k] = mine[k]; }
  }
}
`;

/** Meshes chunks on the GPU, from the cells in a GpuStore into a MeshPool. */
export class GpuMesher {
  private jobs?: GPUBuffer;
  private counts?: GPUBuffer;
  private group?: GPUBindGroup;
  private groupFor?: { cells: GPUBuffer; faces: GPUBuffer; jobs: GPUBuffer };

  private constructor(
    private readonly device: GPUDevice, private readonly layout: GPUBindGroupLayout,
    private readonly count: GPUComputePipeline, private readonly write: GPUComputePipeline,
  ) {}

  static async create(device: GPUDevice): Promise<GpuMesher> {
    const module = device.createShaderModule({ label: 'meshing', code: MESH_WGSL });
    const storage = (type: GPUBufferBindingType) => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = device.createBindGroupLayout({
      label: 'meshing',
      entries: [storage('read-only-storage'), storage('read-only-storage'), storage('storage'), storage('storage')]
        .map((e, binding) => ({ binding, ...e })),
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipeline = (entryPoint: string) => device.createComputePipelineAsync({
      label: `meshing (${entryPoint})`, layout: pipelineLayout, compute: { module, entryPoint },
    });
    const [count, write] = await Promise.all([pipeline('countFaces'), pipeline('writeFaces')]);
    return new GpuMesher(device, layout, count, write);
  }

  /**
   * Mesh chunks into the pool: count their faces, read the counts back, reserve room, write
   * the faces, then make them the chunks' meshes. Jobs whose `current` says no once the
   * counts are back (the chunk left meanwhile) are dropped.
   */
  async mesh(cells: GPUBuffer, jobs: MeshJob[], pool: MeshPool): Promise<void> {
    if (jobs.length === 0) return;
    const { device } = this;
    const data = new Uint32Array(jobs.length * JOB_STRIDE);
    jobs.forEach((job, k) => data.set(job.around, k * JOB_STRIDE));
    if (!this.jobs || this.jobs.size < data.byteLength) {
      this.jobs?.destroy();
      this.counts?.destroy();
      const n = Math.max(jobs.length, 64);
      this.jobs = device.createBuffer({ label: 'mesh jobs', size: n * JOB_STRIDE * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.counts = device.createBuffer({ label: 'mesh counts', size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    }
    const run = (pipeline: GPUComputePipeline, readCounts: boolean) => {
      device.queue.writeBuffer(this.jobs!, 0, data);
      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(this.counts!, 0, jobs.length * 8);
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.bindGroup(cells, pool));
      pass.dispatchWorkgroups(CHUNK_VOLUME / WORKGROUP, jobs.length);
      pass.end();
      let readback: GPUBuffer | undefined;
      if (readCounts) {
        readback = device.createBuffer({ size: jobs.length * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        encoder.copyBufferToBuffer(this.counts!, 0, readback, 0, jobs.length * 8);
      }
      device.queue.submit([encoder.finish()]);
      return readback;
    };

    const readback = run(this.count, true)!;
    let counts: Uint32Array;
    try {
      await readback.mapAsync(GPUMapMode.READ);
      counts = new Uint32Array(readback.getMappedRange().slice(0));
    } finally {
      readback.destroy();
    }
    const meshes: Array<ChunkMesh | undefined> = jobs.map((job, k) => {
      if (job.current && !job.current()) return undefined;
      const mesh = pool.reserve(counts[k * 2], counts[k * 2 + 1]);
      if (!mesh) throw new Error(`no room on the GPU for more chunk meshes (${pool.usage.used} faces in use)`);
      data.set([mesh.opaque.start, mesh.water.start, mesh.opaque.count, mesh.water.count], k * JOB_STRIDE + AROUND);
      return mesh;
    });
    // Runs start zeroed (a zero record draws nothing), in case fewer faces arrive than were counted.
    const clear = device.createCommandEncoder();
    for (const m of meshes) {
      for (const r of m ? [m.opaque, m.water] : []) if (r.count) clear.clearBuffer(pool.faces, r.start * 4, r.count * 4);
    }
    device.queue.submit([clear.finish()]);
    // Reserving may have grown (replaced) the face buffer: bind the current one.
    run(this.write, false);
    jobs.forEach((job, k) => { if (meshes[k]) pool.commit(job.meshSlot, job.cx, job.cz, meshes[k]!); });
  }

  private bindGroup(cells: GPUBuffer, pool: MeshPool): GPUBindGroup {
    const g = this.groupFor;
    if (!this.group || g?.cells !== cells || g.faces !== pool.faces || g.jobs !== this.jobs) {
      this.groupFor = { cells, faces: pool.faces, jobs: this.jobs! };
      this.group = this.device.createBindGroup({
        layout: this.layout,
        entries: [cells, this.jobs!, pool.faces, this.counts!].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }
    return this.group;
  }
}
