import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, SOURCE_LEVEL } from '../constants';
import { AROUND, type MeshJob } from '../sim/store';
import { FACE_CAPACITY, FULL_HEIGHT, Face } from './mesher';
import type { MeshPool } from './meshPool';

/** u32s per mesh job: the AROUND slots around the chunk, then its mesh slot. */
const JOB_STRIDE = AROUND + 1;
const WORKGROUP = 64;

// The mesher of render/mesher.ts as a compute shader: one invocation per cell finds the
// cell's visible faces and reserves room for them in the chunk's mesh slot with one
// atomic add on the slot's indirect draw (the draw's vertex count is the slot's face
// count). Faces land in no particular order; the GPU tests compare them with the CPU
// mesher's as sets. (No workgroup memory or barriers: measured faster without.)
const MESH_WGSL = /* wgsl */ `
const AIR: i32 = ${Block.Air};
const STONE: i32 = ${Block.Stone};
const DIRT: i32 = ${Block.Dirt};
const WATER: i32 = ${Block.Water};
const LAVA: i32 = ${Block.Lava};
const GRASS: i32 = ${Block.Grass};
const WHEAT: i32 = ${Block.Wheat};
const S: i32 = ${CHUNK_SIZE};
const H: i32 = ${CHUNK_HEIGHT};
const VOLUME: i32 = ${CHUNK_VOLUME};
const CAP: u32 = ${FACE_CAPACITY}u;
const FULL_HEIGHT: i32 = ${FULL_HEIGHT};
const SOURCE: i32 = ${SOURCE_LEVEL};
const PLANT_FIRST: i32 = ${Face.PlantA};
const PLANT_LAST: i32 = ${Face.PlantBBack};

@group(0) @binding(0) var<storage, read> cells: array<i32>;
// Per chunk: its ${AROUND} slots around (3x3, row by row), then its mesh slot.
@group(0) @binding(1) var<storage, read> jobs: array<u32>;
@group(0) @binding(2) var<storage, read_write> faces: array<u32>;
// Per mesh slot: drawIndirect arguments for its opaque faces, then its water faces.
@group(0) @binding(3) var<storage, read_write> draws: array<atomic<u32>>;

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
fn isSolid(t: i32) -> bool { return t == STONE || t == DIRT || t == GRASS; }
fn heightCode(c: i32, above: i32) -> i32 {
  if (typeOf(above) == typeOf(c)) { return FULL_HEIGHT; }
  return clamp(c >> 3u, 1, SOURCE);
}
fn faceRecord(x: i32, y: i32, z: i32, face: i32, t: i32, aux: i32) -> u32 {
  return u32(x | (z << 4u) | (y << 8u) | (face << 14u) | (t << 18u) | (aux << 21u));
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  job = wg.y * ${JOB_STRIDE}u;
  let slot = jobs[job + ${AROUND}u];
  let i = i32(wg.x * ${WORKGROUP}u + li);
  let x = i & 15;
  let z = (i >> 4u) & 15;
  let y = i >> 8u;

  // This cell's faces.
  var mine: array<u32, 6>;
  var n = 0u;
  let c = cellAt(y, z, x);
  let t = typeOf(c);
  let water = t == WATER;
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

  if (n == 0u) { return; }
  // Make room: opaque faces fill the slot from its start, water from its end.
  let base = slot * CAP;
  if (water) {
    let start = atomicAdd(&draws[slot * 8u + 4u], n * 6u) / 6u;
    for (var k = 0u; k < n; k++) { faces[base + CAP - 1u - (start + k)] = mine[k]; }
  } else {
    let start = atomicAdd(&draws[slot * 8u], n * 6u) / 6u;
    for (var k = 0u; k < n; k++) { faces[base + start + k] = mine[k]; }
  }
}
`;

/** Meshes chunks on the GPU, from the cells in a GpuStore into a MeshPool. */
export class GpuMesher {
  private jobs?: GPUBuffer;
  private group?: GPUBindGroup;
  private groupFor?: { cells: GPUBuffer; faces: GPUBuffer; jobs: GPUBuffer };

  private constructor(private readonly device: GPUDevice, private readonly pipeline: GPUComputePipeline) {}

  static async create(device: GPUDevice): Promise<GpuMesher> {
    const module = device.createShaderModule({ code: MESH_WGSL });
    return new GpuMesher(device, await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } }));
  }

  mesh(cells: GPUBuffer, jobs: MeshJob[], pool: MeshPool): void {
    if (jobs.length === 0) return;
    const data = new Uint32Array(jobs.length * JOB_STRIDE);
    jobs.forEach((job, k) => {
      data.set(job.around, k * JOB_STRIDE);
      data[k * JOB_STRIDE + AROUND] = job.meshSlot;
      pool.reset(job.meshSlot, job.cx, job.cz);
    });
    const { device } = this;
    if (!this.jobs || this.jobs.size < data.byteLength) {
      this.jobs?.destroy();
      this.jobs = device.createBuffer({ size: Math.max(data.byteLength, 4096), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    device.queue.writeBuffer(this.jobs, 0, data);
    const g = this.groupFor;
    if (!this.group || g?.cells !== cells || g.faces !== pool.faces || g.jobs !== this.jobs) {
      this.groupFor = { cells, faces: pool.faces, jobs: this.jobs };
      this.group = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [cells, this.jobs, pool.faces, pool.draws].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(CHUNK_VOLUME / WORKGROUP, jobs.length);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
}
