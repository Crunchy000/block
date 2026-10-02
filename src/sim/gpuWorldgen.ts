import { Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, SEA_LEVEL, SOURCE_LEVEL, WHEAT_RIPE, cell } from '../constants';
import { DEFAULT_SEED, DIAMOND_MAX_Y, DIAMOND_THRESHOLD, WHEAT_PATCH_CHANCE } from '../tf/worldgen';

/**
 * World generation as one compute shader, writing chunks straight into the world's slots.
 *
 * The same layers as tf/worldgen.ts (which stays the reference, and the generator when the
 * world lives on the CPU), op for op in the same order, so the two agree but for the odd
 * cell where the GPU rounds differently. As TF.js ops a batch is ~800 dispatches issued
 * from the main thread (tens of ms of it a batch, a dropped frame or two while you walk);
 * here it is one dispatch: one thread per column works out the ground's height and
 * plants once, then fills the column's cells, with cave noise only where caves can be.
 */
const WORLDGEN_WGSL = /* wgsl */ `
struct Params {
  seed: f32,
  // hash12 seeds (s, s * 0.618) of the wheat patch rolls and of planting within a patch.
  patch1: vec2f,
  patch2: vec2f,
  plant: vec2f,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> cells: array<i32>;
// Per chunk: its slot, cx, cz.
@group(0) @binding(2) var<storage, read> jobs: array<vec4<i32>>;

const S: i32 = ${CHUNK_SIZE};
const H: i32 = ${CHUNK_HEIGHT};
const VOLUME: i32 = ${CHUNK_VOLUME};
const SEA: f32 = ${SEA_LEVEL}.0;

// tf/noise.ts, written out the way its tensor ops compute.
fn fract1(v: f32) -> f32 { return v - floor(v); }
fn hash2(a: f32, b: f32) -> f32 {
  let s = sin(a * 127.1 + b * 311.7) * 43758.5453;
  return s - floor(s);
}
fn hash3(a: f32, b: f32, c: f32) -> f32 {
  let s = sin(a * 127.1 + b * 311.7 + c * 74.7) * 43758.5453;
  return s - floor(s);
}
fn smoothstep3(t: f32) -> f32 { return t * t * (t * -2.0 + 3.0); }
fn lerp(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }

fn valueNoise2(x: f32, z: f32, seed: f32) -> f32 {
  let xi = floor(x);
  let zi = floor(z) + seed;
  let u = smoothstep3(x - floor(x));
  let v = smoothstep3(z - floor(z));
  let a = lerp(hash2(xi, zi), hash2(xi + 1.0, zi), u);
  let b = lerp(hash2(xi, zi + 1.0), hash2(xi + 1.0, zi + 1.0), u);
  return lerp(a, b, v);
}

fn valueNoise3(x: f32, y: f32, z: f32, seed: f32) -> f32 {
  let xi = floor(x);
  let yi = floor(y);
  let zi = floor(z) + seed;
  let u = smoothstep3(x - xi);
  let v = smoothstep3(y - yi);
  let w = smoothstep3(z - floor(z));
  let p0 = lerp(lerp(hash3(xi, yi, zi), hash3(xi + 1.0, yi, zi), u),
                lerp(hash3(xi, yi + 1.0, zi), hash3(xi + 1.0, yi + 1.0, zi), u), v);
  let p1 = lerp(lerp(hash3(xi, yi, zi + 1.0), hash3(xi + 1.0, yi, zi + 1.0), u),
                lerp(hash3(xi, yi + 1.0, zi + 1.0), hash3(xi + 1.0, yi + 1.0, zi + 1.0), u), v);
  return lerp(p0, p1, w);
}

fn fbm2(x: f32, z: f32, seed: f32, baseScale: f32, octaves: i32) -> f32 {
  var sum = 0.0;
  var amp = 1.0;
  var norm = 0.0;
  var scale = baseScale;
  for (var o = 0; o < octaves; o++) {
    sum = sum + valueNoise2(x / scale, z / scale, seed + f32(o) * 101.0) * amp;
    norm += amp;
    amp *= 0.5;
    scale *= 0.5;
  }
  return sum / norm;
}

fn wrap(t: f32) -> f32 { return t - floor(t / 4096.0) * 4096.0; }
fn hash12(x: f32, z: f32, s: vec2f) -> f32 {
  let a = fract1((wrap(x) + s.x) * 0.1031);
  let b = fract1((wrap(z) + s.y) * 0.1031);
  let d = a * (b + 33.33) + b * (a + 33.33) + a * (a + 33.33);
  return fract1((a + b + d * 2.0) * (a + d));
}
fn roll(x: f32, z: f32, s1: vec2f, s2: vec2f) -> bool {
  let c = f32(${Math.sqrt(WHEAT_PATCH_CHANCE)});
  return hash12(x, z, s1) < c && hash12(x, z, s2) < c;
}

@compute @workgroup_size(${CHUNK_SIZE}, ${CHUNK_SIZE})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let job = jobs[wg.x];
  let x = i32(lid.x);
  let z = i32(lid.y);
  let wx = f32(job.y * S + x);
  let wz = f32(job.z * S + z);
  let base = job.x * VOLUME + z * S + x;
  let seed = params.seed;

  // The column: ground height (terrainHeight), then grass and wheat (surfaceFeatures).
  let continents = fbm2(wx, wz, seed, 128.0, 3);
  let hills = fbm2(wx, wz, seed + 17.0, 32.0, 4);
  let height = floor((continents - 0.5) * 36.0 + (hills - 0.5) * 14.0 + ${SEA_LEVEL + 2}.0);
  let dry = height >= SEA;
  let wheat = dry && roll(floor(wx / 4.0), floor(wz / 4.0), params.patch1, params.patch2)
    && hash12(wx, wz, params.plant) < 0.6;

  for (var y = 0; y < H; y++) {
    let fy = f32(y);
    let ground = fy <= height;
    let dirt = ground && fy > height - 3.0;
    let water = fy > height && fy <= SEA;
    // Caves: only well under the surface, so the noise is only worked out there.
    var cave = false;
    if (y > 0 && fy < height - 4.0) {
      let noise = valueNoise3(wx / 14.0, fy / 9.0, wz / 14.0, seed + 999.0)
        + valueNoise3(wx / 6.0, fy / 6.0, wz / 6.0, seed + 555.0) * 0.35;
      cave = noise > 0.98;
    }
    let lava = cave && y <= 10;
    let grass = fy == height && dry;
    let plant = fy == height + 1.0 && wheat;
    let solid = ground && !cave;
    let stone = (solid && !dirt) || y == 0;
    var diamond = false;
    if (stone && y > 0 && y <= ${DIAMOND_MAX_Y}) {
      diamond = valueNoise3(wx / 3.0, fy / 3.0, wz / 3.0, seed + 333.0) > ${DIAMOND_THRESHOLD};
    }

    var c = 0;
    if (stone && !diamond) { c += ${cell(Block.Stone)}; }
    if (diamond) { c += ${cell(Block.Diamond)}; }
    if (solid && dirt && y > 0 && !grass) { c += ${cell(Block.Dirt)}; }
    if (grass) { c += ${cell(Block.Grass)}; }
    if (water && y > 0) { c += ${cell(Block.Water, SOURCE_LEVEL)}; }
    if (lava) { c += ${cell(Block.Lava, SOURCE_LEVEL)}; }
    if (plant) { c += ${cell(Block.Wheat, WHEAT_RIPE)}; }
    cells[base + y * S * S] = c;
  }
}
`;

export interface GenJob { slot: number; cx: number; cz: number }

/** The worldgen compute shader, bound to a world's cells buffer. */
export class GpuWorldgen {
  private jobs?: GPUBuffer;
  private group?: GPUBindGroup;
  private readonly params: GPUBuffer;

  /** `pipeline` from GpuWorldgen.compile. */
  constructor(private readonly device: GPUDevice, private readonly cells: GPUBuffer, private readonly pipeline: GPUComputePipeline) {
    this.params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  /** Compiles the shader without blocking the page. */
  static compile(device: GPUDevice): Promise<GPUComputePipeline> {
    return device.createComputePipelineAsync({
      label: 'world generation', layout: 'auto',
      compute: { module: device.createShaderModule({ label: 'world generation', code: WORLDGEN_WGSL }), entryPoint: 'main' },
    });
  }

  /** Generate chunks into their slots (issued now; runs in order with everything else on the queue). */
  generate(chunks: GenJob[], seed = DEFAULT_SEED): void {
    if (chunks.length === 0) return;
    const { device } = this;
    if (!this.jobs || this.jobs.size < chunks.length * 16) {
      this.jobs?.destroy();
      this.jobs = device.createBuffer({ size: Math.max(64, chunks.length) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.group = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [this.params, this.cells, this.jobs].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    }
    // The seeds tf/worldgen.ts derives in JS (as doubles, then float32), so they match exactly.
    const s = (v: number) => [v, v * 0.618];
    device.queue.writeBuffer(this.params, 0, new Float32Array([
      seed, 0, ...s((seed * 3) % 983), ...s((seed * 11) % 977 + 0.25), ...s((seed * 13) % 971 + 0.75),
    ]));
    device.queue.writeBuffer(this.jobs, 0, Int32Array.from(chunks.flatMap((c) => [c.slot, c.cx, c.cz, 0])));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group!);
    pass.dispatchWorkgroups(chunks.length);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    this.jobs?.destroy();
    this.params.destroy();
  }
}
