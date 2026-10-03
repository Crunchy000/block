// Converts the animals in assets/cube-pets/ ("Cube Pets" by Kenney, CC0: see its License.txt)
// into what the game loads: public/models/<name>.bin, and their shared palette texture
// public/models/pets.png. Run after changing them: node scripts/convert-models.mjs
//
// The source .glb files keep their texture outside (Textures/colormap.png, shared by all of
// them), and animate whole parts (legs, body, head, tail, wings) rather than a skeleton. Here
// each part's vertices stay in the part's own space and every animation clip is sampled at
// FPS into a matrix per part per frame (the hierarchy and the model's placement baked in), so
// the game plays a clip by picking two frames and blending.
//
// <name>.bin, little endian:
//   u32 JSON length, then the JSON (space-padded to 4 bytes):
//     { vertices, indices, parts, frames, fps, clips: { <name>: { start, frames } }, size: [x, y, z] }
//   vertices × 9 f32: position xyz (in its part's space), normal xyz, uv, part index
//   indices × u16, padded to 4 bytes
//   frames × parts × 12 f32: each part's matrix, its 3 rows (x, y, z, translation) in turn
// Positions come out in blocks, the model standing on y = 0, centred on x and z, facing -z
// (the way the game faces at yaw 0), `height` tall in its idle pose.
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';

const at = (path) => new URL(`../${path}`, import.meta.url);

/** The animals: the source model and how tall it stands (blocks). */
const MODELS = [
  { name: 'cow', source: 'animal-cow', height: 1.3 },
  { name: 'pig', source: 'animal-pig', height: 0.85 },
  { name: 'chicken', source: 'animal-chick', height: 0.6 },
  { name: 'rabbit', source: 'animal-bunny', height: 0.6 },
  { name: 'cat', source: 'animal-cat', height: 0.7 },
  { name: 'dog', source: 'animal-dog', height: 0.8 },
  { name: 'deer', source: 'animal-deer', height: 1.5 },
  { name: 'fox', source: 'animal-fox', height: 0.75 },
];
/** The clips the game plays, sampled at this rate. */
const CLIPS = ['idle', 'walk', 'run', 'eat'];
const FPS = 30;

function load(source) {
  const file = readFileSync(at(`assets/cube-pets/${source}.glb`));
  const jsonLength = file.readUInt32LE(12);
  const gltf = JSON.parse(file.subarray(20, 20 + jsonLength).toString());
  const binStart = 20 + jsonLength + 8;
  const bin = file.subarray(binStart, binStart + file.readUInt32LE(20 + jsonLength));
  const SIZES = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
  const TYPES = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array };
  const read = (index) => {
    const a = gltf.accessors[index], v = gltf.bufferViews[a.bufferView], Type = TYPES[a.componentType], n = SIZES[a.type];
    const stride = (v.byteStride ?? n * Type.BYTES_PER_ELEMENT) / Type.BYTES_PER_ELEMENT;
    const start = bin.byteOffset + (v.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const all = new Type(bin.buffer.slice(start, bin.byteOffset + (v.byteOffset ?? 0) + v.byteLength));
    return Array.from({ length: a.count }, (_, i) => Array.from({ length: n }, (_, k) => all[i * stride + k]));
  };
  return { gltf, read };
}

// 4x4 matrices, column-major as glTF stores them.
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const mul = (a, b) => Array.from({ length: 16 }, (_, i) => {
  const r = i % 4, c = Math.floor(i / 4);
  return a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
});
const trs = ({ translation: t = [0, 0, 0], rotation: [x, y, z, w] = [0, 0, 0, 1], scale: [sx, sy, sz] = [1, 1, 1] }) => [
  (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
  2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
  2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
  t[0], t[1], t[2], 1,
];
const apply = (m, [x, y, z], w = 1) => [0, 1, 2].map((r) => m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r] * w);

/** A channel's value at time t (linear: lerp, normalised lerp for rotations). */
function sample(times, values, t, rotation) {
  if (t <= times[0]) return values[0];
  const k = times.findIndex((x) => x > t);
  if (k < 0) return values[values.length - 1];
  const f = (t - times[k - 1]) / (times[k] - times[k - 1]), a = values[k - 1], b = values[k];
  if (!rotation) return a.map((v, i) => v + (b[i] - v) * f);
  const sign = a.reduce((s, v, i) => s + v * b[i], 0) < 0 ? -1 : 1;
  const q = a.map((v, i) => v + (sign * b[i] - v) * f), l = Math.hypot(...q);
  return q.map((v) => v / l);
}

function convert(spec) {
  const { gltf, read } = load(spec.source);
  const parent = gltf.nodes.map((_, i) => gltf.nodes.findIndex((p) => p.children?.includes(i)));
  const partNodes = gltf.nodes.map((n, i) => (n.mesh !== undefined ? i : -1)).filter((i) => i >= 0);

  /** Every node's model-space matrix with a clip at time t (or the rest pose). */
  const pose = (clip, t) => {
    const local = gltf.nodes.map((n) => ({ translation: n.translation, rotation: n.rotation, scale: n.scale }));
    for (const ch of clip?.channels ?? []) {
      const s = clip.samplers[ch.sampler];
      local[ch.target.node][ch.target.path] = sample(read(s.input).map(([x]) => x), read(s.output), t, ch.target.path === 'rotation');
    }
    const world = [];
    const get = (i) => (world[i] ??= parent[i] < 0 ? trs(local[i]) : mul(get(parent[i]), trs(local[i])));
    return gltf.nodes.map((_, i) => get(i));
  };

  // The parts' vertices in their own space.
  const verts = [], indices = [];
  partNodes.forEach((node, part) => {
    for (const prim of gltf.meshes[gltf.nodes[node].mesh].primitives) {
      const tt = gltf.materials[prim.material]?.pbrMetallicRoughness?.baseColorTexture?.extensions?.KHR_texture_transform ?? {};
      const pos = read(prim.attributes.POSITION), nrm = read(prim.attributes.NORMAL);
      const uv = read(prim.attributes.TEXCOORD_0).map(([u, v]) => [u * (tt.scale?.[0] ?? 1) + (tt.offset?.[0] ?? 0), v * (tt.scale?.[1] ?? 1) + (tt.offset?.[1] ?? 0)]);
      const base = verts.length;
      pos.forEach((p, i) => verts.push([...p, ...nrm[i], ...uv[i], part]));
      for (const [i] of prim.indices === undefined ? pos.map((_, i) => [i]) : read(prim.indices)) indices.push(base + i);
    }
  });
  if (verts.length > 65535) throw new Error(`${spec.name}: ${verts.length} vertices, more than u16 indices hold`);

  // Placement: stand the idle pose on y = 0, centred, `height` tall, turned to face -z (from +z).
  const clipOf = (name) => gltf.animations.find((a) => a.name === name);
  const rest = pose(clipOf('idle'), 0);
  const placed = verts.map((v) => apply(rest[partNodes[v[8]]], v));
  const min = [0, 1, 2].map((k) => Math.min(...placed.map((p) => p[k])));
  const max = [0, 1, 2].map((k) => Math.max(...placed.map((p) => p[k])));
  const s = spec.height / (max[1] - min[1]);
  const place = mul([-s, 0, 0, 0, 0, s, 0, 0, 0, 0, -s, 0, s * (min[0] + max[0]) / 2, -s * min[1], s * (min[2] + max[2]) / 2, 1], IDENTITY);

  // Every clip, sampled: frames 0..n where n = duration × FPS (the last as the first for loops).
  const clips = {}, matrices = [];
  for (const name of CLIPS) {
    const clip = clipOf(name);
    if (!clip) throw new Error(`${spec.name}: no ${name} clip`);
    const duration = Math.max(...clip.samplers.map((sm) => gltf.accessors[sm.input].max[0]));
    const n = Math.max(1, Math.round(duration * FPS));
    clips[name] = { start: matrices.length / partNodes.length, frames: n + 1 };
    for (let f = 0; f <= n; f++) {
      const world = pose(clip, (f / n) * duration);
      for (const node of partNodes) {
        const m = mul(place, world[node]);
        matrices.push([0, 1, 2].flatMap((r) => [m[r], m[4 + r], m[8 + r], m[12 + r]]));
      }
    }
  }
  const frames = matrices.length / partNodes.length;

  const size = [0, 1, 2].map((k) => +((max[k] - min[k]) * s).toFixed(2));
  const json = Buffer.from(JSON.stringify({ vertices: verts.length, indices: indices.length, parts: partNodes.length, frames, fps: FPS, clips, size }));
  const head = Buffer.alloc(4 + Math.ceil(json.length / 4) * 4, 0x20);
  head.writeUInt32LE(head.length - 4, 0);
  json.copy(head, 4);
  const indexBytes = Math.ceil((indices.length * 2) / 4) * 4;
  const body = Buffer.alloc(verts.length * 36 + indexBytes + matrices.length * 48);
  new Float32Array(body.buffer, body.byteOffset, verts.length * 9).set(verts.flat());
  new Uint16Array(body.buffer, body.byteOffset + verts.length * 36, indices.length).set(indices);
  new Float32Array(body.buffer, body.byteOffset + verts.length * 36 + indexBytes, matrices.length * 12).set(matrices.flat());
  const out = Buffer.concat([head, body]);
  writeFileSync(at(`public/models/${spec.name}.bin`), out);
  console.log(`${spec.name}: ${size.join(' x ')} blocks, ${partNodes.length} parts, ${verts.length} vertices, ${indices.length / 3} triangles, `
    + `${Object.entries(clips).map(([k, c]) => `${k} ${c.frames}`).join(', ')} frames, ${(out.length / 1024).toFixed(0)} KB`);
}

for (const spec of MODELS) convert(spec);
copyFileSync(at('assets/cube-pets/Textures/colormap.png'), at('public/models/pets.png'));
