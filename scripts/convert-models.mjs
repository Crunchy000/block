// Converts the mob models in assets/ into what the game loads (public/models/<name>.bin plus
// their textures). Run after changing a model: node scripts/convert-models.mjs [--all-farm]
//
//   assets/farm/        "Cube Farm Animals" by ezgi bakim, CC-BY-4.0 (see assets/farm/license.txt):
//                       eight animals in one glTF scene sharing a palette texture
//   assets/pig.glb      an earlier pig (meshopt-compressed, quantized, a WebP texture), not used now
//
// <name>.bin, little endian: u32 vertex count, u32 index count, then per vertex 12 floats
// (position xyz, normal xyz, uv, colour rgba: where alpha is 1 the colour replaces the
// texture), then u16 indices. Positions are in blocks, the model standing on y = 0, centred
// on x and z, facing -z (the way the game faces at yaw 0).
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';
import { MeshoptDecoder, MeshoptSimplifier } from 'meshoptimizer';

const at = (path) => new URL(`../${path}`, import.meta.url);

/**
 * The models: a source file, the scene nodes making up the animal, its height in blocks,
 * whether it faces +z in the source (turned around here), and at most this many triangles.
 * (Which farm node is which animal: node scripts/convert-models.mjs --all-farm.)
 */
const FARM = 'assets/farm/scene.gltf';
const MODELS = [
  { name: 'horse', source: FARM, nodes: [3], height: 1.6, facesPlusZ: true, texture: 'farm.png', maxTriangles: 2500 },
  { name: 'pig', source: FARM, nodes: [6], height: 0.9, facesPlusZ: true, texture: 'farm.png', maxTriangles: 2500 },
  { name: 'cat', source: FARM, nodes: [8], height: 0.7, facesPlusZ: true, texture: 'farm.png', maxTriangles: 2500 },
  { name: 'rabbit', source: FARM, nodes: [10], height: 0.5, facesPlusZ: true, texture: 'farm.png', maxTriangles: 2500 },
  { name: 'chicken', source: FARM, nodes: [12], height: 0.7, facesPlusZ: true, texture: 'farm.png', maxTriangles: 2500 },
  { name: 'sheep', source: FARM, nodes: [14], height: 1.3, facesPlusZ: true, texture: 'farm.png', maxTriangles: 2500 },
  { name: 'mouse', source: FARM, nodes: [17], height: 0.45, facesPlusZ: true, texture: 'farm.png', maxTriangles: 2500 },
  { name: 'cow', source: FARM, nodes: [19], height: 1.4, facesPlusZ: true, texture: 'farm.png', maxTriangles: 2500 },
  // The first pig (assets/pig.glb, a different style), not used in the game now:
  // { name: 'pig-glb', source: 'assets/pig.glb', nodes: 'all', height: 0.9, facesPlusZ: true, texture: 'pig.webp' },
];
/** --all-farm: every animal of the farm scene as farm<node>.bin (to see which is which). */
const FARM_NODES = [3, 6, 8, 10, 12, 14, 17, 19];

await MeshoptDecoder.ready;
await MeshoptSimplifier.ready;

/** A glTF (.glb, or .gltf with its buffers beside it): its JSON and a buffer view reader. */
function load(source) {
  const file = readFileSync(at(source));
  let gltf, buffers;
  if (source.endsWith('.glb')) {
    const jsonLength = file.readUInt32LE(12);
    gltf = JSON.parse(file.subarray(20, 20 + jsonLength).toString());
    const binStart = 20 + jsonLength + 8;
    buffers = [file.subarray(binStart, binStart + file.readUInt32LE(20 + jsonLength))];
  } else {
    gltf = JSON.parse(file.toString());
    const dir = source.slice(0, source.lastIndexOf('/') + 1);
    buffers = gltf.buffers.map((b) => (b.uri ? readFileSync(at(dir + b.uri)) : undefined));
  }
  const view = (index) => {
    const v = gltf.bufferViews[index];
    const m = v.extensions?.EXT_meshopt_compression;
    if (!m) return buffers[v.buffer].subarray(v.byteOffset ?? 0, (v.byteOffset ?? 0) + v.byteLength);
    const out = new Uint8Array(m.count * m.byteStride);
    MeshoptDecoder.decodeGltfBuffer(out, m.count, m.byteStride, buffers[m.buffer].subarray(m.byteOffset ?? 0, (m.byteOffset ?? 0) + m.byteLength), m.mode, m.filter ?? 'NONE');
    return Buffer.from(out.buffer);
  };
  return { gltf, view, dir: source.slice(0, source.lastIndexOf('/') + 1) };
}

const SIZES = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
const TYPES = { 5120: [Int8Array, 127], 5121: [Uint8Array, 255], 5122: [Int16Array, 32767], 5123: [Uint16Array, 65535], 5125: [Uint32Array, 0], 5126: [Float32Array, 0] };
function read({ gltf, view }, index) {
  const a = gltf.accessors[index];
  const v = gltf.bufferViews[a.bufferView];
  const bytes = view(a.bufferView);
  const [Type, max] = TYPES[a.componentType];
  const n = SIZES[a.type], stride = (v.byteStride ?? n * Type.BYTES_PER_ELEMENT) / Type.BYTES_PER_ELEMENT;
  const start = bytes.byteOffset + (a.byteOffset ?? 0);
  const all = new Type(bytes.buffer.slice(start, bytes.byteOffset + bytes.byteLength));
  return Array.from({ length: a.count }, (_, i) => Array.from({ length: n }, (_, k) => {
    const x = all[i * stride + k];
    return a.normalized ? Math.max(x / max, -1) : x;
  }));
}

// 4x4 matrices, column-major as glTF stores them.
const mul = (a, b) => Array.from({ length: 16 }, (_, i) => {
  const r = i % 4, c = Math.floor(i / 4);
  return a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
});
function local(node) {
  if (node.matrix) return node.matrix;
  const [x, y, z, w] = node.rotation ?? [0, 0, 0, 1], [sx, sy, sz] = node.scale ?? [1, 1, 1], t = node.translation ?? [0, 0, 0];
  return [
    (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
    2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
    2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
    t[0], t[1], t[2], 1,
  ];
}
const apply = (m, [x, y, z], w = 1) => [0, 1, 2].map((r) => m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r] * w);

/** Every mesh primitive under the given nodes, with its world matrix. */
function primitives(model, roots) {
  const { gltf } = model;
  const parent = (n) => gltf.nodes.findIndex((p) => p.children?.includes(n));
  const world = (n) => (n < 0 ? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] : mul(world(parent(n)), local(gltf.nodes[n])));
  const out = [];
  const visit = (n) => {
    const node = gltf.nodes[n];
    if (node.mesh !== undefined) for (const p of gltf.meshes[node.mesh].primitives) out.push({ prim: p, matrix: world(n) });
    for (const c of node.children ?? []) visit(c);
  };
  for (const r of roots === 'all' ? gltf.nodes.map((_, i) => i).filter((i) => parent(i) < 0) : roots) visit(r);
  return out;
}

const toSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

function convert(spec, out) {
  const model = load(spec.source);
  const verts = [], indices = [];
  for (const { prim, matrix } of primitives(model, spec.nodes)) {
    const material = model.gltf.materials[prim.material] ?? {};
    const pbr = material.pbrMetallicRoughness ?? {};
    const tex = pbr.baseColorTexture;
    const tt = tex?.extensions?.KHR_texture_transform ?? {};
    // A plain colour (no texture) replaces the texture: alpha 1. In sRGB, as the game lights colours.
    const colour = tex ? [1, 1, 1, 0] : [...(pbr.baseColorFactor ?? [1, 1, 1]).slice(0, 3).map(toSrgb), 1];
    const pos = read(model, prim.attributes.POSITION).map((p) => apply(matrix, p));
    const nrm = read(model, prim.attributes.NORMAL).map((n) => {
      const v = apply(matrix, n, 0), l = Math.hypot(...v) || 1;
      return v.map((x) => x / l);
    });
    const uv = prim.attributes.TEXCOORD_0 === undefined ? pos.map(() => [0, 0])
      : read(model, prim.attributes.TEXCOORD_0).map(([u, v]) => [u * (tt.scale?.[0] ?? 1) + (tt.offset?.[0] ?? 0), v * (tt.scale?.[1] ?? 1) + (tt.offset?.[1] ?? 0)]);
    const base = verts.length;
    pos.forEach((p, i) => verts.push([...p, ...nrm[i], ...uv[i], ...colour]));
    const idx = prim.indices === undefined ? pos.map((_, i) => i) : read(model, prim.indices).map(([i]) => i);
    for (const i of idx) indices.push(base + i);
  }

  // Stand it on y = 0, centred, `height` tall, facing -z.
  const min = [0, 1, 2].map((k) => Math.min(...verts.map((v) => v[k])));
  const max = [0, 1, 2].map((k) => Math.max(...verts.map((v) => v[k])));
  const s = spec.height / (max[1] - min[1]), turn = spec.facesPlusZ ? -1 : 1;
  for (const v of verts) {
    v[0] = turn * (v[0] - (min[0] + max[0]) / 2) * s;
    v[1] = (v[1] - min[1]) * s;
    v[2] = turn * (v[2] - (min[2] + max[2]) / 2) * s;
    v[3] *= turn;
    v[5] *= turn;
  }

  // Fewer triangles where there are more than it needs (keeping texture seams: borders locked).
  let tris = Uint32Array.from(indices);
  if (spec.maxTriangles && tris.length / 3 > spec.maxTriangles) {
    const positions = Float32Array.from(verts.flatMap((v) => v.slice(0, 3)));
    [tris] = MeshoptSimplifier.simplify(tris, positions, 3, spec.maxTriangles * 3, 0.01, ['LockBorder']);
  }
  // Keep only the vertices used, renumbered.
  const remap = new Map();
  const kept = [];
  const finalIndices = Array.from(tris, (i) => {
    if (!remap.has(i)) { remap.set(i, kept.length); kept.push(verts[i]); }
    return remap.get(i);
  });
  if (kept.length > 65535) throw new Error(`${spec.name}: ${kept.length} vertices, more than u16 indices hold`);

  const n = kept.length, m = finalIndices.length;
  const buf = Buffer.alloc(8 + n * 48 + m * 2);
  buf.writeUInt32LE(n, 0);
  buf.writeUInt32LE(m, 4);
  new Float32Array(buf.buffer, buf.byteOffset + 8, n * 12).set(kept.flat());
  new Uint16Array(buf.buffer, buf.byteOffset + 8 + n * 48, m).set(finalIndices);
  writeFileSync(at(`public/models/${out}.bin`), buf);
  const size = [0, 1, 2].map((k) => ((max[k] - min[k]) * s).toFixed(2)).join(' x ');
  console.log(`${out}: ${size} blocks, ${n} vertices, ${m / 3} triangles (from ${indices.length / 3}), ${(buf.length / 1024).toFixed(0)} KB`);

  // The texture beside it.
  const img = model.gltf.images?.[0];
  if (img?.uri) copyFileSync(at(model.dir + img.uri), at(`public/models/${spec.texture}`));
  else if (img) {
    const source = model.gltf.textures[0].extensions?.EXT_texture_webp?.source ?? model.gltf.textures[0].source;
    writeFileSync(at(`public/models/${spec.texture}`), model.view(model.gltf.images[source].bufferView));
  }
}

if (process.argv.includes('--all-farm')) {
  for (const node of FARM_NODES) convert({ name: `farm${node}`, source: 'assets/farm/scene.gltf', nodes: [node], height: 1, texture: 'farm.png', maxTriangles: 4000 }, `farm${node}`);
} else {
  for (const spec of MODELS) convert(spec, spec.name);
}
