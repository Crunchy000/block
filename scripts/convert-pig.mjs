// Converts assets/pig.glb (meshopt-compressed, quantized; from a generator) into what the
// game loads: public/models/pig.bin (plain vertices and indices) and public/models/pig.webp
// (its colour texture). Run after replacing the model: node scripts/convert-pig.mjs
//
// pig.bin, little endian: u32 vertex count, u32 index count, then f32 positions (x, y, z),
// f32 normals (x, y, z), f32 uvs (u, v) per vertex, then u16 indices. Positions are in
// blocks, the model standing on y = 0, centred on x and z, its snout toward -z.
import { readFileSync, writeFileSync } from 'node:fs';
import { MeshoptDecoder } from 'meshoptimizer';

/** Pig size in blocks: Minecraft's pig is about 0.9 tall and 1.25 long. */
const HEIGHT = 0.9;

const glb = readFileSync(new URL('../assets/pig.glb', import.meta.url));
const jsonLength = glb.readUInt32LE(12);
const gltf = JSON.parse(glb.subarray(20, 20 + jsonLength).toString());
const binStart = 20 + jsonLength + 8;
const bin = glb.subarray(binStart, binStart + glb.readUInt32LE(20 + jsonLength));

await MeshoptDecoder.ready;
/** A buffer view's bytes, decompressed if it uses EXT_meshopt_compression. */
function view(index) {
  const v = gltf.bufferViews[index];
  const m = v.extensions?.EXT_meshopt_compression;
  if (!m) return bin.subarray(v.byteOffset ?? 0, (v.byteOffset ?? 0) + v.byteLength);
  const out = new Uint8Array(m.count * m.byteStride);
  const src = bin.subarray(m.byteOffset ?? 0, (m.byteOffset ?? 0) + m.byteLength);
  MeshoptDecoder.decodeGltfBuffer(out, m.count, m.byteStride, src, m.mode, m.filter ?? 'NONE');
  return Buffer.from(out.buffer);
}

const SIZES = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
const TYPES = {
  5120: [Int8Array, 127], 5121: [Uint8Array, 255], 5122: [Int16Array, 32767], 5123: [Uint16Array, 65535], 5125: [Uint32Array, 0], 5126: [Float32Array, 0],
};
/** An accessor's values as floats, one array per element. */
function read(index) {
  const a = gltf.accessors[index];
  const v = gltf.bufferViews[a.bufferView];
  const bytes = view(a.bufferView);
  const [Type, max] = TYPES[a.componentType];
  const n = SIZES[a.type], stride = (v.byteStride ?? n * Type.BYTES_PER_ELEMENT) / Type.BYTES_PER_ELEMENT;
  const all = new Type(bytes.buffer, bytes.byteOffset + (a.byteOffset ?? 0), Math.floor((bytes.byteLength - (a.byteOffset ?? 0)) / Type.BYTES_PER_ELEMENT));
  return Array.from({ length: a.count }, (_, i) => Array.from({ length: n }, (_, k) => {
    const x = all[i * stride + k];
    return a.normalized ? Math.max(x / max, -1) : x;
  }));
}

const prim = gltf.meshes[0].primitives[0];
let pos = read(prim.attributes.POSITION);
const nrm = read(prim.attributes.NORMAL).map(([x, y, z]) => { const l = Math.hypot(x, y, z) || 1; return [x / l, y / l, z / l]; });
const tex = gltf.materials[prim.material].pbrMetallicRoughness.baseColorTexture;
const tt = tex.extensions?.KHR_texture_transform ?? {};
const uv = read(prim.attributes.TEXCOORD_0).map(([u, v]) => [u * (tt.scale?.[0] ?? 1) + (tt.offset?.[0] ?? 0), v * (tt.scale?.[1] ?? 1) + (tt.offset?.[1] ?? 0)]);
const indices = read(prim.indices).map(([i]) => i);

// The node transforms (translation and uniform scale), innermost first.
const parentOf = (n) => gltf.nodes.findIndex((p) => p.children?.includes(n));
for (let n = gltf.nodes.findIndex((node) => node.mesh === 0); n >= 0; n = parentOf(n)) {
  const { translation: t = [0, 0, 0], scale: s = [1, 1, 1] } = gltf.nodes[n];
  pos = pos.map((p) => p.map((x, k) => t[k] + s[k] * x));
}

// Stand it on y = 0, centred, HEIGHT tall.
const min = [0, 1, 2].map((k) => Math.min(...pos.map((p) => p[k])));
const max = [0, 1, 2].map((k) => Math.max(...pos.map((p) => p[k])));
const scale = HEIGHT / (max[1] - min[1]);
// The model's snout points to +z: turn it around (180° about y) so it faces -z, the way the game faces at yaw 0.
pos = pos.map(([x, y, z]) => [-(x - (min[0] + max[0]) / 2) * scale, (y - min[1]) * scale, -(z - (min[2] + max[2]) / 2) * scale]);
const normals = nrm.map(([x, y, z]) => [-x, y, -z]);
console.log('size (blocks):', [0, 1, 2].map((k) => ((max[k] - min[k]) * scale).toFixed(2)).join(' x '), `${pos.length} vertices, ${indices.length / 3} triangles`);

const n = pos.length;
const out = Buffer.alloc(8 + n * 32 + indices.length * 2);
out.writeUInt32LE(n, 0);
out.writeUInt32LE(indices.length, 4);
const f = new Float32Array(out.buffer, out.byteOffset + 8, n * 8);
f.set(pos.flat(), 0);
f.set(normals.flat(), n * 3);
f.set(uv.flat(), n * 6);
new Uint16Array(out.buffer, out.byteOffset + 8 + n * 32, indices.length).set(indices);
writeFileSync(new URL('../public/models/pig.bin', import.meta.url), out);

const image = gltf.images[tex.extensions?.EXT_texture_webp?.source ?? gltf.textures[tex.index].extensions.EXT_texture_webp.source];
writeFileSync(new URL('../public/models/pig.webp', import.meta.url), view(image.bufferView));
console.log('wrote public/models/pig.bin and pig.webp');
