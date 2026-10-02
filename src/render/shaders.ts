import { Block, FALLING_LEVEL, SOURCE_LEVEL } from '../constants';
import { FACE_CORNERS, FACE_NORMALS, FULL_HEIGHT, Face, PLANT_QUADS } from './mesher';

const uniforms = /* wgsl */ `
struct Uniforms {
  viewProj: mat4x4f,
  camPos: vec4f,   // xyz = camera, w = time (s)
  sky: vec4f,      // rgb = sky / fog colour, w = fog distance
  fogCap: vec4f,   // x = how thick the fog gets (1 = plain sky; less in mist, so far terrain shows through)
};
@group(0) @binding(0) var<uniform> u: Uniforms;
`;

export const blockShader = /* wgsl */ `
${uniforms}
@group(0) @binding(1) var<storage, read> faces: array<u32>;
@group(0) @binding(2) var<storage, read> origins: array<vec4<i32>>;

// Faces come straight from GPU memory: each is one record (render/mesher.ts) that the
// vertex shader turns into a quad, 6 vertices per face. A chunk's faces are a run of the
// face buffer (render/meshPool.ts), drawn as draw(count * 6, 1, start * 6, meshSlot): the
// vertex index finds the face, the instance index the chunk's origin.
const WATER: u32 = ${Block.Water}u;
const LAVA: u32 = ${Block.Lava}u;
const FULL_HEIGHT: u32 = ${FULL_HEIGHT}u;
const SOURCE: u32 = ${SOURCE_LEVEL}u;
const FALLING: f32 = ${FALLING_LEVEL}.0;
const PLANT_FIRST: u32 = ${Face.PlantA}u;
const INSET: f32 = ${PLANT_QUADS[0][0]};

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) kind: u32,
};

fn fluidHeight(code: u32) -> f32 {
  if (code == FULL_HEIGHT) { return 1.0; }
  if (code >= SOURCE) { return 0.875; }
  return max(0.125, f32(code) / FALLING * 0.8);
}

fn faceVertex(record: u32, slot: u32, vertex: u32) -> VSOut {
  var corners = array<vec3f, 24>(${FACE_CORNERS.flat().map((c) => `vec3f(${c.join(', ')})`).join(', ')});
  var normals = array<vec3f, 6>(${FACE_NORMALS.map((n) => `vec3f(${n.join(', ')})`).join(', ')});
  // A quad's two triangles: corners 0, 1, 2 and 0, 2, 3.
  var order = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
  let k = order[vertex];
  let face = (record >> 14u) & 15u;
  let t = (record >> 18u) & 7u;
  let aux = (record >> 21u) & 15u;
  let origin = origins[slot];
  let block = vec3f(f32(origin.x + i32(record & 15u)), f32((record >> 8u) & 63u), f32(origin.z + i32((record >> 4u) & 15u)));

  var pos: vec3f;
  var normal = vec3f(0.0, 1.0, 0.0); // plants are lit like a top face
  var kind = t;
  if (face >= PLANT_FIRST) {
    // Crossed quads: corners bottom start, bottom end, top end, top start (reversed on the back).
    let plane = (face - PLANT_FIRST) / 2u;
    let back = ((face - PLANT_FIRST) & 1u) == 1u;
    let atEnd = (k == 1u || k == 2u) != back;
    let start = vec3f(select(INSET, 1.0 - INSET, plane == 1u), 0.0, INSET);
    let end = vec3f(select(1.0 - INSET, INSET, plane == 1u), 0.0, 1.0 - INSET);
    let height = 0.25 + f32(aux) * 0.1; // wheat grows with its stage
    pos = block + select(start, end, atEnd) + vec3f(0.0, select(0.0, height, k >= 2u), 0.0);
    kind = t + 16u * aux;
  } else {
    let c = corners[face * 4u + k];
    let h = select(1.0, fluidHeight(aux), t == WATER || t == LAVA);
    pos = block + vec3f(c.x, c.y * h, c.z);
    normal = normals[face];
  }

  var o: VSOut;
  var p = pos;
  // Gentle bob on fluid surfaces.
  if ((t == WATER || t == LAVA) && normal.y > 0.5) {
    p.y += 0.04 * sin(u.camPos.w * 2.0 + pos.x * 0.7 + pos.z * 0.9) - 0.04;
  }
  o.pos = u.viewProj * vec4f(p, 1.0);
  // A zero record (type air) is room left in a run: nothing to draw.
  if (t == 0u) { o.pos = vec4f(0.0, 0.0, 0.0, 1.0); }
  o.world = pos;
  o.normal = normal;
  o.kind = kind;
  return o;
}

@vertex
fn vsFace(@builtin(vertex_index) v: u32, @builtin(instance_index) slot: u32) -> VSOut {
  return faceVertex(faces[v / 6u], slot, v % 6u);
}

fn hash3(p: vec3f) -> f32 {
  return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  // 8x8 "texel" grid on each face for a pixel-art look.
  let cellPos = floor((in.world - in.normal * 0.001) * 8.0);
  let n = hash3(cellPos);
  let t = u.camPos.w;

  var base: vec3f;
  var alpha = 1.0;
  var emissive = 0.0;
  // Block type in the low 4 bits; wheat carries its growth stage above them.
  let stage = in.kind >> 4u;
  switch in.kind & 15u {
    case 1u: { base = vec3f(0.50, 0.50, 0.52) * (0.85 + 0.3 * n); }               // stone
    case 2u: { base = vec3f(0.55, 0.38, 0.24) * (0.8 + 0.35 * n); }               // dirt
    case 3u: {                                                                    // water
      let w = 0.5 + 0.5 * sin(t * 1.5 + in.world.x * 0.8 + in.world.z * 0.6 + n * 2.0);
      base = mix(vec3f(0.10, 0.30, 0.75), vec3f(0.25, 0.50, 0.90), w * 0.6);
      alpha = 0.68;
    }
    case 4u: {                                                                    // lava
      let flow = 0.5 + 0.5 * sin(t * 2.0 + n * 6.28 + in.world.x * 0.5 - in.world.z * 0.4);
      base = mix(vec3f(0.85, 0.25, 0.02), vec3f(1.0, 0.75, 0.15), flow * (0.6 + 0.4 * n));
      emissive = 1.0;
    }
    case 5u: {                                                                    // grass
      let dirt = vec3f(0.55, 0.38, 0.24) * (0.8 + 0.35 * n);
      let green = vec3f(0.36, 0.62, 0.22) * (0.8 + 0.3 * n);
      // Green on top; on the sides a ragged green fringe over dirt; dirt underneath.
      let row = floor(fract(in.world.y) * 8.0);
      let fringe = row >= 7.0 || (row >= 6.0 && n > 0.55);
      base = select(dirt, green, in.normal.y > 0.5 || (abs(in.normal.y) < 0.5 && fringe));
    }
    case 6u: {                                                                    // wheat
      let h = 0.25 + f32(stage) * 0.1;
      let along = fract(in.world.x + in.world.z * 0.37); // across the crossed quads
      let up = fract(in.world.y) / h;                    // 0 at the soil, 1 at the tip
      let ear = stage >= 4u && up > 0.7;
      // Four thin stalks, thicker ears near the tip once it has grown a while; the rest is see-through.
      if (abs(fract(along * 4.0) - 0.5) > select(0.12, 0.3, ear)) { discard; }
      base = mix(vec3f(0.30, 0.62, 0.20), vec3f(0.88, 0.74, 0.32), f32(stage) / 7.0) * (0.8 + 0.3 * n);
      if (ear) { base *= 0.85; }
    }
    case 7u: {                                                                    // diamond ore
      // Stone with clusters of cyan gems in the texel grid, faintly glowing.
      let gem = hash3(floor(cellPos / 2.0) + vec3f(17.0, 5.0, 11.0)) > 0.6 && n > 0.35;
      base = select(vec3f(0.50, 0.50, 0.52) * (0.85 + 0.3 * n), vec3f(0.30, 0.88, 0.92) * (0.8 + 0.4 * n), gem);
      emissive = select(0.0, 0.25, gem);
    }
    default: { base = vec3f(1.0, 0.0, 1.0); }
  }

  let sun = normalize(vec3f(0.4, 0.85, 0.3));
  let diffuse = max(dot(in.normal, sun), 0.0);
  let side = 0.8 + 0.2 * abs(in.normal.y) + 0.08 * abs(in.normal.x);
  var lit = base * mix((0.6 + 0.4 * diffuse) * side, 1.0, emissive);

  let dist = distance(in.world, u.camPos.xyz);
  let fog = clamp((dist - u.sky.w * 0.6) / (u.sky.w * 0.4), 0.0, u.fogCap.x);
  lit = mix(lit, u.sky.rgb, fog);
  return vec4f(lit, alpha);
}
`;

/**
 * Far terrain (world/farTerrain.ts): a height field of plain vertex positions, flat shaded
 * from screen-space derivatives. Either mist (colours under thick fog, thinning into the sky
 * toward the far edge), a dark silhouette hazing into the sky, or colours:
 * grass on land (bare dirt on steep slopes), water at the sea's surface, the blocks' sun
 * and fog. Not drawn where real chunks are.
 */
export const farShader = /* wgsl */ `
${uniforms}
struct Far {
  near: vec4f,   // xz min, xz max of the area the real chunks cover
  sea: vec4f,    // x = y of the sea's surface, y = look (0 colours, 1 silhouette, 2 mist), z = how far it reaches
};
@group(0) @binding(1) var<uniform> far: Far;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
};

@vertex
fn vs(@location(0) pos: vec3f) -> VSOut {
  var o: VSOut;
  o.pos = u.viewProj * vec4f(pos, 1.0);
  o.world = pos;
  return o;
}

fn hash2(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  // (Derivatives first: they need every pixel of the quad, so before any discard.)
  var normal = normalize(cross(dpdx(in.world), dpdy(in.world)));
  if (normal.y < 0.0) { normal = -normal; }
  if (all(in.world.xz >= far.near.xy) && all(in.world.xz < far.near.zw)) { discard; }

  let dist = distance(in.world, u.camPos.xyz);
  let sun = normalize(vec3f(0.4, 0.85, 0.3));
  if (far.sea.y > 0.5 && far.sea.y < 1.5) {
    // Silhouette: near-black land against the sky, hazier with distance so ridges stand apart.
    let haze = 0.75 * pow(clamp(dist / u.sky.w, 0.0, 1.0), 0.6);
    let shade = 0.9 + 0.1 * normal.y;
    return vec4f(mix(vec3f(0.04, 0.05, 0.08) * shade, u.sky.rgb, haze), 1.0);
  }

  let n = hash2(floor(in.world.xz / 4.0));
  var base: vec3f;
  if (in.world.y <= far.sea.x + 0.01) {
    base = vec3f(0.2, 0.38, 0.74); // about what translucent water over a sea bed looks like
    normal = vec3f(0.0, 1.0, 0.0);
  } else {
    let green = vec3f(0.36, 0.62, 0.22) * (0.85 + 0.2 * n);
    let dirt = vec3f(0.55, 0.38, 0.24) * (0.85 + 0.2 * n);
    base = mix(dirt, green, smoothstep(0.55, 0.8, normal.y));
  }

  let diffuse = max(dot(normal, sun), 0.0);
  var lit = base * (0.6 + 0.4 * diffuse);

  var fog = clamp((dist - u.sky.w * 0.6) / (u.sky.w * 0.4), 0.0, u.fogCap.x);
  if (far.sea.y > 1.5) {
    // Mist: as thick as the chunks' capped fog at their edge, thinning to plain sky at the far edge.
    fog = max(fog, mix(u.fogCap.x, 1.0, sqrt(clamp(dist / far.sea.z, 0.0, 1.0))));
  }
  lit = mix(lit, u.sky.rgb, fog);
  return vec4f(lit, 1.0);
}
`;

export const lineShader = /* wgsl */ `
${uniforms}

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
};

@vertex
fn vs(@location(0) pos: vec3f, @location(1) color: vec3f) -> VSOut {
  var o: VSOut;
  o.pos = u.viewProj * vec4f(pos, 1.0);
  o.color = color;
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
`;

/** The previous renderer's block shader, kept unchanged for safe mode (?safe): plain vertex attributes, built on the CPU. */
export const classicBlockShader = /* wgsl */ `
${uniforms}

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) kind: u32,
};

@vertex
fn vs(@location(0) pos: vec3f, @location(1) normal: vec3f, @location(2) kind: f32) -> VSOut {
  var o: VSOut;
  var p = pos;
  // Gentle bob on fluid surfaces (water 3, lava 4).
  if (kind > 2.5 && kind < 4.5 && normal.y > 0.5) {
    p.y += 0.04 * sin(u.camPos.w * 2.0 + pos.x * 0.7 + pos.z * 0.9) - 0.04;
  }
  o.pos = u.viewProj * vec4f(p, 1.0);
  o.world = pos;
  o.normal = normal;
  o.kind = u32(kind + 0.5);
  return o;
}

fn hash3(p: vec3f) -> f32 {
  return fract(sin(dot(p, vec3f(127.1, 311.7, 74.7))) * 43758.5453);
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  // 8x8 "texel" grid on each face for a pixel-art look.
  let cellPos = floor((in.world - in.normal * 0.001) * 8.0);
  let n = hash3(cellPos);
  let t = u.camPos.w;

  var base: vec3f;
  var alpha = 1.0;
  var emissive = 0.0;
  // Block type in the low 4 bits; wheat carries its growth stage above them.
  let stage = in.kind >> 4u;
  switch in.kind & 15u {
    case 1u: { base = vec3f(0.50, 0.50, 0.52) * (0.85 + 0.3 * n); }               // stone
    case 2u: { base = vec3f(0.55, 0.38, 0.24) * (0.8 + 0.35 * n); }               // dirt
    case 3u: {                                                                    // water
      let w = 0.5 + 0.5 * sin(t * 1.5 + in.world.x * 0.8 + in.world.z * 0.6 + n * 2.0);
      base = mix(vec3f(0.10, 0.30, 0.75), vec3f(0.25, 0.50, 0.90), w * 0.6);
      alpha = 0.68;
    }
    case 4u: {                                                                    // lava
      let flow = 0.5 + 0.5 * sin(t * 2.0 + n * 6.28 + in.world.x * 0.5 - in.world.z * 0.4);
      base = mix(vec3f(0.85, 0.25, 0.02), vec3f(1.0, 0.75, 0.15), flow * (0.6 + 0.4 * n));
      emissive = 1.0;
    }
    case 5u: {                                                                    // grass
      let dirt = vec3f(0.55, 0.38, 0.24) * (0.8 + 0.35 * n);
      let green = vec3f(0.36, 0.62, 0.22) * (0.8 + 0.3 * n);
      // Green on top; on the sides a ragged green fringe over dirt; dirt underneath.
      let row = floor(fract(in.world.y) * 8.0);
      let fringe = row >= 7.0 || (row >= 6.0 && n > 0.55);
      base = select(dirt, green, in.normal.y > 0.5 || (abs(in.normal.y) < 0.5 && fringe));
    }
    case 6u: {                                                                    // wheat
      let h = 0.25 + f32(stage) * 0.1;
      let along = fract(in.world.x + in.world.z * 0.37); // across the crossed quads
      let up = fract(in.world.y) / h;                    // 0 at the soil, 1 at the tip
      let ear = stage >= 4u && up > 0.7;
      // Four thin stalks, thicker ears near the tip once it has grown a while; the rest is see-through.
      if (abs(fract(along * 4.0) - 0.5) > select(0.12, 0.3, ear)) { discard; }
      base = mix(vec3f(0.30, 0.62, 0.20), vec3f(0.88, 0.74, 0.32), f32(stage) / 7.0) * (0.8 + 0.3 * n);
      if (ear) { base *= 0.85; }
    }
    case 7u: {                                                                    // diamond ore
      // Stone with clusters of cyan gems in the texel grid, faintly glowing.
      let gem = hash3(floor(cellPos / 2.0) + vec3f(17.0, 5.0, 11.0)) > 0.6 && n > 0.35;
      base = select(vec3f(0.50, 0.50, 0.52) * (0.85 + 0.3 * n), vec3f(0.30, 0.88, 0.92) * (0.8 + 0.4 * n), gem);
      emissive = select(0.0, 0.25, gem);
    }
    default: { base = vec3f(1.0, 0.0, 1.0); }
  }

  let sun = normalize(vec3f(0.4, 0.85, 0.3));
  let diffuse = max(dot(in.normal, sun), 0.0);
  let side = 0.8 + 0.2 * abs(in.normal.y) + 0.08 * abs(in.normal.x);
  var lit = base * mix((0.6 + 0.4 * diffuse) * side, 1.0, emissive);

  let dist = distance(in.world, u.camPos.xyz);
  let fog = clamp((dist - u.sky.w * 0.6) / (u.sky.w * 0.4), 0.0, u.fogCap.x);
  lit = mix(lit, u.sky.rgb, fog);
  return vec4f(lit, alpha);
}
`;
