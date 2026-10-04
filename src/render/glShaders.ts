import { Block, CONCRETE_COLOURS, GRAVEL_LEVEL, SAND_LEVEL, SEA_LEVEL } from '../constants';
import { SAND_ABOVE } from '../tf/worldgen';
import { Layer } from './blockTextures';
import { PLANT_QUADS } from './mesher';

/**
 * The WebGL2 renderer's shaders (render/glRenderer.ts), GLSL ES 3.00: the WebGPU ones
 * (render/shaders.ts) written again, safe mode's block shader among them (vertices built on
 * the CPU, no fade-in), and without the procedural block look (until the textures load,
 * blocks are plain colours).
 */
const head = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler2DArray;
uniform mat4 u_viewProj;
uniform vec4 u_camPos;  // xyz = camera, w = time (s)
uniform vec4 u_sky;     // rgb = sky / fog colour, w = fog distance
uniform vec4 u_fogCap;  // x = how thick the fog gets, y = 1 once blocks are textured
`;

const f = (v: number) => (Number.isInteger(v) ? `${v}.0` : `${v}`);
const vec3 = (c: readonly number[]) => `vec3(${c.map(f).join(', ')})`;
const INSET = PLANT_QUADS[0][0];

export const glBlockVertex = /* glsl */ `${head}
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in float a_kind;
out vec3 v_world;
out vec3 v_normal;
flat out uint v_kind;
void main() {
  vec3 p = a_pos;
  // Gentle bob on fluid surfaces (water 3, lava 4).
  if (a_kind > 2.5 && a_kind < 4.5 && a_normal.y > 0.5) {
    p.y += 0.04 * sin(u_camPos.w * 2.0 + a_pos.x * 0.7 + a_pos.z * 0.9) - 0.04;
  }
  gl_Position = u_viewProj * vec4(p, 1.0);
  v_world = a_pos;
  v_normal = a_normal;
  v_kind = uint(a_kind + 0.5);
}
`;

export const glBlockFragment = /* glsl */ `${head}
uniform sampler2DArray u_blocks;
in vec3 v_world;
in vec3 v_normal;
flat in uint v_kind;
out vec4 o_colour;

// Which texture layer a face shows (Layer in render/blockTextures.ts), or -1 for none (concrete).
int layerFor(uint kind, vec3 normal, float t) {
  uint stage = kind >> 4u;
  switch (kind & 15u) {
    case 1u: return stage == 0u ? ${Layer.Stone} : stage == ${SAND_LEVEL}u ? ${Layer.Sand} : stage == ${GRAVEL_LEVEL}u ? ${Layer.Gravel} : -1;
    case 2u: return ${Layer.Dirt};
    case 3u: return ${Layer.Water} + int(uint(t * 8.0) % ${Layer.WaterFrames}u);
    case 4u: return ${Layer.Lava} + int(uint(t * 2.5) % ${Layer.LavaFrames}u);
    case 5u: return normal.y < -0.5 ? ${Layer.Dirt} : normal.y > 0.5 ? ${Layer.GrassTop} : ${Layer.GrassSide};
    case 6u: return ${Layer.Wheat} + int(min(stage, 7u));
    case 7u: return ${Layer.Diamond};
    default: return -1;
  }
}

void main() {
  // Texture coordinates by world position (tops by x and z, sides by their horizontal axis and
  // height), so blocks tile the repeating texture without seams; wheat by its place across the block.
  vec3 local = v_world - floor(v_world - v_normal * 0.001);
  vec2 uv = abs(v_normal.y) > 0.5 ? v_world.xz : abs(v_normal.x) > 0.5 ? vec2(v_world.z, -v_world.y) : vec2(v_world.x, -v_world.y);
  if ((v_kind & 15u) == ${Block.Wheat}u) uv = vec2((local.z - ${f(INSET)}) / ${f(1 - 2 * INSET)}, 1.0 - local.y);
  vec2 duvx = dFdx(uv);
  vec2 duvy = dFdy(uv);
  float t = u_camPos.w;
  uint btype = v_kind & 15u, stage = v_kind >> 4u;

  vec3 base;
  float alpha = 1.0;
  float emissive = 0.0;
  int layer = u_fogCap.y > 0.5 ? layerFor(v_kind, v_normal, t) : -1;
  if (btype == 1u && stage >= 1u && stage <= ${CONCRETE_COLOURS.length}u) {
    // Pastel concrete: plain, staying pastel in shade.
    vec3 concrete[${CONCRETE_COLOURS.length}] = vec3[](${CONCRETE_COLOURS.map((c) => vec3(c.rgb)).join(', ')});
    base = concrete[stage - 1u];
    emissive = 0.4;
  } else if (layer >= 0) {
    vec4 c = textureGrad(u_blocks, vec3(uv, float(layer)), duvx, duvy);
    base = c.rgb;
    if (btype == 3u) alpha = c.a;
    if (btype == 4u) emissive = 1.0;
    if (btype == 6u && c.a < 0.35) discard;
    if (btype == 7u) emissive = 0.3 * clamp((c.b - c.r) * 3.0, 0.0, 1.0);
  } else {
    // Until the textures load: plain colours.
    vec3 colours[8] = vec3[](vec3(1.0, 0.0, 1.0), vec3(0.5, 0.5, 0.52), vec3(0.55, 0.38, 0.24), vec3(0.15, 0.38, 0.8),
      vec3(0.95, 0.5, 0.1), vec3(0.36, 0.62, 0.22), vec3(0.8, 0.7, 0.3), vec3(0.4, 0.75, 0.8));
    base = colours[btype];
    if (btype == 1u && stage == ${SAND_LEVEL}u) base = vec3(0.86, 0.8, 0.58);
    if (btype == 1u && stage == ${GRAVEL_LEVEL}u) base = vec3(0.52, 0.5, 0.49);
    if (btype == 5u && v_normal.y < 0.5) base = vec3(0.55, 0.38, 0.24);
    if (btype == 3u) alpha = 0.68;
    if (btype == 4u) emissive = 1.0;
  }

  vec3 sun = normalize(vec3(0.4, 0.85, 0.3));
  float diffuse = max(dot(v_normal, sun), 0.0);
  float side = 0.8 + 0.2 * abs(v_normal.y) + 0.08 * abs(v_normal.x);
  vec3 lit = base * mix((0.6 + 0.4 * diffuse) * side, 1.0, emissive);
  float dist = distance(v_world, u_camPos.xyz);
  float fog = clamp((dist - u_sky.w * 0.6) / (u_sky.w * 0.4), 0.0, u_fogCap.x);
  o_colour = vec4(mix(lit, u_sky.rgb, fog), alpha);
}
`;

export const glFarVertex = /* glsl */ `${head}
layout(location = 0) in vec3 a_pos;
out vec3 v_world;
void main() {
  // A little lower than the blocks, so a chunk fading in over it doesn't flicker against it.
  gl_Position = u_viewProj * vec4(a_pos - vec3(0.0, 0.3, 0.0), 1.0);
  v_world = a_pos;
}
`;

export const glFarFragment = /* glsl */ `${head}
uniform vec4 u_near;  // x, z of the coverage map's first chunk (in blocks), its size in chunks
uniform vec4 u_sea;   // x = y of the sea's surface, y = look (0 colours, 1 silhouette, 2 mist), z = how far it reaches
uniform sampler2D u_coverage;
in vec3 v_world;
out vec4 o_colour;

float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

void main() {
  vec3 normal = normalize(cross(dFdx(v_world), dFdy(v_world)));
  float span = max(length(dFdx(v_world.xz)), length(dFdy(v_world.xz))) / 4.0;
  float blur = clamp((span - 0.6) / 1.4, 0.0, 1.0);
  if (normal.y < 0.0) normal = -normal;
  ivec2 c = ivec2(floor((v_world.xz - u_near.xy) / 16.0));
  if (all(greaterThanEqual(c, ivec2(0))) && all(lessThan(c, ivec2(int(u_near.z)))) && texelFetch(u_coverage, c, 0).r > 0.5) discard;

  float dist = distance(v_world, u_camPos.xyz);
  vec3 sun = normalize(vec3(0.4, 0.85, 0.3));
  if (u_sea.y > 0.5 && u_sea.y < 1.5) {
    float haze = 0.75 * pow(clamp(dist / u_sky.w, 0.0, 1.0), 0.6);
    float shade = 0.9 + 0.1 * normal.y;
    o_colour = vec4(mix(vec3(0.04, 0.05, 0.08) * shade, u_sky.rgb, haze), 1.0);
    return;
  }
  float n = mix(hash2(floor(v_world.xz / 4.0)), 0.5, blur);
  vec3 base;
  if (v_world.y <= u_sea.x + 0.01) {
    base = vec3(0.2, 0.38, 0.74);
    normal = vec3(0.0, 1.0, 0.0);
  } else {
    vec3 green = vec3(0.36, 0.62, 0.22) * (0.85 + 0.2 * n);
    vec3 dirt = vec3(0.55, 0.38, 0.24) * (0.85 + 0.2 * n);
    base = mix(dirt, green, smoothstep(0.55, 0.8, normal.y));
    vec3 sand = vec3(0.86, 0.80, 0.58) * (0.85 + 0.2 * n);
    base = mix(base, sand, 1.0 - smoothstep(${f(SEA_LEVEL + SAND_ABOVE + 1)}, ${f(SEA_LEVEL + SAND_ABOVE + 1.6)}, v_world.y));
  }
  vec3 lit = base * (0.6 + 0.4 * max(dot(normal, sun), 0.0));
  float fog = clamp((dist - u_sky.w * 0.6) / (u_sky.w * 0.4), 0.0, u_fogCap.x);
  if (u_sea.y > 1.5) {
    float mist = mix(u_fogCap.x, 1.0, sqrt(clamp(dist / u_sea.z, 0.0, 1.0)));
    fog = max(fog, mist * fog / u_fogCap.x);
  }
  o_colour = vec4(mix(lit, u_sky.rgb, fog), 1.0);
}
`;

export const glMobVertex = /* glsl */ `${head}
// The model's animation (render/mobModel.ts): per frame (row) and part, 3 texels, the rows of
// the part's matrix into the model's space.
uniform sampler2D u_poses;
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in float a_part;
layout(location = 4) in vec4 a_place;   // feet position, yaw (per mob)
layout(location = 5) in vec4 a_frames;  // the two frames it's between, and how far (per mob)
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;

vec4 poseRow(int part, int r) {
  vec4 a = texelFetch(u_poses, ivec2(part * 3 + r, int(a_frames.x)), 0);
  vec4 b = texelFetch(u_poses, ivec2(part * 3 + r, int(a_frames.y)), 0);
  return mix(a, b, a_frames.z);
}

void main() {
  int part = int(a_part + 0.5);
  vec4 r0 = poseRow(part, 0), r1 = poseRow(part, 1), r2 = poseRow(part, 2);
  vec4 p = vec4(a_pos, 1.0);
  vec3 posed = vec3(dot(r0, p), dot(r1, p), dot(r2, p));
  vec3 n = vec3(dot(r0.xyz, a_normal), dot(r1.xyz, a_normal), dot(r2.xyz, a_normal));
  // Yaw: the model faces -z, turned like the camera (forward = (-sin yaw, 0, -cos yaw)).
  float c = cos(a_place.w), s = sin(a_place.w);
  vec3 world = vec3(posed.x * c + posed.z * s, posed.y, -posed.x * s + posed.z * c) + a_place.xyz;
  gl_Position = u_viewProj * vec4(world, 1.0);
  v_world = world;
  v_normal = vec3(n.x * c + n.z * s, n.y, -n.x * s + n.z * c);
  v_uv = a_uv;
}
`;

export const glMobFragment = /* glsl */ `${head}
uniform sampler2D u_skin;
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
out vec4 o_colour;
void main() {
  vec3 base = texture(u_skin, v_uv).rgb;
  vec3 normal = normalize(gl_FrontFacing ? v_normal : -v_normal);
  vec3 sun = normalize(vec3(0.4, 0.85, 0.3));
  vec3 lit = base * (0.6 + 0.4 * max(dot(normal, sun), 0.0));
  float dist = distance(v_world, u_camPos.xyz);
  float fog = clamp((dist - u_sky.w * 0.6) / (u_sky.w * 0.4), 0.0, u_fogCap.x);
  o_colour = vec4(mix(lit, u_sky.rgb, fog), 1.0);
}
`;

export const glLineVertex = /* glsl */ `${head}
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_colour;
out vec3 v_colour;
void main() {
  gl_Position = u_viewProj * vec4(a_pos, 1.0);
  v_colour = a_colour;
}
`;

export const glLineFragment = /* glsl */ `${head}
in vec3 v_colour;
out vec4 o_colour;
void main() { o_colour = vec4(v_colour, 1.0); }
`;
