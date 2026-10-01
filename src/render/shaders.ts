const uniforms = /* wgsl */ `
struct Uniforms {
  viewProj: mat4x4f,
  camPos: vec4f,   // xyz = camera, w = time (s)
  sky: vec4f,      // rgb = sky / fog colour, w = fog distance
};
@group(0) @binding(0) var<uniform> u: Uniforms;
`;

export const blockShader = /* wgsl */ `
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
    default: { base = vec3f(1.0, 0.0, 1.0); }
  }

  let sun = normalize(vec3f(0.4, 0.85, 0.3));
  let diffuse = max(dot(in.normal, sun), 0.0);
  let side = 0.8 + 0.2 * abs(in.normal.y) + 0.08 * abs(in.normal.x);
  var lit = base * mix((0.6 + 0.4 * diffuse) * side, 1.0, emissive);

  let dist = distance(in.world, u.camPos.xyz);
  let fog = clamp((dist - u.sky.w * 0.6) / (u.sky.w * 0.4), 0.0, 1.0);
  lit = mix(lit, u.sky.rgb, fog);
  return vec4f(lit, alpha);
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
