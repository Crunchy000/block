import * as tf from '@tensorflow/tfjs';

// Value noise built from elementwise tensor ops, so every sample in a batch
// of chunks is evaluated in parallel on the TF backend.

/**
 * Hash of integer lattice coordinates in [0, 1): the permutation polynomial (34v + 1)v mod 289
 * (as in Ashima's simplex noise), chained over the coordinates. Every intermediate value is an
 * integer below 2^24, exact in float32, so the TF.js backends, the worldgen shader
 * (sim/gpuWorldgen.ts) and the plain-JS generator (world/jsWorldgen.ts) make the same world on
 * every device (a sine hash amplifies each platform's rounding into different values). The
 * noise repeats every 289 lattice cells along each axis.
 */
const HASH_P = 289;
/**
 * v mod 289 for integer v (|v| up to a few million): floor((v + 0.5) / 289) stays clear of
 * rounding either way, even with a GPU's less exact division.
 */
const mod289 = (v: tf.Tensor) => v.sub(v.add(0.5).div(HASH_P).floor().mul(HASH_P));
/** (34v + 1)v mod 289 for v in [0, 289), reduced on the way so every value stays below 84,000. */
const permute = (v: tf.Tensor) => mod289(mod289(v.mul(34).add(1)).mul(v));
export function hash(...coords: tf.Tensor[]): tf.Tensor {
  let h = permute(mod289(coords[0]));
  for (let i = 1; i < coords.length; i++) h = permute(mod289(h.add(mod289(coords[i]))));
  return h.div(HASH_P);
}

/**
 * Sine-free 2D hash (Dave Hoskins' hash12) in [0, 1), for per-column decisions.
 * The sine hash above loses most of its precision at large coordinates in float32,
 * so coordinates are wrapped to [0, 4096) first. The result still only resolves
 * about 1/512, so combine independent hashes for rarer events.
 */
export function hash12(x: tf.Tensor, z: tf.Tensor, seed: number): tf.Tensor {
  return tf.tidy(() => {
    const wrap = (t: tf.Tensor) => t.sub(t.div(4096).floor().mul(4096));
    const fract = (t: tf.Tensor) => t.sub(t.floor());
    // p3 = fract(p.xyx * .1031), written as (a, b, a)
    const a = fract(wrap(x).add(seed).mul(0.1031)), b = fract(wrap(z).add(seed * 0.618).mul(0.1031));
    // p3 += dot(p3, p3.yzx + 33.33)
    const d = a.mul(b.add(33.33)).add(b.mul(a.add(33.33))).add(a.mul(a.add(33.33)));
    // fract((p3.x + p3.y) * p3.z)
    return fract(a.add(b).add(d.mul(2)).mul(a.add(d)));
  });
}

const smooth = (t: tf.Tensor) => t.mul(t).mul(t.mul(-2).add(3));
const lerp = (a: tf.Tensor, b: tf.Tensor, t: tf.Tensor) => a.add(b.sub(a).mul(t));

/** 2D value noise in [0, 1). x, z: float tensors of equal shape. */
export function valueNoise2(x: tf.Tensor, z: tf.Tensor, seed: number): tf.Tensor {
  return tf.tidy(() => {
    const xi = x.floor(), zi = z.floor().add(seed);
    const u = smooth(x.sub(x.floor())), v = smooth(z.sub(z.floor()));
    const xi1 = xi.add(1), zi1 = zi.add(1);
    const a = lerp(hash(xi, zi), hash(xi1, zi), u);
    const b = lerp(hash(xi, zi1), hash(xi1, zi1), u);
    return lerp(a, b, v);
  });
}

/** 3D value noise in [0, 1). */
export function valueNoise3(x: tf.Tensor, y: tf.Tensor, z: tf.Tensor, seed: number): tf.Tensor {
  return tf.tidy(() => {
    const xi = x.floor(), yi = y.floor(), zi = z.floor().add(seed);
    const u = smooth(x.sub(xi)), v = smooth(y.sub(yi)), w = smooth(z.sub(z.floor()));
    const xi1 = xi.add(1), yi1 = yi.add(1), zi1 = zi.add(1);
    const plane = (zz: tf.Tensor) =>
      lerp(lerp(hash(xi, yi, zz), hash(xi1, yi, zz), u), lerp(hash(xi, yi1, zz), hash(xi1, yi1, zz), u), v);
    return lerp(plane(zi), plane(zi1), w);
  });
}

/** Fractal (fBm) 2D noise normalised to roughly [0, 1). */
export function fbm2(x: tf.Tensor, z: tf.Tensor, seed: number, baseScale: number, octaves: number): tf.Tensor {
  return tf.tidy(() => {
    let sum: tf.Tensor = tf.zerosLike(x);
    let amp = 1, norm = 0, scale = baseScale;
    for (let o = 0; o < octaves; o++) {
      sum = sum.add(valueNoise2(x.div(scale), z.div(scale), seed + o * 101).mul(amp));
      norm += amp;
      amp *= 0.5;
      scale *= 0.5;
    }
    return sum.div(norm);
  });
}
