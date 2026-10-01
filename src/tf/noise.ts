import * as tf from '@tensorflow/tfjs';

// Value noise built from elementwise tensor ops, so every sample in a batch
// of chunks is evaluated in parallel on the TF backend.

/** Classic shader hash: fract(sin(dot(p, k)) * 43758.5453) in [0, 1). */
function hash(...coords: tf.Tensor[]): tf.Tensor {
  const k = [127.1, 311.7, 74.7];
  let dot = coords[0].mul(k[0]);
  for (let i = 1; i < coords.length; i++) dot = dot.add(coords[i].mul(k[i]));
  const s = dot.sin().mul(43758.5453);
  return s.sub(s.floor());
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
