import * as tf from '@tensorflow/tfjs';

/**
 * Uniform [0, 1) noise, one value per cell of a [N, H, D, W] batch, computed on the GPU.
 *
 * tf.randomUniform fills its tensor in JavaScript on the main thread (and then
 * uploads it), which for a million cells per tick is exactly the kind of work the
 * block updates must avoid. Instead this hashes each cell's coordinates with
 * Dave Hoskins' sine-free hash13, offset by three per-call seeds. Per-axis terms
 * are computed at their own small shapes and only broadcast to full size inside
 * the final few element-wise ops. The batch index is folded into z so every chunk
 * in the batch gets different numbers.
 */
export function randomField(shape: readonly [number, number, number, number], seeds: readonly [number, number, number]): tf.Tensor4D {
  return tf.tidy(() => {
    const [N, H, D, W] = shape;
    const fract = (t: tf.Tensor) => t.sub(t.floor());
    // p = fract(p * .1031)
    const px = fract(tf.range(0, W).add(seeds[0]).mul(0.1031)).reshape([1, 1, 1, W]);
    const py = fract(tf.range(0, H).add(seeds[1]).mul(0.1031)).reshape([1, H, 1, 1]);
    const pz = fract(tf.range(0, N * D).add(seeds[2]).mul(0.1031)).reshape([N, 1, D, 1]);
    // p += dot(p, p.zyx + 31.32)
    const d = px.mul(pz.add(31.32)).add(pz.mul(px.add(31.32))).add(py.mul(py.add(31.32)));
    // fract((p.x + p.y) * p.z)
    return fract(px.add(py).add(d.mul(2)).mul(pz.add(d))) as tf.Tensor4D;
  });
}
