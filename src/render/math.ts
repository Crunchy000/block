export type Mat4 = Float32Array;

/**
 * Column-major perspective matrix with WebGPU's [0, 1] clip-space depth, reversed and with
 * no far plane: depth = near / distance, 1 at the near plane falling towards 0 far away.
 * With a float depth buffer that keeps precision out to any view distance (draw with depth
 * compare 'greater', clear to 0).
 */
export function perspective(fovY: number, aspect: number, near: number): Mat4 {
  const f = 1 / Math.tan(fovY / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, 0, -1,
    0, 0, near, 0,
  ]);
}

/**
 * A test for boxes against the sides of the view (left, right, top and bottom planes of
 * a view-projection matrix): false when a box is wholly outside.
 */
export function frustum(viewProj: Mat4): (min: readonly number[], max: readonly number[]) => boolean {
  const row = (i: number) => [viewProj[i], viewProj[4 + i], viewProj[8 + i], viewProj[12 + i]];
  const r0 = row(0), r1 = row(1), r3 = row(3);
  const planes = [[1, r0], [-1, r0], [1, r1], [-1, r1]].map(([s, r]) =>
    (r3 as number[]).map((v, k) => v + (s as number) * (r as number[])[k]));
  return (min, max) => planes.every((p) =>
    p[0] * (p[0] > 0 ? max[0] : min[0]) + p[1] * (p[1] > 0 ? max[1] : min[1]) + p[2] * (p[2] > 0 ? max[2] : min[2]) + p[3] >= 0);
}

/** View matrix for a first-person camera at `eye` with yaw (around Y) and pitch. */
export function fpsView(eye: readonly number[], yaw: number, pitch: number): Mat4 {
  const cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x = [cy, 0, -sy];
  const y = [sy * sp, cp, cy * sp];
  const z = [sy * cp, -sp, cp * cy];
  const dot = (a: number[]) => a[0] * eye[0] + a[1] * eye[1] + a[2] * eye[2];
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x), -dot(y), -dot(z), 1,
  ]);
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  return out;
}

/** Forward direction for a yaw/pitch camera (matches fpsView: looks down -Z at yaw 0). */
export function forward(yaw: number, pitch: number): [number, number, number] {
  return [-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch)];
}
