import { faceQuad } from './mesher';

/** Floats per vertex: position (3), normal (3), kind (1). */
export const VERTEX_FLOATS = 7;

/** Face records as plain vertices (VERTEX_FLOATS each, 4 a face) and indices (6 a face): the WebGL2 renderer's (safe mode's) meshes. */
export function classicMeshData(records: number[], cx: number, cz: number): { vertices: Float32Array<ArrayBuffer>; indices: Uint32Array<ArrayBuffer> } {
  const vertices = new Float32Array(records.length * 4 * VERTEX_FLOATS);
  const indices = new Uint32Array(records.length * 6);
  records.forEach((record, q) => {
    const { corners, normal, kind } = faceQuad(record, cx * 16, cz * 16);
    corners.forEach((c, k) => vertices.set([c[0], c[1], c[2], normal[0], normal[1], normal[2], kind], (q * 4 + k) * VERTEX_FLOATS));
    const v = q * 4;
    indices.set([v, v + 1, v + 2, v, v + 2, v + 3], q * 6);
  });
  return { vertices, indices };
}
