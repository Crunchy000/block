import { describe, expect, it } from 'vitest';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, SOURCE_LEVEL, WHEAT_RIPE, blockIndex, cell } from '../src/constants';
import {
  FULL_HEIGHT, Face, faceQuad, fluidHeight, heightCode, meshFaces, packFace, unpackFace, wheatHeight,
} from '../src/render/mesher';

/** A 3x3-chunk area of flat stone up to y = 9; meshes the middle chunk. */
function flatArea() {
  const S = CHUNK_SIZE, W = 3 * S;
  const cells = new Int32Array(CHUNK_HEIGHT * W * W);
  for (let y = 0; y < 10; y++) for (let z = 0; z < W; z++) for (let x = 0; x < W; x++) cells[(y * W + z) * W + x] = cell(Block.Stone);
  return {
    // Chunk-local coordinates of the middle chunk.
    set: (x: number, y: number, z: number, c: number) => { cells[(y * W + z + S) * W + x + S] = c; },
    mesh: () => meshFaces((x, y, z) => cells[(y * W + z + S) * W + x + S]),
  };
}

const quads = (records: number[]) => records.map((r) => faceQuad(r, 0, 0));

describe('meshFaces', () => {
  it('emits only the top faces of a flat chunk (borders culled against neighbours)', () => {
    const { opaque, water } = flatArea().mesh();
    expect(opaque).toHaveLength(CHUNK_SIZE * CHUNK_SIZE);
    expect(opaque.every((r) => unpackFace(r).face === Face.Top && unpackFace(r).y === 9)).toBe(true);
    expect(water).toHaveLength(0);
  });

  it('keeps solid walls next to shallow lava (no see-through holes)', () => {
    const area = flatArea();
    area.set(5, 9, 5, cell(Block.Lava, 2)); // a 1-deep pit with shallow flowing lava in it
    const faces = area.mesh().opaque.map(unpackFace);
    // Lava doesn't hide solids, so all four pit walls are still there.
    expect(faces.filter((f) => f.type === Block.Stone && f.face !== Face.Top && f.face !== Face.Bottom)).toHaveLength(4);
    expect(faces.filter((f) => f.type === Block.Lava).map((f) => f.face)).toEqual([Face.Top]); // the walls hide its sides
  });

  it('puts water in the translucent mesh with a lowered surface', () => {
    const area = flatArea();
    area.set(3, 10, 3, cell(Block.Water, SOURCE_LEVEL));
    const { water } = area.mesh();
    expect(water).toHaveLength(5); // top + 4 sides (bottom rests on stone)
    const maxY = Math.max(...quads(water).flatMap((q) => q.corners.map((c) => c[1])));
    expect(maxY).toBeCloseTo(10.875);
  });

  it('fills a fluid block to the top when the same fluid is above it, and hides faces between equal levels', () => {
    const area = flatArea();
    area.set(3, 10, 3, cell(Block.Water, SOURCE_LEVEL));
    area.set(3, 11, 3, cell(Block.Water, SOURCE_LEVEL));
    area.set(4, 10, 3, cell(Block.Water, SOURCE_LEVEL));
    const faces = area.mesh().water.map(unpackFace);
    const lower = faces.filter((f) => f.y === 10 && f.x === 3);
    expect(lower.every((f) => f.aux === FULL_HEIGHT)).toBe(true);
    // The full block shows its east side above its lower neighbour's surface; the
    // neighbour doesn't show its west side against it.
    expect(lower.some((f) => f.face === Face.East)).toBe(true);
    expect(faces.some((f) => f.x === 4 && f.face === Face.West)).toBe(false);
  });

  it('draws wheat as four plant quads whose height grows with its stage', () => {
    const area = flatArea();
    area.set(2, 10, 2, cell(Block.Wheat, 3));
    const plants = area.mesh().opaque.filter((r) => unpackFace(r).type === Block.Wheat);
    expect(plants.map((r) => unpackFace(r).face).sort()).toEqual([Face.PlantA, Face.PlantABack, Face.PlantB, Face.PlantBBack]);
    for (const q of quads(plants)) {
      expect(Math.max(...q.corners.map((c) => c[1]))).toBeCloseTo(10 + wheatHeight(3));
      expect(q.kind).toBe(Block.Wheat + 16 * 3);
    }
    // Front and back of each plane wind opposite ways (both are seen with back-face culling on).
    const [a, back] = quads(plants.filter((r) => unpackFace(r).face <= Face.PlantABack));
    expect(back.corners).toEqual([a.corners[1], a.corners[0], a.corners[3], a.corners[2]]);
  });
});

describe('face records', () => {
  it('round-trip through packing', () => {
    const r = packFace(15, 63, 7, Face.PlantBBack, Block.Wheat, WHEAT_RIPE);
    expect(unpackFace(r)).toEqual({ x: 15, y: 63, z: 7, face: Face.PlantBBack, type: Block.Wheat, aux: WHEAT_RIPE });
    expect(r).toBeLessThan(2 ** 25);
  });

  it('order fluid surfaces by height code', () => {
    const codes = [1, 2, 3, 4, 5, 6, 7, SOURCE_LEVEL, FULL_HEIGHT];
    const heights = codes.map(fluidHeight);
    expect([...heights].sort((a, b) => a - b)).toEqual(heights);
    expect(new Set(heights).size).toBe(codes.length);
    expect(heightCode(cell(Block.Water, 0), cell(Block.Air))).toBe(1);
    expect(heightCode(cell(Block.Lava, 3), cell(Block.Lava, 8))).toBe(FULL_HEIGHT);
  });

  it('place the corners of side faces on the block (one block = one unit)', () => {
    const q = faceQuad(packFace(1, 2, 3, Face.East, Block.Stone, 0), 32, -16);
    expect(q.normal).toEqual([1, 0, 0]);
    expect(q.corners).toEqual([[34, 2, -13], [34, 3, -13], [34, 3, -12], [34, 2, -12]]);
    expect(blockIndex(1, 2, 3)).toBe((2 * CHUNK_SIZE + 3) * CHUNK_SIZE + 1);
  });
});
