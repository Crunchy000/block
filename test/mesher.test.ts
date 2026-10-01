import { describe, expect, it } from 'vitest';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, SOURCE_LEVEL, blockIndex, cell } from '../src/constants';
import { VERTEX_FLOATS, meshChunk } from '../src/render/mesher';
import { World } from '../src/world/world';

/** 3x3 chunks of flat stone up to y = 9, centred on chunk 0,0. */
function flatWorld(): World {
  const world = new World(0, 1);
  for (let cz = -1; cz <= 1; cz++) {
    for (let cx = -1; cx <= 1; cx++) {
      const data = new Uint8Array(CHUNK_SIZE * CHUNK_SIZE * CHUNK_HEIGHT);
      for (let y = 0; y < 10; y++)
        for (let z = 0; z < CHUNK_SIZE; z++)
          for (let x = 0; x < CHUNK_SIZE; x++) data[blockIndex(x, y, z)] = cell(Block.Stone);
      world.addGenerated({ cx, cz }, data);
    }
  }
  return world;
}

const faceCount = (indices: Uint32Array) => indices.length / 6;

describe('meshChunk', () => {
  it('emits only the top faces of a flat chunk (borders culled against neighbours)', () => {
    const mesh = meshChunk(flatWorld(), 0, 0);
    expect(faceCount(mesh.opaque.indices)).toBe(CHUNK_SIZE * CHUNK_SIZE);
    expect(mesh.opaque.vertices.length).toBe(CHUNK_SIZE * CHUNK_SIZE * 4 * VERTEX_FLOATS);
    expect(faceCount(mesh.water.indices)).toBe(0);
  });

  it('keeps solid walls next to shallow lava (no see-through holes)', () => {
    const world = flatWorld();
    // 1-deep pit with a shallow flowing lava cell in it.
    world.setCell(5, 9, 5, cell(Block.Lava, 2));
    const mesh = meshChunk(world, 0, 0);
    // Lava doesn't occlude solids, so all 4 pit walls must still be meshed.
    const v = mesh.opaque.vertices;
    let walls = 0;
    for (let i = 0; i < v.length; i += VERTEX_FLOATS * 4) {
      const ny = v[i + 4], type = v[i + 6];
      if (type === Block.Stone && ny === 0) walls++;
    }
    expect(walls).toBe(4);
  });

  it('puts water in the translucent mesh with a lowered surface', () => {
    const world = flatWorld();
    world.setCell(3, 10, 3, cell(Block.Water, SOURCE_LEVEL));
    const mesh = meshChunk(world, 0, 0);
    expect(faceCount(mesh.water.indices)).toBe(5); // top + 4 sides (bottom rests on stone)
    let maxY = 0;
    for (let i = 1; i < mesh.water.vertices.length; i += VERTEX_FLOATS) maxY = Math.max(maxY, mesh.water.vertices[i]);
    expect(maxY).toBeCloseTo(10.875);
  });
});
