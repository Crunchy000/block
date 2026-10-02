import { describe, expect, it } from 'vitest';
import { mipLevels } from '../src/render/blockTextures';

describe('mipLevels', () => {
  it('halves each level down to 1 x 1, per layer', () => {
    const data = new Uint8Array(4 * 4 * 4 * 2);
    for (let i = 0; i < 16; i++) data.set([200, 100, 0, 255], i * 4); // layer 0: one colour
    for (let i = 16; i < 32; i++) data.set([i % 2 ? 0 : 255, 0, 0, 255], i * 4); // layer 1: stripes
    const levels = mipLevels(data, 4, 2);
    expect(levels.map((l) => l.length)).toEqual([128, 32, 8]);
    expect([...levels[2].subarray(0, 4)]).toEqual([200, 100, 0, 255]);
    expect([...levels[2].subarray(4, 8)]).toEqual([128, 0, 0, 255]);
  });

  it("weights colour by alpha, so see-through texels don't darken", () => {
    const data = new Uint8Array(2 * 2 * 4);
    data.set([0, 200, 0, 255], 0); // one green texel, the rest clear black
    const [, one] = mipLevels(data, 2, 1);
    expect([...one]).toEqual([0, 200, 0, 64]);
  });
});
