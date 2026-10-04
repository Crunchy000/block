import type { MeshTarget } from '../sim/store';
import type { ChunkDraw } from '../world/world';
import type { BlockTextureData } from './blockTextures';
import type { Clip, MobModelData } from './mobModel';

export const SKY: [number, number, number] = [0.55, 0.75, 0.95];
/** In mist, fog stops this thick: chunks past the fog keep a trace of themselves, and the far terrain carries on from there. */
export const MIST_FOG = 0.85;

/** The far terrain to draw (world/farTerrain.ts): an indexed triangle list of positions. */
export interface FarDraw {
  /** Positions (x, y, z), rewritten as the player travels: re-uploaded when `version` changes. */
  vertices: Float32Array<ArrayBuffer>;
  indices: Uint32Array<ArrayBuffer>;
  version: number;
  /**
   * Which chunks are drawn and fully faded in (world.coverage): the far terrain isn't drawn
   * there. Chunks still loading or fading in keep it underneath, so there are no holes.
   */
  coverage: { x0: number; z0: number; size: number; data: Uint8Array<ArrayBuffer> };
  /** The y of the sea's surface. */
  seaY: number;
  /** How it looks: mist (a little darker than the fog), a dark silhouette, or colours. */
  look: 'mist' | 'silhouette' | 'colour';
  /** How far it reaches (blocks). */
  extent: number;
}

/** A mob model a renderer has uploaded: what the game needs to animate it (render/mobModel.ts poseFrames). */
export interface MobModelHandle {
  clips: Record<string, Clip>;
  fps: number;
}

/**
 * Mobs of one kind to draw: a model (from the same renderer's createMobModel), and per mob 8
 * floats: x, y, z, yaw, then the two frames of the model's poses it's between and how far
 * (render/mobModel.ts poseFrames), and 0.
 */
export interface MobDraw {
  model: MobModelHandle;
  instances: Float32Array<ArrayBuffer>;
}

/**
 * What the game draws with: WebGPU (render/renderer.ts) or, where there's no WebGPU, WebGL2
 * (render/glRenderer.ts). Meshes and models come from the renderer that draws them.
 */
export interface GameRenderer {
  /** 'WebGPU' or 'WebGL2'. */
  readonly api: string;
  /** The GPU's name, as far as the browser says. */
  readonly gpuName: string;
  readonly aspect: number;
  /** Resolves if the GPU is lost (a crash or reset, or the browser reclaiming it): nothing more can be drawn. */
  readonly lost: Promise<{ reason: string; message: string }>;
  /** Somewhere to put chunk meshes, `slots` chunks of them. */
  createMeshes(slots: number): MeshTarget;
  /** Texture the blocks with these layers (render/blockTextures.ts) from now on. */
  setBlockTextures(data: BlockTextureData): void;
  createMobModel(data: MobModelData): MobModelHandle;
  /**
   * Draw the chunks in `draws` from their meshes, plus lines: interleaved [x, y, z, r, g, b]
   * pairs for a line list, the far terrain beyond them if given, and the mobs.
   */
  render(
    viewProj: Float32Array, cam: readonly number[], time: number, fogDistance: number, lines: Float32Array<ArrayBuffer>,
    meshes: MeshTarget, draws: ChunkDraw[], far?: FarDraw, mobs?: MobDraw[],
  ): void;
}
