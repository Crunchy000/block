# Block

A Minecraft-style voxel world where **world generation and block updates run in
TensorFlow.js** and rendering is a hand-written **WebGPU** renderer.

```
npm install
npm run dev        # http://localhost:5173 — needs a WebGPU browser (Chrome/Edge 113+, Safari 26+, Firefox 141+)
npm test           # TF.js logic on the CPU backend (worldgen, block updates, meshing, halo)
npm run build
```

Controls: click to grab the mouse · WASD move · Space / Shift up/down · Ctrl sprint ·
left click break · right click place · `1` dirt `2` stone `3` water `4` lava ·
`G` chunk / ghost-halo outlines · `P` pause block updates.

URL params: `?radius=N` active radius in chunks (default 3) · `pos=x,y,z` · `yaw=` / `pitch=` (radians) ·
`chunks` (outlines on) · `offscreen` (render to a texture and copy it to a 2D canvas, for
headless browsers where WebGPU canvas presentation isn't available).

## Layout

| Path | What |
| --- | --- |
| `src/constants.ts` | Chunk dims (16×16×64), block ids, cell encoding (`type + 8 * fluidLevel`) |
| `src/tf/worldgen.ts` | Batched chunk generation as one TF graph: fBm heightmap → stone/dirt, sea-level water, 3D-noise caves, deep lava lakes |
| `src/tf/noise.ts` | Value noise / fBm built from elementwise tensor ops |
| `src/tf/blockUpdate.ts` | One block-update tick as a cellular automaton over an `[H, Z, X]` int32 tensor |
| `src/tf/simulation.ts` | Packs active chunks + ghost halo into one tensor, steps it, writes back active chunks only |
| `src/tf/backend.ts` | Runs TF.js's WebGPU backend **on the renderer's `GPUDevice`**; falls back to WebGL, then CPU |
| `src/world/` | Chunk store, active area / ghost halo tracking, queued edits |
| `src/render/` | Face-culling mesher, WGSL shaders, WebGPU renderer (opaque pass, line pass, translucent water pass) |
| `src/player/` | Fly camera + pointer lock, voxel DDA ray picking |

## Chunks and the ghost halo

Chunks are 16×16 columns, 64 blocks tall. Around the player's chunk:

- **Active chunks** (Chebyshev distance ≤ `ACTIVE_RADIUS`, default 3 → 7×7): simulated and rendered.
- **Ghost chunks**: the ring one chunk further out (→ 32 chunks). These are generated and held
  in memory but never stepped or drawn. They are the halo (ghost cells) for the stencil
  computations:
  - the block-update tick runs over active + ghost as one tensor, so fluids at the edge of the
    active area see real neighbours; the ghost results are thrown away, so ghost chunks stay read-only;
  - the mesher reads ghost data to cull faces on the outer border of the active area.

When the player crosses a chunk border, the window re-centres: ghosts that come within range
become active (they get meshed and simulated), active chunks that drop out become ghosts, and
chunks that leave the halo are dropped. Edited chunks are kept, so changes persist.
Press `G` to see active (green) and ghost (orange) outlines.

## Block update rules

All cells update in parallel from the previous state (5 ticks/s):

- Sources have level 8; flowing fluid has level 1–7 and is recomputed every tick, so it drains when cut off.
- Fluid above an air or flowing cell falls into it (level 7).
- Fluid resting on solid ground or on a source spreads sideways at `level − decay`
  (water decays by 1, lava by 2, so lava spreads less far).
- Lava touching water turns to stone; a cell that both fluids would flow into becomes stone.
