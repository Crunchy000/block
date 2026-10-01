# Block

A Minecraft-style voxel world where **world generation and block updates run in
TensorFlow.js** and rendering is a hand-written **WebGPU** renderer.

```
npm install
npm run dev        # http://localhost:5173 — needs a WebGPU browser (Chrome/Edge 113+, Safari 26+, Firefox 141+)
npm test           # TF.js logic on the CPU backend (worldgen, block updates, meshing, halo)
npm run build
```

**Keyboard & mouse:** click anywhere on the start screen to grab the mouse (Esc releases it) · WASD move ·
Space / Shift up/down · Ctrl sprint · left click break · right click place ·
`1` dirt `2` stone `3` water `4` lava · `G` chunk / ghost-halo outlines · `P` pause block updates.

**Touch (phones, tablets):** tap the start screen for on-screen controls. The left half of the screen
is a **dynamic stick**: it appears wherever your thumb lands, is analog (push further to go faster),
and disappears when you let go; double-tap and hold it to sprint. Drag on the right half to look around.
Buttons: Break (hold to keep breaking), Place, ▲ / ▼ to fly, a hotbar to pick the block, and
Outlines / Pause / Menu at the top right. You can move and look at the same time with two thumbs.

The bottom layer (y = 0) is unbreakable bedrock.

Every push is built and tested by `.github/workflows/pages.yml`; pushes to the default branch are
published to GitHub Pages at `https://<owner>.github.io/<repo>/`.

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
| `src/tf/blockUpdateReference.ts` | The same rules written cell by cell in plain JS: the readable spec, and the oracle the tests compare the TF.js step against |
| `src/tf/simulation.ts` | Packs the awake chunks + a ghost border into one tensor, steps it, writes back the interior only |
| `src/tf/backend.ts` | Runs TF.js's WebGPU backend **on the renderer's `GPUDevice`**; falls back to WebGL, then CPU; pre-compiles kernels |
| `src/world/` | Chunk store, active area / ghost halo tracking, awake (sleeping) chunks, queued edits |
| `src/render/` | Face-culling mesher, WGSL shaders, WebGPU renderer (opaque pass, line pass, translucent water pass) |
| `src/player/` | Fly camera + pointer lock, touch controls (dynamic stick, look drag, buttons), voxel DDA ray picking |
| `src/ui/hotbar.ts` | Block picker (keys 1–4 or tap) |

## Chunks and the ghost halo

Chunks are 16×16 columns, 64 blocks tall. Around the player's chunk:

- **Active chunks** (Chebyshev distance ≤ `ACTIVE_RADIUS`, default 3 → 7×7): simulated and rendered.
- **Ghost chunks**: the ring one chunk further out (→ 32 chunks). These are generated and held
  in memory but never stepped or drawn. They are the halo (ghost cells) for the stencil
  computations:
  - block updates for chunks at the edge of the active area read their neighbours from the
    ghost ring, so fluids there see real terrain; ghost results are thrown away, so ghost
    chunks stay read-only;
  - the mesher reads ghost data to cull faces on the outer border of the active area.

When the player crosses a chunk border, the window re-centres: ghosts that come within range
become active (they get meshed and simulated), active chunks that drop out become ghosts, and
chunks that leave the halo are dropped. Edited chunks are kept, so changes persist.
Press `G` to see active (green) and ghost (orange) outlines.

## Sleeping chunks

Most of the world is static most of the time, so block updates only run where something can change.
A chunk is **awake** after an edit, when it's freshly generated or becomes active, or when a
neighbouring chunk changed along their shared border. Each tick (5 per second) simulates the bounding
box of the awake chunks plus a one-chunk ghost border, then writes back only the interior.
A chunk that didn't change, with neighbours that didn't change, is a fixed point of the rules, so it
goes back to sleep. With nothing changing, no tick runs at all.
The HUD shows whether updates are asleep and how big the last region was.

## Block update rules

All cells in the region update in parallel from the previous state:

- Sources have level 8; flowing fluid has level 1–7 and is recomputed every tick, so it drains when cut off.
- Fluid above an air or flowing cell falls into it (level 7).
- Fluid resting on solid ground or on a source spreads sideways at `level − decay`
  (water decays by 1, lava by 2, so lava spreads less far).
- Lava with water beside or above it turns to stone; a cell that both fluids would flow into becomes stone.

## TF.js performance notes

- **Never pass a scalar to `tf.where`, and avoid `broadcastTo`.** TF.js expands the small tensor to full
  size with its `Tile` kernel, and for small CPU-side inputs that runs in JavaScript on the main thread,
  even on the GPU backends. Use `tf.fill` for a full-size constant, or let binary ops broadcast inside the
  kernel. A test fails if `Tile` shows up in world generation or block updates.
- **Use int32 scalars with int32 tensors** (`tf.scalar(v, 'int32')`): `intTensor.sub(1)` silently
  produces float32.
- **Kernels are compiled up front.** Shaders are cached by tensor rank and dtype, not size, so
  `warmUpKernels()` compiles everything once at startup in TF.js's compile-only mode (in parallel,
  off the main thread) instead of stalling the first frames that use each kernel.
