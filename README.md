# Block

A Minecraft-style voxel world: **world generation is written in TensorFlow.js** (and run as one
compute shader ported from it), and the world **lives in GPU memory**, where generation, block
updates, meshing and picking run as WebGPU compute shaders
and a hand-written **WebGPU** renderer draws it. The CPU keeps only bookkeeping (which chunks are
where, which are awake); per tick it gets back a few bytes of flags.

```
npm install
npm run dev        # http://localhost:5173 — needs a WebGPU browser (Chrome/Edge 113+, Safari 26+, Firefox 141+)
npm test           # the world on the CPU reference store, rules, meshing, worldgen (Node, TF.js CPU backend)
npm run test:gpu   # everything on the GPU vs the reference, in headless Chromium with WebGPU (Playwright)
npm run build
```

**Keyboard & mouse:** click anywhere on the start screen to grab the mouse (Esc releases it) · WASD walk ·
Space jump (and swim up) · double-tap W, or hold Ctrl, to run · `F` fly (through blocks; Space / Shift up/down,
Ctrl to fly faster) · left click break · right click place ·
`1`–`8` pastel concrete colours · `M` music on/off · `G` chunk / ghost-halo outlines · `P` pause block updates.

**Farm animals:** up to 10 wander the grass around you: cows, sheep, pigs, chickens, horses, rabbits, cats and
mice ("Cube Farm Animals" by [ezgi bakim](https://sketchfab.com/ezgibakim),
[CC BY 4.0](http://creativecommons.org/licenses/by/4.0/), from
[Sketchfab](https://sketchfab.com/3d-models/cube-farm-animals-b28b7fd5c1454e9d9327fd546463d79f); credited on the
start screen too). They spawn on grass 8–20 blocks away, stroll, stand about, turn, hop up steps (rabbits hop all
the time), and waddle as they walk; hit one (dig at it) and it's knocked back and runs off. Each kind has its own
size and speeds (`src/world/animals.ts`) and uses the player's physics, against a wider box of blocks read back
twice a second (40 × 28 × 40); they despawn 40 blocks away. The models come from one glTF scene
(`assets/farm/`), split into an animal each, scaled, turned to face -z and simplified to at most 2,500 triangles
by `node scripts/convert-models.mjs` into `public/models/<animal>.bin` (80–190 KB each) sharing `farm.png`, and
are drawn as instances of a textured mesh per kind (`src/render/mobModel.ts`, `mobShader`) with the blocks' light
and fog, safe mode included.

**Music:** "The Longest Afternoon" loops quietly while you play (`public/music/`, re-encoded at 128 kbps: 2.9 MB,
fetched only once play starts). It fades in when play starts (browsers only let a page start sound from a click or
tap) and out on the start screen or when the tab is hidden; `M` or the touch Music button switch it off, remembered
between visits (`src/ui/music.ts`).

**Building:** you start with pastel concrete in 8 colours (pink, peach, butter, mint, sky, periwinkle, lavender,
cream). Concrete is stone with its colour in the cell's level bits (stone's level is otherwise unused; the 3 type
bits are all taken), so it's solid like stone everywhere and digs like stone; the meshers pass the colour to the
shader the way they pass a fluid's height.

**Diamonds:** diamond ore (stone with cyan gems) runs in small blobs through the deep stone, at y 24 and below
(sea level is 30), about 190 blocks a chunk: dig down, or look in caves. Digging takes a moment, as with a
pickaxe: hold the button (or touch Break) and a ring around the crosshair fills; dirt goes in 0.35 s, stone 0.75 s,
diamond ore 1.1 s, wheat at once (`src/player/digging.ts`). Diamond ore drops 1 to 4 diamonds rather than the block (1, 2, 3 or 4 at 40 / 30 / 20 / 10%):
they pop out, fall, and fly to you once you're within 3 blocks (`src/world/drops.ts`, drawn as small spinning line
diamonds). The diamonds you've collected show at the end of the hotbar (kept between visits). Diamond ore is block
type 7, the last of the 3 bits a cell has for its type, and is solid like stone everywhere (block updates,
meshing, picking, collisions).

**Touch (phones, tablets):** tap the start screen for on-screen controls. The left half of the screen
is a **dynamic stick**: it appears wherever your thumb lands, is analog (push further to go faster),
and disappears when you let go; double-tap and hold it to run. Drag on the right half to look around.
Buttons: Break (hold to dig), Place, ▲ to jump or swim up (▲ / ▼ up and down when flying), a
hotbar to pick the colour, and Fly / Music / Outlines / Pause / Menu at the top right. You can move and look at the same time with two thumbs.

The bottom layer (y = 0) is unbreakable bedrock.

Every push is built and tested by `.github/workflows/pages.yml`; pushes to the default branch are
published to GitHub Pages at `https://<owner>.github.io/<repo>/`.

**View distance:** pick 3, 4, 8, 16, 32 or 64 chunks on the start screen (remembered; default 8, or 3 on phones), up to
what the GPU's buffer limits hold: every loaded chunk's cells take 64 KB of one GPU buffer (about 280 MB at 32, 1.1 GB
at 64). Block updates run within 8 chunks at most (the simulation distance); further out chunks are drawn but frozen,
as in Minecraft. Meshes are packed into one face buffer at their real size (a few thousand faces a chunk), only chunks
in the camera's view are drawn, and depth is reversed with a float depth buffer so distant terrain stays steady. The
game is playable as soon as the area around you is loaded; the distance keeps loading, nearest first, and what
the camera sees before the rest (so flying forward fills in what's ahead). Generated terrain is settled, so new
chunks don't wake for block updates unless they or a neighbour were edited: moving costs only generating and
meshing. Flying is 20 blocks a second, sprinting 80.

**Walking:** you walk (4.5 blocks a second), run (7) and jump (1.25 blocks) with Minecraft's body size
(0.6 × 1.8 blocks), collide with solid blocks, and swim in water and lava; `F` switches to flying. The world
lives in GPU memory, so each frame the blocks around you (16 × 24 × 16) are read back, 24 KB arriving a frame
or two later, and the physics (`src/player/physics.ts`) runs against those on the CPU. Chunks that aren't
loaded yet count as solid, so you can't fall out of the world; if you end up inside a block you climb out.

**Far terrain** (mist by default; silhouette, colour or off on the start screen, or `?far=silhouette` and so on): beyond the chunks, the land
is drawn out to 2 km as one low-detail height field, in the spirit of Distant Horizons' level-of-detail chunks. The
ground's height is a formula (`terrainHeight` in `src/tf/worldgen.ts`, the one generation uses), so the far terrain
needs no chunks: it samples the formula on a grid around you (points 4 blocks apart nearby, up to 64 apart far away,
about 37,000 in all, 1.3 MB), recentred every 64 blocks you travel. In mist the chunks fade into the fog as usual
but the fog stops 85% thick, and the far terrain carries on from there, thinning into the sky toward its edge: faint
hills and coastlines through the haze. The other looks are a near-black silhouette that hazes into the sky, and
plain colour (grass, bare slopes and sea with the blocks' lighting, fog pushed out to 2 km). It isn't drawn where real chunks are. It has no caves, plants or edits, only the shape of
the land, and plain vertex buffers, so safe mode shows it too (`src/world/farTerrain.ts`).

**Safe mode** (`?safe`, linked from the start screen): the world on the CPU and the previous renderer (meshes built
on the CPU, plain indexed draws), for GPUs that crash on the GPU world, as an Adreno 6xx phone on Android 10 did.

URL params: `?radius=N` view distance in chunks · `pos=x,y,z` · `yaw=` / `pitch=` (radians) ·
`chunks` (outlines on) · `spread=` grass spread chance per tick (default 1/16, 0 = never) ·
`grow=` wheat growth chance per tick (default 1/40, 1/12 next to water) · `offscreen` (render to a
texture and copy it to a 2D canvas, for headless browsers where WebGPU canvas presentation isn't available) ·
`cpu` (keep the world on the CPU with the reference code instead, for comparison).

**If something goes wrong:** the **Log** button (top right on the start screen, and at the top whenever
an error has happened) shows the page log, with **Copy** and **Share** buttons for sending it from a phone.
It records the startup steps, the browser and GPU (adapter, features and limits), and everything that goes
wrong: errors with their stack traces, failed promises, console errors and warnings, WebGPU validation and
shader-compile errors, and a lost GPU device with the browser's reason. It's a plain script (`public/log.js`)
loaded before the game, so it also catches errors that stop the game loading. It keeps the previous
visits' logs, saved as they happen: if the game crashes the whole tab, open `log.html` (linked from the start screen) to see and copy them without starting WebGPU. If the GPU device is lost the game stops and says so.
After a GPU crash, browsers can switch WebGPU off for a while: fully close and reopen the browser.

**Benchmark:** the start screen's *Benchmark this device* button opens `bench.html`
(`https://<owner>.github.io/<repo>/bench.html`), which measures **block updates per second** on your GPU
with the game's code. One block update is one cell's next state for one tick. Press *Start benchmark*:
it times the game's tick loop (chunks updated in GPU memory, waiting for their flags each tick) for 1 to
200 different chunks per tick, and reports the best rate, how many chunks that could keep updating at
5 ticks per second, the "GPU only" rate (ticks issued back to back), the rate with fluid rules only, and
the **previous design** for comparison: the world on the CPU, each tick packing chunks with their ghost
borders, running the same rules as a TF.js kernel and reading every cell back. *Run takeover* simulates
grass and wheat taking over the terrain around spawn and logs tick times.

## Layout

| Path | What |
| --- | --- |
| `src/constants.ts` | Chunk dims (16×16×64), block ids, cell encoding (`type + 8 * level`) |
| `src/tf/worldgen.ts` | Batched chunk generation as one TF graph: fBm heightmap → stone/dirt, sea-level water, 3D-noise caves, deep lava lakes, grass-topped land, ripe wild wheat |
| `src/world/farTerrain.ts` | Far terrain: the ground-height formula sampled on a widening grid around the player, drawn beyond the chunks |
| `src/tf/noise.ts` | Value noise / fBm built from elementwise tensor ops |
| `src/tf/backend.ts` | Runs TF.js's WebGPU backend **on the renderer's `GPUDevice`** (so worldgen output can be copied GPU to GPU); falls back to WebGL, then CPU; pre-compiles kernels |
| `src/sim/rules.ts` | Every block-update rule as WGSL, shared by the GPU world and the TF.js kernel |
| `src/sim/gpuStore.ts` | **The world in GPU memory**: ticks in place, per-chunk flags, picking, chunk reads/writes |
| `src/sim/gpuWorldgen.ts` | World generation as one compute shader, straight into the world's slots (`tf/worldgen.ts` ported op for op) |
| `src/sim/cpuStore.ts` | The same operations on the CPU with the reference code: the Node tests' world, the GPU tests' oracle, the fallback |
| `src/sim/simulation.ts` | Ticks the awake chunks and acts on the flags that come back |
| `src/sim/check.ts` | Runs the GPU world against the reference on random cells; the game and benchmark run it at startup |
| `src/world/` | Chunk bookkeeping: ring slots, active area / ghost halo, awake chunks, versions for remeshing, edited chunks saved when they leave; the TF.js → store loader |
| `src/render/mesher.ts` | Face records (one `u32` per quad) and the reference mesher |
| `src/render/gpuMesher.ts` | The mesher as a compute shader, writing face records and indirect draw counts |
| `src/render/` (rest) | Mesh pool, WGSL shaders (vertex pulling from face records), renderer (opaque, lines, translucent water) |
| `src/tf/blockUpdateReference.ts` | The rules cell by cell in plain JS: the readable spec, and the oracle every other version is tested against |
| `src/tf/blockUpdate.ts`, `blockUpdateKernel.ts` | The rules as TF.js tensor ops, and as a TF.js custom kernel (the previous design's tick, kept for the benchmark) |
| `src/player/` | Fly camera + pointer lock, touch controls (dynamic stick, look drag, buttons), the reference voxel ray walk |
| `src/ui/hotbar.ts` | Block picker (keys 1–6 or tap) |
| `test/gpu/` | WebGPU tests: a page that checks everything on the GPU against the reference, and a Playwright runner |

## Chunks and the ghost halo

Chunks are 16×16 columns, 64 blocks tall. Around the player's chunk:

- **Active chunks** (Chebyshev distance ≤ `ACTIVE_RADIUS`, default 3 → 7×7): simulated and drawn.
- **Ghost chunks**: the ring one chunk further out (→ 32 chunks). These are generated and kept
  but never stepped or drawn. They are the halo (ghost cells) for the stencil computations:
  block updates at the edge of the active area read their neighbours from the ghost ring, so
  fluids there see real terrain, and the mesher reads them to cull faces on the outer border.

When the player crosses a chunk border, the window re-centres: ghosts that come within range
become active (they get meshed and simulated), active chunks that drop out become ghosts, and
chunks that leave the halo are dropped. Chunks that changed since they were generated are read back
from the GPU as they leave and written back when they return, so changes persist.
Press `G` to see active (green) and ghost (orange) outlines.

## The world lives in GPU memory

All the cells are in one storage buffer: a ring of 9×9 chunk slots (the active area plus the ghost
ring), chunk (cx, cz) in slot (cx mod 9, cz mod 9). Moving reuses the slots of the chunks that drop
out, so nothing is ever shifted around, and a chunk's neighbours are always the slots next to its own.

- **World generation** is one compute shader (`src/sim/gpuWorldgen.ts`) that writes chunks straight
  into their slots: one thread per column works out the ground height and plants once, then fills its
  64 cells, with cave noise only where caves can be. It is `src/tf/worldgen.ts` op for op (the TF.js
  version stays the reference and the generator for a world on the CPU); the GPU tests require them
  to agree on all but 1 cell in 10,000 (rounding in cave noise far from the origin), and in practice
  11 of 2.4 million differ. As TF.js ops a batch of 16 chunks was ~800 dispatches and ~25 ms of main
  thread, so frames dropped while you walked; the shader is one dispatch, 0.2 ms of main thread, and
  about 50 times faster overall (0.35 ms a chunk on a software GPU, 17 ms before).
- **Block updates** run where the cells are. A tick takes the awake chunks and, for each, the slots
  of its eight neighbours; one compute pass works out every cell's next state (into a scratch buffer,
  so all chunks step from the same state) and ORs each cell's *flags* into its chunk's flags word:
  whether something changed that matters, which borders that touched, and whether plants are still
  growing. The results are copied back into the slots. The flags, 4 bytes per chunk, are the only
  thing read back, and decide what remeshes and what stays awake.
- **Meshing** is a compute shader too. Each visible face becomes a 32-bit record (block position, face,
  type, fluid height or growth stage) in the chunk's mesh slot, and each slot has indirect draw
  arguments that the mesher counts up, so the CPU never learns how many faces a chunk has. The vertex
  shader turns each record into a quad.
- **Picking** (the block under the crosshair) is a tiny compute shader that walks the ray through the
  world buffer. Its answer arrives a frame or two later; an edit discards older answers, so clicks
  never act on blocks that have already changed. **Edits** are single-cell writes into the buffer.

GPU work runs in the order it's issued, so none of this needs locking: an edit made while a tick is
running lands after it, and a mesh always shows the cells as of when it was made.

## Sleeping chunks and batches

Most of the world is static most of the time, so block updates only run where something can change.
A chunk is **awake** after an edit, when it's freshly loaded or becomes active, when a neighbouring
chunk changed along their shared border (or corner, since grass spreads diagonally), or while it has
plants that can still change by chance. Each tick (5 per second) updates the awake chunks; cost scales
with how many there are, wherever they are.

A chunk that didn't change, with neighbours that didn't change, is a fixed point of the deterministic
rules, so it goes back to sleep. Random rules break that ("nothing happened" can just be luck), so the
step marks dirt that grass could spread onto (*primed* dirt, a flag in the cell's spare level bits), and
chunks with primed dirt or unripe wheat stay awake. Priming doesn't count as a change, so it never
triggers a remesh. With nothing changing, no tick runs. The HUD shows the batch size, how many chunks
are still growing, and how many bytes a tick reads back.

## Block update rules

All cells in the region update in parallel from the previous state:

- Sources have level 8; flowing fluid has level 1–7 and is recomputed every tick, so it drains when cut off.
- Fluid above an air or flowing cell falls into it (level 7).
- Fluid resting on solid ground or on a source spreads sideways at `level − decay`
  (water decays by 1, lava by 2, so lava spreads less far).
- Lava with water beside or above it turns to stone; a cell that both fluids would flow into becomes stone.
- **Grass** spreads like Minecraft's: onto dirt with air above it, from living grass one block to the side
  and from one below to three above (a 5×3×3 box), with a chance per tick. Grass under a solid
  block or a fluid source dies back to dirt.
- **Wheat** grows through stages 0–7 with a chance per tick, faster with water beside it or its soil
  (Minecraft hydrates from 4 blocks away; here it's 1, so no rule looks more than one block sideways).
  It pops off without dirt or grass under it, and flowing fluid washes it away.
- World generation tops dry land with grass and adds small patches of ripe wild wheat, and keeps
  lava away from water, so a fresh world is already settled: it sleeps after its first tick and
  only your edits wake it. Grass and wheat are also in the hotbar.

## The rules, the reference and the tests

- **WGSL** (`sim/rules.ts`): one function that works out a cell's next state from its neighbours. The
  GPU world runs it on chunks in place; the TF.js custom kernel (`blockUpdateKernel.ts`) runs it on
  chunks packed with their ghost borders, the previous design, which the benchmark still times.
- **Tensor ops** (`blockUpdate.ts`): the same rules as about a hundred TF.js operations.
- **Reference** (`blockUpdateReference.ts`): plain JS, one cell at a time. The spec.

The plant rules' random numbers make exact comparison possible: an integer hash (PCG) of each cell's
index and a per-tick seed, which the reference reproduces bit for bit (`cellRandom`). `CpuStore` does
everything the GPU world does with reference code (the rules, the mesher, the ray walk), so the tests
can compare the two exactly: `npm test` runs the world logic on it (halo, sleeping, flags, saving
chunks, plants) and checks the tensor ops against the reference; `npm run test:gpu` runs on a real
WebGPU device, in CI too, and compares ticks, flags, meshes and picking on random cells, a whole game
session (worldgen, edits, ticks, moving away and back) on both stores, the worldgen shader against
TF.js generation, and the TF.js versions. The
game and the benchmark run a small version of that check at startup; if this GPU disagrees, the game
keeps its world on the CPU instead.

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
