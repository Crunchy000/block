import * as tf from '@tensorflow/tfjs';
import {
  ACTIVE_RADIUS, BUILDING_BLOCKS, Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, DEFAULT_RATES, cell,
} from './constants';
import { log, logError } from './log';
import { Controls } from './player/controls';
import { Digging } from './player/digging';
import { EYE_HEIGHT, Nearby, overlaps } from './player/physics';
import { TouchControls } from './player/touchControls';
import type { RayHit } from './player/raycast';
import { fpsView, frustum, multiply, perspective } from './render/math';
import { MeshPool } from './render/meshPool';
import { fetchBlockTextures } from './render/blockTextures';
import { GlRenderer } from './render/glRenderer';
import { fetchMobModel, poseFrames } from './render/mobModel';
import { Renderer } from './render/renderer';
import type { GameRenderer, MobModelHandle } from './render/types';
import { FADE_MS } from './render/shaders';
import { checkGpuStore } from './sim/check';
import { CpuStore } from './sim/cpuStore';
import { GpuStore } from './sim/gpuStore';
import { Simulation } from './sim/simulation';
import { NOT_LOADED, ringSize, type CellStore } from './sim/store';
import { fallbackBackend, initTensorflow, warmUpKernels } from './tf/backend';
import { createHotbar } from './ui/hotbar';
import { Music } from './ui/music';
import { Sounds } from './ui/sounds';
import { Drops, diamondDropCount } from './world/drops';
import { Animals, SPECIES, type PoseFrames } from './world/animals';
import { FarTerrain, SEA_SURFACE } from './world/farTerrain';
import { farHeightsJs, WorldgenWorker } from './world/jsWorldgen';
import { generateMissing, generateMissingJs } from './world/loader';
import { World, meshSlotCount } from './world/world';

const TICK_MS = 200;          // block-update rate (5 ticks / second)
const PICK_DISTANCE = 8;      // blocks

const $ = (id: string) => document.getElementById(id)!;

/** View distances offered on the start screen (chunks from the player's chunk). */
const VIEW_DISTANCES = [3, 4, 8, 16, 32, 64];
const VIEW_KEY = 'block.viewDistance';
const FAR_KEY = 'block.farTerrain';
/** Block updates run this far out at most (chunks; in safe mode, on the CPU, less); beyond it chunks are drawn but frozen. */
const MAX_SIMULATION_DISTANCE = 8;
const SAFE_SIMULATION_DISTANCE = 1;

/**
 * The furthest view distance this GPU can hold: the cells of every loaded chunk (the view
 * plus a ring) are one GPU buffer of 64 KB a chunk. Safe mode and WebGL2 build meshes on the
 * CPU (no `device`), which is slow for big areas.
 */
function maxViewDistance(device: GPUDevice | undefined, safe: boolean): number {
  if (safe || !device) return 8;
  const bytes = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const ring = Math.floor(Math.sqrt(bytes / (CHUNK_VOLUME * 4)));
  return Math.max(1, Math.floor((ring - 1) / 2) - 1);
}

/** ?radius=N, else the one picked on the start screen, else 3 on phones, 4 with WebGL2 and 8 elsewhere; within what the GPU holds. */
function viewDistance(params: URLSearchParams, device: GPUDevice | undefined, safe: boolean): number {
  let stored: string | null = null;
  try { stored = localStorage.getItem(VIEW_KEY); } catch { /* storage blocked */ }
  const fallback = matchMedia('(pointer: coarse)').matches ? ACTIVE_RADIUS : device ? 8 : 4;
  const asked = Number(params.get('radius') ?? stored ?? fallback);
  return Math.min(maxViewDistance(device, safe), Math.max(1, Math.floor(asked) || ACTIVE_RADIUS));
}

/** How the far terrain looks, if at all. */
type FarStyle = 'mist' | 'silhouette' | 'colour' | 'off';
const FAR_STYLES: FarStyle[] = ['mist', 'silhouette', 'colour', 'off'];

/** ?far=mist|silhouette|colour|off (0 is off), else the start screen's choice, else mist. */
function farStyle(params: URLSearchParams): FarStyle {
  let stored: string | null = null;
  try { stored = localStorage.getItem(FAR_KEY); } catch { /* storage blocked */ }
  const asked = params.get('far') ?? stored;
  if (asked === '0') return 'off';
  return FAR_STYLES.find((s) => s === asked) ?? 'mist';
}

/** The far terrain switch on the start screen, under the view distances: remembered, and reloads. */
function showFarTerrain(current: FarStyle): void {
  const box = $('view-distance');
  box.append(document.createElement('br'), 'Far terrain: ');
  for (const style of FAR_STYLES) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = style;
    b.className = style === current ? 'chosen' : '';
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      try { localStorage.setItem(FAR_KEY, style); } catch { /* storage blocked */ }
      const url = new URL(location.href);
      url.searchParams.delete('far');
      if (style !== current) location.href = url.toString();
    });
    box.append(b, ' ');
  }
}

/** The view-distance picker on the start screen: choosing one remembers it and reloads. */
function showViewDistances(current: number, max: number, safe: boolean): void {
  const box = $('view-distance');
  box.replaceChildren('View distance: ');
  for (const d of [...new Set([...VIEW_DISTANCES, current])].filter((v) => v <= max).sort((a, b) => a - b)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = String(d);
    b.className = d === current ? 'chosen' : '';
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      try { localStorage.setItem(VIEW_KEY, String(d)); } catch { /* storage blocked: the URL carries it */ }
      const url = new URL(location.href);
      url.searchParams.delete('radius');
      if (d !== current) location.href = url.toString();
    });
    box.append(b, ' ');
  }
  const limit = safe ? ` (up to ${max} in safe mode)` : ` (up to ${max} here)`;
  box.append(`chunks${max < VIEW_DISTANCES[VIEW_DISTANCES.length - 1] ? limit : ''}`);
}

async function main(): Promise<void> {
  const startedAt = performance.now();
  const canvas = $('gpu') as HTMLCanvasElement;
  const overlay = $('overlay'), hud = $('hud'), errorBox = $('error');
  let statusText = '';
  // Startup steps go into the page log too (the Log button), so a failure shows where it happened.
  const setStatus = (text: string, logIt = true) => {
    if (text === statusText) return;
    $('status').textContent = statusText = text;
    if (logIt) log.info(text);
  };
  let lastError = '';
  const showError = (text: string) => {
    lastError = text;
    errorBox.textContent = text;
  };
  /** The GPU device was lost: nothing more can run, the start screen says so. */
  let gpuLost = false;
  let worldReady = false;

  log.info(`Block build ${__BUILD__}`);
  setStatus('Starting the graphics…');
  const params = new URLSearchParams(location.search);
  // 4x MSAA smooths block edges (?msaa=0 turns it off).
  const msaa = params.get('msaa') !== '0';
  // WebGPU, with the world in GPU memory. Or safe mode: WebGL2, with the world on the CPU and
  // meshes built there, block updates only near the player. Safe mode is for GPUs (some phones)
  // that crash on the GPU world, and browsers without WebGPU: asked for (?safe, or ?webgl), or
  // whenever WebGPU is missing or fails to start.
  let safe = params.has('safe') || params.has('webgl');
  let renderer: GameRenderer, device: GPUDevice | undefined, adapterInfo: GPUAdapterInfo | undefined;
  try {
    if (safe) throw new Error('safe mode asked for');
    const webgpu = await Renderer.create(canvas, { offscreen: params.has('offscreen'), msaa });
    renderer = webgpu;
    ({ device, adapterInfo } = webgpu);
  } catch (e) {
    const [why] = log.describe(e);
    if (safe) log.info('Safe mode: WebGL2, the world on the CPU');
    else log.warn(`No WebGPU (${why}): safe mode instead (WebGL2, the world on the CPU)`);
    safe = true;
    renderer = GlRenderer.create(canvas, { msaa, offscreen: params.has('offscreen') });
  }
  // Block textures ("Baunilha" by Mirtilo, CC BY-SA 4.0): until they load, or if they don't, blocks keep their plain look.
  fetchBlockTextures()
    .then((t) => renderer.setBlockTextures(t))
    .catch((e: unknown) => logError('Loading the block textures failed (plain blocks instead)', e));
  log.info(`Graphics: ${renderer.api}; antialiasing: ${msaa ? '4x MSAA' : 'off'}`);
  // World generation: with WebGPU, the worldgen shader (TF.js too, should the GPU world fail its
  // check); in safe mode, the plain-JS generator in a worker, and no TF.js at all.
  const worldgenWorker = safe ? new WorldgenWorker() : undefined;
  let tfBackend = 'none';
  if (!safe) {
    setStatus('Starting TensorFlow.js…');
    tfBackend = await initTensorflow(device, adapterInfo);
    log.info(`TensorFlow.js backend: ${tfBackend}`);
    setStatus(`Compiling GPU kernels (${tfBackend})…`);
    try {
      await warmUpKernels();
    } catch (e) {
      logError('Kernel warm-up failed (kernels will compile on first use)', e);
    }
  }
  const viewRadius = viewDistance(params, device, safe);
  const activeRadius = Math.min(viewRadius, safe ? SAFE_SIMULATION_DISTANCE : MAX_SIMULATION_DISTANCE);
  const ghostRadius = viewRadius + 1;
  log.info(`View distance ${viewRadius} (${(2 * viewRadius + 1) ** 2} chunks drawn), simulation distance ${activeRadius}`);
  showViewDistances(viewRadius, maxViewDistance(device, safe), safe);
  // Safe mode leaves the far terrain out (unless ?far= asks for it), and its switch off the start screen.
  const farLook = safe && !params.has('far') ? 'off' : farStyle(params);
  if (!safe) showFarTerrain(farLook);
  // (Already in safe mode: no link to it.)
  if (safe) $('safe-link').style.display = 'none';
  // The world lives in GPU memory, where block updates, meshing and picking run. First
  // check that this GPU computes them exactly as the reference code does; if it doesn't,
  // the world lives on the CPU with the reference code instead (slower, same game).
  const onCpu = safe || !device || params.has('cpu');
  if (!onCpu) setStatus('Checking the GPU world code…');
  const check = onCpu || !device ? { ok: false, summary: '', detail: safe ? 'safe mode' : '?cpu' }
    : await checkGpuStore(device).catch((e: unknown) => ({ ok: false, summary: '', detail: log.describe(e).join('\n') }));
  if (check.ok) log.info('GPU world check passed', check.summary);
  else if (!onCpu) log.warn('GPU world check failed: the world stays on the CPU', check.detail);
  setStatus('Setting up the world…');
  const store: CellStore = check.ok && !onCpu && device
    ? await GpuStore.create(device, ringSize(ghostRadius))
    : new CpuStore(ringSize(ghostRadius));
  log.info(store instanceof GpuStore ? 'World: in GPU memory' : 'World: on the CPU');
  const meshes = renderer.createMeshes(meshSlotCount(viewRadius));
  // The land beyond the chunks, out to at least 2 km (1 km past the chunks at long view distances).
  const far = farLook !== 'off' ? new FarTerrain(Math.max(2048, (viewRadius + 1) * CHUNK_SIZE + 1024), undefined, safe ? farHeightsJs : undefined) : undefined;
  if (far) log.info(`Far terrain: out to ${far.extent} blocks, ${far.points.toLocaleString()} points, ${(far.bytes / 2 ** 20).toFixed(1)} MB`);

  // If the TF backend breaks at runtime (e.g. a driver limit), drop to the next one.
  let switching = false;
  const onTfError = (what: string) => (e: unknown) => {
    logError(`${what} failed on TF.js backend ${tfBackend}`, e);
    showError(`${what} failed on ${tfBackend}: ${e instanceof Error ? e.message : String(e)}`);
    if (switching) return;
    switching = true;
    fallbackBackend()
      .then(async (name) => {
        if (name !== tfBackend) {
          showError(`${lastError} (switched TensorFlow.js to ${name})`);
          log.warn(`Switched TensorFlow.js from ${tfBackend} to ${name}`);
        }
        tfBackend = name;
        await warmUpKernels().catch(() => {});
      })
      .finally(() => { switching = false; });
  };

  const world = new World(store, activeRadius, ghostRadius, viewRadius);
  // ?spread= / ?grow= tune the per-tick chances of grass spreading (default 1/16; 0 stops it)
  // and wheat growing a stage (default 1/40, and 1/12 next to water; ?grow= sets both).
  const chance = (name: string) => Math.min(1, Math.max(0, Number(params.get(name))));
  const sim = new Simulation(world, {
    ...DEFAULT_RATES,
    ...(params.has('spread') && { grassSpread: chance('spread') }),
    ...(params.has('grow') && { wheatGrow: chance('grow'), wheatGrowWet: chance('grow') }),
  });
  // Optional URL params: ?pos=x,y,z&yaw=rad&pitch=rad&chunks (outlines on)&radius=N&spread=&grow=&offscreen&cpu&fly (start flying)&msaa=0&safe (or webgl: WebGL2, the world on the CPU)
  const pos = (params.get('pos') ?? '8,52,8').split(',').map(Number) as [number, number, number];
  const controls = new Controls(canvas, pos);
  controls.flying = params.has('fly');
  if (params.has('yaw')) controls.yaw = Number(params.get('yaw'));
  if (params.has('pitch')) controls.pitch = Number(params.get('pitch'));
  controls.onLockError = (message) => {
    showError(`Couldn't capture the mouse: ${message} Click again to retry.`);
    music.setPlaying(controls.playing);
  };
  const touchUI = new TouchControls(controls);
  // "The Longest Afternoon", looped while playing (M or the touch Music button switch it off).
  const music = new Music('music/the-longest-afternoon.mp3');
  touchUI.setToggle('KeyM', music.enabled);
  // Sound effects (N or the touch Sounds button switch them off).
  const sounds = new Sounds();
  touchUI.setToggle('KeyN', sounds.enabled);
  controls.onPlayingChange = (playing) => {
    music.setPlaying(playing);
    overlay.classList.toggle('hidden', playing);
    document.body.classList.toggle('playing', playing);
    touchUI.setVisible(controls.touchPlaying);
    if (playing && lastError.startsWith("Couldn't capture")) showError('');
  };

  // The start screen starts play: a tap gets on-screen touch controls, a mouse click captures the mouse.
  // (Safari's click events don't say which pointer made them, so remember it from pointerdown.)
  // The benchmark link sits on the start screen; following it shouldn't start the game too.
  for (const link of overlay.querySelectorAll('a')) link.addEventListener('click', (e) => e.stopPropagation());
  let startPointer = 'mouse';
  overlay.addEventListener('pointerdown', (e) => { startPointer = e.pointerType; });
  overlay.addEventListener('click', (e) => {
    if (gpuLost) {
      location.reload();
      return;
    }
    const type = (e as PointerEvent).pointerType || startPointer;
    if ((type === 'touch' || type === 'pen') && document.fullscreenEnabled && !document.fullscreenElement) {
      // More room on phones, and no accidental pull-to-refresh. Unsupported on iPhone; that's fine.
      document.documentElement.requestFullscreen({ navigationUI: 'hide' }).catch(() => {});
    }
    controls.start(type);
    music.setPlaying(true); // within the click, which lets the page start sound
    sounds.unlock();
  });
  const touchFirst = matchMedia('(pointer: coarse)').matches;

  // A lost GPU device (a GPU crash or hang, or the browser reclaiming it) can't be used again:
  // stop, explain on the start screen, and open the log (gpu.ts has logged the browser's reason).
  renderer.lost.then((info) => {
    gpuLost = true;
    if (document.pointerLockElement) document.exitPointerLock();
    if (controls.touchPlaying) controls.stopTouch();
    overlay.classList.remove('hidden');
    document.body.classList.remove('playing');
    touchUI.setVisible(false);
    setStatus('The GPU stopped working', false);
    showError(`${renderer.api} ${renderer.api === 'WebGPU' ? 'device' : 'context'} lost (${info.reason}): ${(info.message || 'no details').replace(/\.$/, '')}. `
      + 'Tap or click to reload; if it then fails, fully close and reopen the browser.');
    log.open();
  });

  // Console / automation handle, e.g. block.world.setCell(x, y, z, value).
  Object.assign(window, { block: { world, sim, controls, renderer, store, tf } });

  const gpuName = `${renderer.gpuName} (${renderer.api})`;

  let generating = false;
  /** The block placed (an index into BUILDING_BLOCKS). */
  let selected = 0;
  let showChunks = params.has('chunks');
  // Block updates (water flowing, grass spreading, wheat growing): off to start with in safe mode,
  // where they run on the CPU and stall the game (P, or the touch Pause button, turns them on).
  let paused = safe;
  touchUI.setToggle('KeyP', paused);
  const hotbar = createHotbar((index) => { selected = index; hotbar.setSelected(index); });
  hotbar.setSelected(selected);
  touchUI.setToggle('KeyG', showChunks);
  touchUI.setToggle('KeyF', controls.flying);

  // Diamonds mined (kept between visits, though edits to the world aren't yet).
  const DIAMONDS_KEY = 'block.diamonds';
  let diamonds = 0;
  try { diamonds = Math.max(0, Number(localStorage.getItem(DIAMONDS_KEY)) || 0); } catch { /* storage blocked */ }
  const setDiamonds = (n: number, picked: boolean) => {
    diamonds = n;
    hotbar.setDiamonds(n);
    if (picked) hotbar.flashDiamonds();
    try { localStorage.setItem(DIAMONDS_KEY, String(n)); } catch { /* storage blocked */ }
  };
  hotbar.setDiamonds(diamonds);
  let lastTick = 0;
  let last = performance.now();
  let fps = 0;

  // One batch of world generation at a time: TF.js writes it straight into the world's GPU
  // memory, and the next batch waits until the GPU has done this one.
  const pumpGeneration = () => {
    if (generating || switching || world.missingChunks(1).length === 0) return;
    generating = true;
    (worldgenWorker ? generateMissingJs(world, worldgenWorker) : generateMissing(world))
      .then(async (n) => {
        await device?.queue.onSubmittedWorkDone();
        // Each batch in the log, so a crash while generating shows how far it got.
        const { loaded, total } = world.haloProgress();
        log.info(`Generated ${n} chunks (${loaded} / ${total})`);
      })
      .catch(worldgenWorker ? (e: unknown) => logError('World generation failed', e) : onTfError('worldgen'))
      .finally(() => { generating = false; });
  };

  // What's under the crosshair, worked out on the GPU from the world there. The answer
  // arrives a frame or two later; until then the last one stands, except that an edit
  // throws it away (and any answer worked out before the edit), so clicks never act on
  // blocks that have already changed.
  let hit: RayHit | null = null;
  let picking = false;
  let edits = 0;
  const pick = () => {
    if (picking) return;
    picking = true;
    const asked = edits;
    store.raycast(controls.position, controls.look, PICK_DISTANCE)
      .then((h) => { if (asked === edits) hit = h; })
      .catch((e: unknown) => console.warn('picking failed', e))
      .finally(() => { picking = false; });
  };

  // Collisions: the blocks around the player, read back from the world (24 KB) once the last
  // read has arrived, a frame or two behind. A read issued before an edit is thrown away
  // (the edit is applied to the last one instead), so a broken block never comes back.
  const BOX = [16, 24, 16];
  let nearby: Nearby | undefined;
  let readingBox = false;
  const readNearby = () => {
    if (readingBox) return;
    readingBox = true;
    const p = controls.position, asked = edits;
    const min = [Math.floor(p[0]) - BOX[0] / 2, Math.floor(p[1]) - 14, Math.floor(p[2]) - BOX[2] / 2];
    store.readBox(min, BOX)
      .then((types) => { if (asked === edits) nearby = new Nearby(min, BOX, types); })
      .catch((e: unknown) => logError('Reading the blocks around the player failed', e))
      .finally(() => { readingBox = false; });
  };

  // Farm animals wander on the grass around the player. They need the blocks over a wider
  // area than the player does, but not every frame: a bigger box, read back twice a second.
  const MOB_BOX = [40, 28, 40];
  const animals = new Animals();
  let mobBlocks: Nearby | undefined;
  let readingMobBox = false, mobBoxAt = 0;
  const readMobBox = (now: number) => {
    if (readingMobBox || now - mobBoxAt < 500) return;
    readingMobBox = true;
    mobBoxAt = now;
    const p = controls.position, asked = edits;
    const min = [Math.floor(p[0]) - MOB_BOX[0] / 2, Math.floor(p[1]) - 16, Math.floor(p[2]) - MOB_BOX[2] / 2];
    store.readBox(min, MOB_BOX)
      .then((types) => { if (asked === edits) mobBlocks = new Nearby(min, MOB_BOX, types); else mobBoxAt = 0; })
      .catch((e: unknown) => logError('Reading the blocks around the animals failed', e))
      .finally(() => { readingMobBox = false; });
  };
  // Their models ("Cube Pets" by Kenney, CC0), animated, all sharing one texture.
  const animalModels = new Map<string, MobModelHandle>();
  let animalsReady = false;
  const animalFrames: PoseFrames = (name, clip, time) => poseFrames(animalModels.get(name)!, clip, time);
  Promise.all(SPECIES.map((s) => fetchMobModel(`models/${s.name}.bin`, 'models/pets.png').then((m) => animalModels.set(s.name, renderer.createMobModel(m)))))
    .then(() => { animalsReady = true; })
    .catch((e: unknown) => logError('Loading the animal models failed (no animals)', e));
  let animalHitCooldown = 0;
  // ?debug: the world, animals and controls on window.blockDebug, for poking at from the console (and scripted checks).
  if (params.has('debug')) Object.assign(window, { blockDebug: { world, animals, controls, music, sounds } });

  const handleInput = () => {
    for (const key of controls.takeKeyPresses()) {
      const slot = /^Digit([1-9])$/.exec(key);
      const index = slot ? Number(slot[1]) - 1 : -1;
      if (index >= 0 && index < BUILDING_BLOCKS.length) {
        selected = index;
        hotbar.setSelected(selected);
      }
      if (key === 'KeyF') {
        controls.toggleFlying();
        touchUI.setToggle(key, controls.flying);
      }
      if (key === 'KeyG') touchUI.setToggle(key, showChunks = !showChunks);
      if (key === 'KeyM') touchUI.setToggle(key, music.toggle());
      if (key === 'KeyN') touchUI.setToggle(key, sounds.toggle());
      if (key === 'KeyP') touchUI.setToggle(key, paused = !paused);
    }
    for (const button of controls.takeClicks()) {
      if (!hit) continue;
      // y = 0 is bedrock: it holds up fluids at the bottom of the world.
      if (button === 2) {
        // Not a solid block where the player stands.
        const [bx, by, bz] = hit.before, p = controls.position;
        const inside = overlaps([p[0], p[1] - EYE_HEIGHT, p[2]], (x, y, z) => (x === bx && y === by && z === bz ? Block.Stone : Block.Air));
        if (!controls.flying && inside) continue;
        const placing = BUILDING_BLOCKS[selected];
        world.setCell(...hit.before, placing.cell);
        sounds.placeBlock(hit.before.map((v) => v + 0.5));
        nearby?.set(...hit.before, placing.block);
        mobBlocks?.set(...hit.before, placing.block);
      } else continue;
      edits++;
      hit = null;
    }
  };

  // Digging: hold to break the block under the crosshair, harder blocks taking longer
  // (player/digging.ts). Diamond ore drops a diamond to pick up, as if mined with a pickaxe.
  const digging = new Digging();
  const drops = new Drops();
  const digRing = $('dig');
  const dig = (dt: number) => {
    // An animal in front of the block under the crosshair gets hit instead (knocked back, it runs off).
    animalHitCooldown -= dt;
    const eye = controls.position;
    const struck = animals.raycast(eye, controls.look, PICK_DISTANCE);
    const blockDist = hit ? Math.hypot(...hit.block.map((v, k) => v + 0.5 - eye[k])) : Infinity;
    if (struck && struck.dist < blockDist) {
      if (controls.digging && animalHitCooldown <= 0) {
        animals.hit(struck.animal, eye);
        const a = struck.animal;
        sounds.call(a.species.name, [a.feet[0], a.feet[1] + a.species.size.height / 2, a.feet[2]], true);
        animalHitCooldown = 0.5;
      }
      digging.step(dt, false, undefined, undefined);
      digRing.classList.remove('on');
      return;
    }
    // y = 0 is bedrock: it holds up fluids at the bottom of the world.
    const target = hit && hit.block[1] > 0 ? hit : null;
    const centre = target?.block.map((v) => v + 0.5);
    if (digging.progress > 0 && target && (digTick -= dt) <= 0) {
      sounds.dig(target.type, centre!);
      digTick = 0.3;
    }
    if (digging.step(dt, controls.digging, target?.block, target?.type) && target) {
      sounds.breakBlock(target.type, centre!);
      digTick = 0;
      world.setCell(...target.block, cell(Block.Air));
      nearby?.set(...target.block, Block.Air);
      mobBlocks?.set(...target.block, Block.Air);
      if (target.type === Block.Diamond) for (let n = diamondDropCount(); n > 0; n--) drops.spawn(...target.block);
      edits++;
      hit = null;
    }
    digRing.classList.toggle('on', digging.progress > 0);
    digRing.style.setProperty('--p', digging.progress.toFixed(3));
    // Pick up what's lying around (where the blocks are known).
    const p = controls.position;
    const got = drops.update(dt, [p[0], p[1] - 0.7, p[2]], nearby ? (x, y, z) => nearby!.at(x, y, z) : () => NOT_LOADED);
    if (got > 0) setDiamonds(diamonds + got, true);
  };
  let digTick = 0;

  // Footsteps, landing and splashing, from how the player's body moved this frame.
  let stepDistance = 0;
  const bodySounds = (before: { onGround: boolean; inFluid: boolean; fallSpeed: number; feet: number[] }) => {
    if (controls.flying) return;
    const b = controls.body, p = controls.position, feet = [p[0], p[1] - EYE_HEIGHT, p[2]];
    if (!before.inFluid && b.inFluid) sounds.splash();
    if (!before.onGround && b.onGround && before.fallSpeed > 7 && !b.inFluid) sounds.land(before.fallSpeed);
    // A step every 2.5 blocks walked: on the block underfoot, or wading through water.
    const at = (dy: number) => nearby?.at(Math.floor(feet[0]), Math.floor(feet[1] + dy), Math.floor(feet[2]));
    const wading = b.inFluid && at(0.1) === Block.Water;
    if ((b.onGround && !b.inFluid) || wading) {
      stepDistance += Math.hypot(feet[0] - before.feet[0], feet[2] - before.feet[2]);
      if (stepDistance > 2.5) {
        stepDistance = 0;
        const under = at(-0.05);
        if (wading) sounds.wade();
        else if (under !== undefined && under !== NOT_LOADED) sounds.step(under as Block);
      }
    }
  };

  const buildLines = (hit: RayHit | null): Float32Array<ArrayBuffer> => {
    const out: number[] = [];
    const box = (x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, col: number[]) => {
      const p = [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]];
      const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
      for (const [a, b] of edges) out.push(...p[a], ...col, ...p[b], ...col);
    };
    if (hit) {
      const [x, y, z] = hit.block, e = 0.002;
      box(x - e, y - e, z - e, x + 1 + e, y + 1 + e, z + 1 + e, [0.05, 0.05, 0.05]);
    }
    if (showChunks) {
      for (const c of world.chunks.values()) {
        const d = world.distance(c.cx, c.cz);
        if (d > world.activeRadius + 1) continue; // the simulated area and the ring around it
        const col = c.state === 'active' ? [0.2, 1, 0.3] : [1, 0.55, 0.1];
        const x0 = c.cx * CHUNK_SIZE + 0.05, z0 = c.cz * CHUNK_SIZE + 0.05;
        box(x0, 0.05, z0, x0 + CHUNK_SIZE - 0.1, CHUNK_HEIGHT - 0.05, z0 + CHUNK_SIZE - 0.1, col);
      }
    }
    out.push(...drops.lines(performance.now() / 1000));
    return new Float32Array(out);
  };

  const blockUpdateStatus = () => {
    const batch = `${sim.lastBatch} chunk${sim.lastBatch === 1 ? '' : 's'}`;
    if (paused) return safe ? 'off in safe mode ([P] turns them on)' : 'paused';
    if (!world.haloReady()) return 'waiting for terrain';
    if (sim.busy && sim.ticks === 0) return 'running the first tick…';
    if (sim.ticks === 0) return 'starting…';
    if (sim.asleep) return `asleep after tick ${sim.ticks} (nothing changing)`;
    return `tick ${sim.ticks}: ${batch}, ${sim.lastTickMs.toFixed(0)} ms ` +
      `(main thread ${sim.lastCpuMs.toFixed(0)} ms), ${sim.lastChangedChunks} changed`;
  };

  const frame = (now: number) => {
    if (gpuLost) return; // nothing more the GPU can do: stop here, with the start screen explaining
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    fps = fps * 0.95 + (dt > 0 ? 1 / dt : 0) * 0.05;

    readNearby();
    const was = {
      onGround: controls.body.onGround, inFluid: controls.body.inFluid, fallSpeed: -controls.body.velocity[1],
      feet: [controls.position[0], controls.position[1] - EYE_HEIGHT, controls.position[2]],
    };
    controls.update(dt, nearby && ((x, y, z) => nearby!.at(x, y, z)));
    bodySounds(was);
    sounds.setListener({ position: controls.position, yaw: controls.yaw });
    const [px, py, pz] = controls.position;
    world.recenter(px, pz);
    pumpGeneration();

    const eye = controls.position;
    handleInput();
    pick();
    dig(dt);
    readMobBox(now);
    if (mobBlocks && animalsReady) {
      const p = controls.position, blocksNear = mobBlocks;
      animals.update(dt, [p[0], p[1] - EYE_HEIGHT, p[2]], (x, y, z) => blocksNear.at(x, y, z));
      // Now and then an animal calls (about every 12 s each).
      for (const a of animals.animals) {
        if (sounds.hasCall(a.species.name) && Math.random() < dt / 12) {
          sounds.call(a.species.name, [a.feet[0], a.feet[1] + a.species.size.height / 2, a.feet[2]]);
        }
      }
    }

    if (!paused && now - lastTick >= TICK_MS && !sim.busy) {
      lastTick = now;
      sim.tick().catch((e: unknown) => {
        logError('Block update failed', e);
        showError(`block update failed: ${e instanceof Error ? e.message : String(e)}`);
      });
    }
    world.remesh(meshes);

    const { loaded, total } = world.haloProgress();
    if (loaded === total && !worldReady) {
      worldReady = true;
      log.info(`World ready: ${total} chunks, ${((performance.now() - startedAt) / 1000).toFixed(1)} s after start`);
    }
    // Playable once the area around the player is in; the distance keeps loading.
    const play = `${touchFirst ? 'Tap' : 'Click'} anywhere to play`;
    setStatus(!world.simReady() ? `Generating world… ${loaded} / ${total} chunks` : loaded < total ? `${play} (loading ${loaded} / ${total} chunks)` : play, false);

    far?.update(eye[0], eye[2]);
    // In mist the chunks fade into the fog as without far terrain; the other looks push the fog out to its edge.
    const fogDistance = far && farLook !== 'mist' ? far.extent : world.viewRadius * CHUNK_SIZE + 8;
    const proj = perspective((70 * Math.PI) / 180, renderer.aspect, 0.1);
    const viewProj = multiply(proj, fpsView(eye, controls.yaw, controls.pitch));
    // Only chunks the camera can see.
    const inView = frustum(viewProj);
    const draws = world.draws((x, z) => inView([x * CHUNK_SIZE, 0, z * CHUNK_SIZE], [(x + 1) * CHUNK_SIZE, CHUNK_HEIGHT, (z + 1) * CHUNK_SIZE]));
    // Load and mesh what's ahead first: chunks the camera sees, and a chunk around them (their neighbours, for meshing).
    world.focus = (x, z) => inView([(x - 1) * CHUNK_SIZE, 0, (z - 1) * CHUNK_SIZE], [(x + 2) * CHUNK_SIZE, CHUNK_HEIGHT, (z + 2) * CHUNK_SIZE]);
    renderer.render(viewProj, eye, now / 1000, fogDistance, buildLines(hit), meshes, draws, far?.ready ? {
      vertices: far.vertices, indices: far.indices, version: far.version, seaY: SEA_SURFACE, look: farLook as 'mist' | 'silhouette' | 'colour', extent: far.extent,
      // Safe mode draws chunks without fading them in.
      coverage: world.coverage(performance.now(), safe ? 0 : FADE_MS),
    } : undefined, [...animals.instances(animalFrames)].map(([name, instances]) => ({ model: animalModels.get(name)!, instances })));

    const saved = world.savedCount(), counts = world.counts();
    hud.textContent = [
      `fps ${fps.toFixed(0)}   gpu: ${gpuName}   world generation: ${worldgenWorker ? 'plain JS (worker)' : store.generate ? 'compute shader' : `TF.js on ${tfBackend}`}`,
      `pos ${px.toFixed(1)} ${py.toFixed(1)} ${pz.toFixed(1)}   chunk ${world.window.cx},${world.window.cz}`,
      `chunks: ${counts.inView} in view (${draws.length} drawn), ${counts.active} simulated (${world.awakeCount()} awake, ` +
        `${sim.growingChunks} growing plants)${saved ? `, ${saved} changed ones saved` : ''}`,
      store instanceof GpuStore
        ? `world: in GPU memory (${(store.bytes / 2 ** 20).toFixed(1)} MB, meshes ${meshes instanceof MeshPool ? (meshes.usage.used * 4 / 2 ** 20).toFixed(1) : '?'} MB); ` +
          `a tick reads back ${sim.lastBatch * 4} bytes of flags`
        : `world: on the CPU (${onCpu ? check.detail : `the GPU failed its check: ${check.detail}`})`,
      far ? `far terrain: ${farLook}, out to ${far.extent} blocks (${far.points.toLocaleString()} points, ${(far.bytes / 2 ** 20).toFixed(1)} MB)` : 'far terrain: off',
      `block updates: ${blockUpdateStatus()}`,
      `placing: ${BUILDING_BLOCKS[selected].name}   diamonds: ${diamonds}   animals: ${animals.animals.length}   [F] ${controls.flying ? 'flying' : controls.body.inFluid ? 'swimming' : 'walking'}   [M] music ${music.enabled ? 'on' : 'off'}   [N] sounds ${sounds.enabled ? 'on' : 'off'}   [G] chunk outlines ${showChunks ? 'on' : 'off'}   [P] pause updates`,
      ...(lastError ? [`error: ${lastError}`] : []),
    ].join('\n');
  };
  requestAnimationFrame(frame);
}

main().catch((e) => {
  logError('Failed to start', e);
  $('error').textContent = String(e instanceof Error ? e.message : e);
  $('status').textContent = 'Failed to start';
  log.open();
});
