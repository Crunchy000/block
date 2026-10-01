import * as tf from '@tensorflow/tfjs';
import {
  ACTIVE_RADIUS, Block, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_VOLUME, DEFAULT_RATES, SOURCE_LEVEL, BLOCK_NAMES, cell,
} from './constants';
import { log, logError } from './log';
import { Controls } from './player/controls';
import { TouchControls } from './player/touchControls';
import type { RayHit } from './player/raycast';
import { fpsView, frustum, multiply, perspective } from './render/math';
import { ClassicMeshes } from './render/classicMeshes';
import { MeshPool } from './render/meshPool';
import { Renderer } from './render/renderer';
import { checkGpuStore } from './sim/check';
import { CpuStore } from './sim/cpuStore';
import { GpuStore } from './sim/gpuStore';
import { Simulation } from './sim/simulation';
import { ringSize, type CellStore } from './sim/store';
import { fallbackBackend, initTensorflow, warmUpKernels } from './tf/backend';
import { HOTBAR_BLOCKS, createHotbar } from './ui/hotbar';
import { FarTerrain, SEA_SURFACE } from './world/farTerrain';
import { generateMissing } from './world/loader';
import { World, meshSlotCount } from './world/world';

const TICK_MS = 200;          // block-update rate (5 ticks / second)
const PICK_DISTANCE = 8;      // blocks

const $ = (id: string) => document.getElementById(id)!;

/** View distances offered on the start screen (chunks from the player's chunk). */
const VIEW_DISTANCES = [3, 4, 8, 16, 32, 64];
const VIEW_KEY = 'block.viewDistance';
const FAR_KEY = 'block.farTerrain';
/** Block updates run this far out at most; beyond it chunks are drawn but frozen. */
const MAX_SIMULATION_DISTANCE = 8;

/**
 * The furthest view distance this GPU can hold: the cells of every loaded chunk (the view
 * plus a ring) are one GPU buffer of 64 KB a chunk. Safe mode builds meshes on the CPU,
 * which is slow for big areas.
 */
function maxViewDistance(device: GPUDevice, safe: boolean): number {
  if (safe) return 8;
  const bytes = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const ring = Math.floor(Math.sqrt(bytes / (CHUNK_VOLUME * 4)));
  return Math.max(1, Math.floor((ring - 1) / 2) - 1);
}

/** ?radius=N, else the one picked on the start screen, else 3 on phones and 8 elsewhere; within what the GPU holds. */
function viewDistance(params: URLSearchParams, device: GPUDevice, safe: boolean): number {
  let stored: string | null = null;
  try { stored = localStorage.getItem(VIEW_KEY); } catch { /* storage blocked */ }
  const asked = Number(params.get('radius') ?? stored ?? (matchMedia('(pointer: coarse)').matches ? ACTIVE_RADIUS : 8));
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
  const limit = safe ? ` (up to ${max} in safe mode)` : ` (this GPU holds up to ${max})`;
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
  setStatus('Starting WebGPU…');
  const params = new URLSearchParams(location.search);
  // Safe mode (?safe): the world on the CPU and the previous renderer, for GPUs (some phones)
  // that crash on the GPU world or its renderer.
  const safe = params.has('safe');
  if (safe) log.info('Safe mode: world on the CPU, meshes built on the CPU, plain draws');
  const renderer = await Renderer.create(canvas, { offscreen: params.has('offscreen'), safe });
  setStatus('Starting TensorFlow.js…');
  let tfBackend = await initTensorflow(renderer.device, renderer.adapterInfo);
  log.info(`TensorFlow.js backend: ${tfBackend}`);
  setStatus(`Compiling GPU kernels (${tfBackend})…`);
  try {
    await warmUpKernels();
  } catch (e) {
    logError('Kernel warm-up failed (kernels will compile on first use)', e);
  }
  const { device } = renderer;
  const viewRadius = viewDistance(params, device, safe);
  const activeRadius = Math.min(viewRadius, MAX_SIMULATION_DISTANCE);
  const ghostRadius = viewRadius + 1;
  log.info(`View distance ${viewRadius} (${(2 * viewRadius + 1) ** 2} chunks drawn), simulation distance ${activeRadius}`);
  showViewDistances(viewRadius, maxViewDistance(device, safe), safe);
  const farLook = farStyle(params);
  showFarTerrain(farLook);
  // The world lives in GPU memory, where block updates, meshing and picking run. First
  // check that this GPU computes them exactly as the reference code does; if it doesn't,
  // the world lives on the CPU with the reference code instead (slower, same game).
  const onCpu = safe || params.has('cpu');
  if (!onCpu) setStatus('Checking the GPU world code…');
  const check = onCpu ? { ok: false, summary: '', detail: safe ? 'safe mode' : '?cpu' }
    : await checkGpuStore(device).catch((e: unknown) => ({ ok: false, summary: '', detail: log.describe(e).join('\n') }));
  if (check.ok) log.info('GPU world check passed', check.summary);
  else if (!onCpu) log.warn('GPU world check failed: the world stays on the CPU', check.detail);
  setStatus('Setting up the world…');
  const store: CellStore = check.ok && !onCpu
    ? await GpuStore.create(device, ringSize(ghostRadius))
    : new CpuStore(ringSize(ghostRadius));
  log.info(store instanceof GpuStore ? 'World: in GPU memory' : 'World: on the CPU');
  const meshes = safe ? new ClassicMeshes(device, meshSlotCount(viewRadius)) : new MeshPool(device, meshSlotCount(viewRadius));
  // The land beyond the chunks, out to at least 2 km (1 km past the chunks at long view distances).
  const far = farLook !== 'off' ? new FarTerrain(device, Math.max(2048, (viewRadius + 1) * CHUNK_SIZE + 1024)) : undefined;
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
  // Optional URL params: ?pos=x,y,z&yaw=rad&pitch=rad&chunks (outlines on)&radius=N&spread=&grow=&offscreen&cpu
  const pos = (params.get('pos') ?? '8,52,8').split(',').map(Number) as [number, number, number];
  const controls = new Controls(canvas, pos);
  if (params.has('yaw')) controls.yaw = Number(params.get('yaw'));
  if (params.has('pitch')) controls.pitch = Number(params.get('pitch'));
  controls.onLockError = (message) => showError(`Couldn't capture the mouse: ${message} Click again to retry.`);
  const touchUI = new TouchControls(controls);
  controls.onPlayingChange = (playing) => {
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
  });
  const touchFirst = matchMedia('(pointer: coarse)').matches;

  // A lost GPU device (a GPU crash or hang, or the browser reclaiming it) can't be used again:
  // stop, explain on the start screen, and open the log (gpu.ts has logged the browser's reason).
  device.lost.then((info) => {
    gpuLost = true;
    if (document.pointerLockElement) document.exitPointerLock();
    if (controls.touchPlaying) controls.stopTouch();
    overlay.classList.remove('hidden');
    document.body.classList.remove('playing');
    touchUI.setVisible(false);
    setStatus('The GPU stopped working', false);
    showError(`WebGPU device lost (${info.reason}): ${(info.message || 'no details').replace(/\.$/, '')}. Tap or click to reload; `
      + 'if WebGPU is then unavailable, fully close and reopen the browser.');
    log.open();
  });

  // Console / automation handle, e.g. block.world.setCell(x, y, z, value).
  Object.assign(window, { block: { world, sim, controls, renderer, store, tf } });

  const info = renderer.adapterInfo;
  const gpuName = [info.vendor, info.architecture || info.device || info.description].filter(Boolean).join(' ') || 'unknown';

  let generating = false;
  let selected: Block = Block.Dirt;
  let showChunks = params.has('chunks');
  let paused = false;
  const hotbar = createHotbar((block) => { selected = block; hotbar.setSelected(block); });
  hotbar.setSelected(selected);
  touchUI.setToggle('KeyG', showChunks);
  let lastTick = 0;
  let last = performance.now();
  let fps = 0;

  // One batch of world generation at a time: TF.js writes it straight into the world's GPU
  // memory, and the next batch waits until the GPU has done this one.
  const pumpGeneration = () => {
    if (generating || switching || world.missingChunks(1).length === 0) return;
    generating = true;
    generateMissing(world)
      .then(async (n) => {
        await device.queue.onSubmittedWorkDone();
        // Each batch in the log, so a crash while generating shows how far it got.
        const { loaded, total } = world.haloProgress();
        log.info(`Generated ${n} chunks (${loaded} / ${total})`);
      })
      .catch(onTfError('worldgen'))
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

  const handleInput = () => {
    for (const key of controls.takeKeyPresses()) {
      const slot = /^Digit([1-9])$/.exec(key);
      if (slot && HOTBAR_BLOCKS[Number(slot[1]) - 1] !== undefined) {
        selected = HOTBAR_BLOCKS[Number(slot[1]) - 1];
        hotbar.setSelected(selected);
      }
      if (key === 'KeyG') touchUI.setToggle(key, showChunks = !showChunks);
      if (key === 'KeyP') touchUI.setToggle(key, paused = !paused);
    }
    for (const button of controls.takeClicks()) {
      if (!hit) continue;
      // y = 0 is bedrock: it holds up fluids at the bottom of the world.
      if (button === 0 && hit.block[1] > 0) world.setCell(...hit.block, cell(Block.Air));
      else if (button === 2) {
        const level = selected === Block.Water || selected === Block.Lava ? SOURCE_LEVEL : 0;
        world.setCell(...hit.before, cell(selected, level));
      } else continue;
      edits++;
      hit = null;
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
    return new Float32Array(out);
  };

  const blockUpdateStatus = () => {
    const batch = `${sim.lastBatch} chunk${sim.lastBatch === 1 ? '' : 's'}`;
    if (paused) return 'paused';
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

    controls.update(dt);
    const [px, py, pz] = controls.position;
    world.recenter(px, pz);
    pumpGeneration();

    const eye = controls.position;
    handleInput();
    pick();

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
    const { cx: wcx, cz: wcz } = world.window, vr = world.viewRadius;
    renderer.render(viewProj, eye, now / 1000, fogDistance, buildLines(hit), meshes, draws, far?.ready ? {
      vertex: far.vertex, index: far.index, indexCount: far.indexCount, seaY: SEA_SURFACE, look: farLook as 'mist' | 'silhouette' | 'colour', extent: far.extent,
      near: [(wcx - vr) * CHUNK_SIZE, (wcz - vr) * CHUNK_SIZE, (wcx + vr + 1) * CHUNK_SIZE, (wcz + vr + 1) * CHUNK_SIZE],
    } : undefined);

    const saved = world.savedCount(), counts = world.counts();
    hud.textContent = [
      `fps ${fps.toFixed(0)}   gpu: ${gpuName}   world generation: TF.js on ${tfBackend}`,
      `pos ${px.toFixed(1)} ${py.toFixed(1)} ${pz.toFixed(1)}   chunk ${world.window.cx},${world.window.cz}`,
      `chunks: ${counts.inView} in view (${draws.length} drawn), ${counts.active} simulated (${world.awakeCount()} awake, ` +
        `${sim.growingChunks} growing plants)${saved ? `, ${saved} changed ones saved` : ''}`,
      store instanceof GpuStore
        ? `world: in GPU memory (${(store.bytes / 2 ** 20).toFixed(1)} MB, meshes ${meshes instanceof MeshPool ? (meshes.usage.used * 4 / 2 ** 20).toFixed(1) : '?'} MB); ` +
          `a tick reads back ${sim.lastBatch * 4} bytes of flags`
        : `world: on the CPU (${onCpu ? check.detail : `the GPU failed its check: ${check.detail}`})`,
      far ? `far terrain: ${farLook}, out to ${far.extent} blocks (${far.points.toLocaleString()} points, ${(far.bytes / 2 ** 20).toFixed(1)} MB)` : 'far terrain: off',
      `block updates: ${blockUpdateStatus()}`,
      `placing: ${BLOCK_NAMES[selected]}   [G] chunk outlines ${showChunks ? 'on' : 'off'}   [P] pause updates`,
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
