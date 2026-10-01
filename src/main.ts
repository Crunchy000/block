import * as tf from '@tensorflow/tfjs';
import {
  ACTIVE_RADIUS, Block, CHUNK_HEIGHT, CHUNK_SIZE, DEFAULT_RATES, SOURCE_LEVEL, BLOCK_NAMES, cell,
} from './constants';
import { Controls } from './player/controls';
import { TouchControls } from './player/touchControls';
import type { RayHit } from './player/raycast';
import { fpsView, multiply, perspective } from './render/math';
import { MeshPool } from './render/meshPool';
import { Renderer } from './render/renderer';
import { checkGpuStore } from './sim/check';
import { CpuStore } from './sim/cpuStore';
import { GpuStore } from './sim/gpuStore';
import { Simulation } from './sim/simulation';
import { ringSize, type CellStore } from './sim/store';
import { fallbackBackend, initTensorflow, warmUpKernels } from './tf/backend';
import { HOTBAR_BLOCKS, createHotbar } from './ui/hotbar';
import { generateMissing } from './world/loader';
import { World, meshSlotCount } from './world/world';

const TICK_MS = 200;          // block-update rate (5 ticks / second)
const PICK_DISTANCE = 8;      // blocks

const $ = (id: string) => document.getElementById(id)!;

async function main(): Promise<void> {
  const canvas = $('gpu') as HTMLCanvasElement;
  const overlay = $('overlay'), hud = $('hud'), errorBox = $('error');
  let statusText = '';
  const setStatus = (text: string) => {
    if (text !== statusText) $('status').textContent = statusText = text;
  };
  let lastError = '';
  const showError = (text: string) => {
    lastError = text;
    errorBox.textContent = text;
  };

  setStatus('Starting WebGPU…');
  const params = new URLSearchParams(location.search);
  const renderer = await Renderer.create(canvas, { offscreen: params.has('offscreen') });
  setStatus('Starting TensorFlow.js…');
  let tfBackend = await initTensorflow(renderer.device, renderer.adapterInfo);
  setStatus(`Compiling GPU kernels (${tfBackend})…`);
  try {
    await warmUpKernels();
  } catch (e) {
    console.warn('kernel warm-up failed; kernels will compile on first use', e);
  }
  const { device } = renderer;
  const activeRadius = params.has('radius') ? Math.max(1, Math.floor(Number(params.get('radius')))) : ACTIVE_RADIUS;
  const ghostRadius = activeRadius + 1;
  // The world lives in GPU memory, where block updates, meshing and picking run. First
  // check that this GPU computes them exactly as the reference code does; if it doesn't,
  // the world lives on the CPU with the reference code instead (slower, same game).
  setStatus('Checking the GPU world code…');
  const check = await checkGpuStore(device).catch((e: unknown) => ({ ok: false, summary: '', detail: String(e) }));
  if (!check.ok) console.warn('The GPU world code disagrees with the reference; keeping the world on the CPU.', check.detail);
  const store: CellStore = check.ok && !params.has('cpu')
    ? await GpuStore.create(device, ringSize(ghostRadius))
    : new CpuStore(ringSize(ghostRadius));
  const meshes = new MeshPool(device, meshSlotCount(activeRadius));

  // If the TF backend breaks at runtime (e.g. a driver limit), drop to the next one.
  let switching = false;
  const onTfError = (what: string) => (e: unknown) => {
    console.error(`${what} failed on tfjs backend ${tfBackend}`, e);
    showError(`${what} failed on ${tfBackend}: ${e instanceof Error ? e.message : String(e)}`);
    if (switching) return;
    switching = true;
    fallbackBackend()
      .then(async (name) => {
        if (name !== tfBackend) showError(`${lastError} (switched TensorFlow.js to ${name})`);
        tfBackend = name;
        await warmUpKernels().catch(() => {});
      })
      .finally(() => { switching = false; });
  };

  const world = new World(store, activeRadius, ghostRadius);
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
  $('bench-link').addEventListener('click', (e) => e.stopPropagation());
  let startPointer = 'mouse';
  overlay.addEventListener('pointerdown', (e) => { startPointer = e.pointerType; });
  overlay.addEventListener('click', (e) => {
    const type = (e as PointerEvent).pointerType || startPointer;
    if ((type === 'touch' || type === 'pen') && document.fullscreenEnabled && !document.fullscreenElement) {
      // More room on phones, and no accidental pull-to-refresh. Unsupported on iPhone; that's fine.
      document.documentElement.requestFullscreen({ navigationUI: 'hide' }).catch(() => {});
    }
    controls.start(type);
  });
  const touchFirst = matchMedia('(pointer: coarse)').matches;

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
      .then(() => device.queue.onSubmittedWorkDone())
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
        if (d > world.ghostRadius) continue;
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
        console.error('block update failed', e);
        showError(`block update failed: ${e instanceof Error ? e.message : String(e)}`);
      });
    }
    world.remesh(meshes);

    const { loaded, total } = world.haloProgress();
    setStatus(loaded < total ? `Generating world… ${loaded} / ${total} chunks` : `${touchFirst ? 'Tap' : 'Click'} anywhere to play`);

    const fogDistance = world.activeRadius * CHUNK_SIZE + 8;
    const proj = perspective((70 * Math.PI) / 180, renderer.aspect, 0.1, fogDistance * 1.5);
    const viewProj = multiply(proj, fpsView(eye, controls.yaw, controls.pitch));
    renderer.render(viewProj, eye, now / 1000, fogDistance, buildLines(hit), meshes, world.draws());

    const saved = world.savedCount();
    hud.textContent = [
      `fps ${fps.toFixed(0)}   gpu: ${gpuName}   world generation: TF.js on ${tfBackend}`,
      `pos ${px.toFixed(1)} ${py.toFixed(1)} ${pz.toFixed(1)}   chunk ${world.window.cx},${world.window.cz}`,
      `chunks: ${world.activeChunks().length} active (${world.awakeCount()} awake, ${sim.growingChunks} growing plants), ` +
        `${world.ghostChunks().length} ghost (halo)${saved ? `, ${saved} changed ones saved` : ''}`,
      store instanceof GpuStore
        ? `world: in GPU memory (${(store.bytes / 2 ** 20).toFixed(1)} MB); a tick reads back ${sim.lastBatch * 4} bytes of flags`
        : `world: on the CPU (${check.ok ? '?cpu' : `the GPU failed its check: ${check.detail}`})`,
      `block updates: ${blockUpdateStatus()}`,
      `placing: ${BLOCK_NAMES[selected]}   [G] chunk outlines ${showChunks ? 'on' : 'off'}   [P] pause updates`,
      ...(lastError ? [`error: ${lastError}`] : []),
    ].join('\n');
  };
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  $('error').textContent = String(e instanceof Error ? e.message : e);
  $('status').textContent = 'Failed to start';
});
