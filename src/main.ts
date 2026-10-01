import * as tf from '@tensorflow/tfjs';
import { Block, CHUNK_HEIGHT, CHUNK_SIZE, SOURCE_LEVEL, BLOCK_NAMES, cell } from './constants';
import { Controls } from './player/controls';
import { raycast, type RayHit } from './player/raycast';
import { fpsView, multiply, perspective } from './render/math';
import { meshChunk } from './render/mesher';
import { Renderer } from './render/renderer';
import { fallbackBackend, initTensorflow } from './tf/backend';
import { Simulation } from './tf/simulation';
import { generateChunks } from './tf/worldgen';
import { World } from './world/world';

const TICK_MS = 200;          // block-update rate (5 ticks / second)
const GEN_BATCH = 16;         // chunks generated per TF batch
const MESH_BUDGET = 4;        // chunks meshed per frame

const $ = (id: string) => document.getElementById(id)!;

async function main(): Promise<void> {
  const canvas = $('gpu') as HTMLCanvasElement;
  const status = $('status'), overlay = $('overlay'), hud = $('hud');

  status.textContent = 'Starting WebGPU…';
  const params = new URLSearchParams(location.search);
  const renderer = await Renderer.create(canvas, { offscreen: params.has('offscreen') });
  status.textContent = 'Starting TensorFlow.js…';
  let tfBackend = await initTensorflow(renderer.device, renderer.adapterInfo);

  // If the TF backend breaks at runtime (e.g. a driver limit), drop to the next one.
  let switching = false;
  const onTfError = (what: string) => (e: unknown) => {
    console.error(`${what} failed on tfjs backend ${tfBackend}`, e);
    if (switching) return;
    switching = true;
    fallbackBackend().then((name) => { tfBackend = name; }).finally(() => { switching = false; });
  };

  const radius = params.has('radius') ? Math.max(1, Number(params.get('radius'))) : undefined;
  const world = radius ? new World(radius, radius + 1) : new World();
  const sim = new Simulation(world);
  // Optional URL params: ?pos=x,y,z&yaw=rad&pitch=rad&chunks (outlines on)&radius=N&offscreen
  const pos = (params.get('pos') ?? '8,52,8').split(',').map(Number) as [number, number, number];
  const controls = new Controls(canvas, pos);
  if (params.has('yaw')) controls.yaw = Number(params.get('yaw'));
  if (params.has('pitch')) controls.pitch = Number(params.get('pitch'));
  status.textContent = 'Click to play';
  document.addEventListener('pointerlockchange', () => overlay.classList.toggle('hidden', controls.locked));

  // Console / automation handle, e.g. block.world.setCell(x, y, z, value).
  Object.assign(window, { block: { world, sim, controls, renderer, tf } });

  let generating = false;
  let selected: Block = Block.Dirt;
  let showChunks = params.has('chunks');
  let paused = false;
  let lastTick = 0;
  let last = performance.now();
  let fps = 0;

  const pumpGeneration = () => {
    if (generating) return;
    const batch = world.missingChunks(GEN_BATCH);
    if (batch.length === 0) return;
    generating = true;
    world.markGenerating(batch);
    generateChunks(batch)
      .then((datas) => batch.forEach((c, i) => world.addGenerated(c, datas[i])))
      .catch((e) => {
        world.markGenerated(batch);
        onTfError('worldgen')(e);
      })
      .finally(() => { generating = false; });
  };

  const pumpMeshing = () => {
    // Drop GPU meshes for chunks that left the active area.
    for (const key of [...renderer.chunkKeys()]) {
      const c = world.chunks.get(key);
      if (!c || c.state !== 'active') renderer.removeChunk(key);
    }
    const { cx, cz } = world.window;
    const todo = world.activeChunks()
      .filter((c) => c.version !== c.meshedVersion
        // Wait for all 4 neighbours (active or ghost) so border faces cull correctly.
        && world.getChunk(c.cx + 1, c.cz) && world.getChunk(c.cx - 1, c.cz)
        && world.getChunk(c.cx, c.cz + 1) && world.getChunk(c.cx, c.cz - 1))
      .sort((a, b) => Math.hypot(a.cx - cx, a.cz - cz) - Math.hypot(b.cx - cx, b.cz - cz))
      .slice(0, MESH_BUDGET);
    for (const c of todo) {
      const version = c.version;
      const mesh = meshChunk(world, c.cx, c.cz);
      renderer.setChunkMesh(c.key, mesh, [c.cx * CHUNK_SIZE + 8, CHUNK_HEIGHT / 2, c.cz * CHUNK_SIZE + 8]);
      c.meshedVersion = version;
    }
  };

  const handleInput = (hit: RayHit | null) => {
    for (const key of controls.takeKeyPresses()) {
      if (key === 'Digit1') selected = Block.Dirt;
      if (key === 'Digit2') selected = Block.Stone;
      if (key === 'Digit3') selected = Block.Water;
      if (key === 'Digit4') selected = Block.Lava;
      if (key === 'KeyG') showChunks = !showChunks;
      if (key === 'KeyP') paused = !paused;
    }
    for (const button of controls.takeClicks()) {
      if (!hit) continue;
      if (button === 0) world.setCell(...hit.block, cell(Block.Air));
      if (button === 2) {
        const level = selected === Block.Water || selected === Block.Lava ? SOURCE_LEVEL : 0;
        world.setCell(...hit.before, cell(selected, level));
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
        if (d > world.ghostRadius) continue;
        const col = c.state === 'active' ? [0.2, 1, 0.3] : [1, 0.55, 0.1];
        const x0 = c.cx * CHUNK_SIZE + 0.05, z0 = c.cz * CHUNK_SIZE + 0.05;
        box(x0, 0.05, z0, x0 + CHUNK_SIZE - 0.1, CHUNK_HEIGHT - 0.05, z0 + CHUNK_SIZE - 0.1, col);
      }
    }
    return new Float32Array(out);
  };

  const frame = (now: number) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    fps = fps * 0.95 + (dt > 0 ? 1 / dt : 0) * 0.05;

    controls.update(dt);
    const [px, py, pz] = controls.position;
    world.recenter(px, pz);
    pumpGeneration();

    const eye = controls.position;
    const hit = raycast((x, y, z) => world.getBlock(x, y, z), eye, controls.look, 8);
    handleInput(hit);

    if (!paused && now - lastTick >= TICK_MS && !world.locked) {
      lastTick = now;
      sim.tick().catch(onTfError('block update'));
    }
    pumpMeshing();

    const fogDistance = world.activeRadius * CHUNK_SIZE + 8;
    const proj = perspective((70 * Math.PI) / 180, renderer.aspect, 0.1, fogDistance * 1.5);
    const viewProj = multiply(proj, fpsView(eye, controls.yaw, controls.pitch));
    renderer.render(viewProj, eye, now / 1000, fogDistance, buildLines(hit));

    hud.textContent = [
      `fps ${fps.toFixed(0)}   tf backend: ${tfBackend}`,
      `pos ${px.toFixed(1)} ${py.toFixed(1)} ${pz.toFixed(1)}   chunk ${world.window.cx},${world.window.cz}`,
      `chunks: ${world.activeChunks().length} active, ${world.ghostChunks().length} ghost (halo)`,
      `block updates: tick ${sim.ticks}  ${sim.lastTickMs.toFixed(0)} ms  ${sim.lastChangedChunks} chunks changed${paused ? '  [PAUSED]' : ''}`,
      `placing: ${BLOCK_NAMES[selected]}   [G] chunk outlines ${showChunks ? 'on' : 'off'}`,
    ].join('\n');

    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  $('error').textContent = String(e instanceof Error ? e.message : e);
  $('status').textContent = 'Failed to start';
});
