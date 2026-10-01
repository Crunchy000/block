import { forward } from '../render/math';

/** Free-flying first-person camera with pointer lock. */
export class Controls {
  position: [number, number, number];
  yaw = Math.PI * 0.75;
  pitch = -0.35;
  speed = 12;
  locked = false;
  private keys = new Set<string>();
  /** Called when the browser refuses pointer lock (e.g. clicking again too soon after Esc). */
  onLockError?: (message: string) => void;
  /** Mouse buttons pressed since the last poll. */
  private clicks: number[] = [];
  private keyPresses: string[] = [];

  constructor(private readonly canvas: HTMLCanvasElement, start: [number, number, number]) {
    this.position = start;
    // Any click starts play: the start screen overlays the canvas, so listen on the document.
    document.addEventListener('click', () => {
      if (!this.locked) this.requestLock();
    });
    document.addEventListener('pointerlockerror', () => this.onLockError?.('the browser refused.'));
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === canvas;
      if (!this.locked) this.keys.clear();
    });
    document.addEventListener('mousemove', (e) => {
      if (!this.locked) return;
      this.yaw -= e.movementX * 0.0025;
      this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch - e.movementY * 0.0025));
    });
    document.addEventListener('mousedown', (e) => {
      if (this.locked) this.clicks.push(e.button);
    });
    window.addEventListener('keydown', (e) => {
      if (!this.locked) return;
      this.keys.add(e.code);
      if (!e.repeat) this.keyPresses.push(e.code);
      if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
  }

  requestLock(): void {
    try {
      // A promise in current browsers (rejects e.g. within ~1s of leaving with Esc), undefined in older ones.
      const pending = this.canvas.requestPointerLock() as Promise<void> | undefined;
      pending?.catch((e: unknown) => this.onLockError?.(e instanceof Error ? e.message : String(e)));
    } catch (e) {
      this.onLockError?.(e instanceof Error ? e.message : String(e));
    }
  }

  takeClicks(): number[] {
    const c = this.clicks;
    this.clicks = [];
    return c;
  }

  takeKeyPresses(): string[] {
    const k = this.keyPresses;
    this.keyPresses = [];
    return k;
  }

  get look(): [number, number, number] {
    return forward(this.yaw, this.pitch);
  }

  update(dt: number): void {
    const k = this.keys;
    const fx = -Math.sin(this.yaw), fz = -Math.cos(this.yaw);
    const rx = Math.cos(this.yaw), rz = -Math.sin(this.yaw);
    let mx = 0, my = 0, mz = 0;
    if (k.has('KeyW')) { mx += fx; mz += fz; }
    if (k.has('KeyS')) { mx -= fx; mz -= fz; }
    if (k.has('KeyD')) { mx += rx; mz += rz; }
    if (k.has('KeyA')) { mx -= rx; mz -= rz; }
    if (k.has('Space')) my += 1;
    if (k.has('ShiftLeft') || k.has('ShiftRight')) my -= 1;
    const len = Math.hypot(mx, my, mz);
    if (len === 0) return;
    const s = (this.speed * (k.has('ControlLeft') ? 3 : 1) * dt) / len;
    this.position[0] += mx * s;
    this.position[1] += my * s;
    this.position[2] += mz * s;
  }
}
