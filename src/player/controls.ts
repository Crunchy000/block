import { forward } from '../render/math';
import { Button, GamepadInput } from './gamepad';
import { Body, EYE_HEIGHT } from './physics';

const MOUSE_LOOK_SPEED = 0.0025; // radians per mouse count
const MAX_PITCH = 1.55;
/** Turning with a gamepad's right stick pushed all the way, radians a second (sideways, up / down). */
const PAD_YAW_SPEED = 3.2;
const PAD_PITCH_SPEED = 2.2;

/** Sprinting while flying moves this many times faster. */
const SPRINT = 4;

/**
 * First-person player: walking, running and jumping with collisions (player/physics.ts),
 * or flying freely through everything. Played with mouse + keyboard (pointer lock), on touch
 * screens through TouchControls (which feeds the touch* fields), or with a gamepad
 * (player/gamepad.ts, read each update).
 */
export class Controls {
  position: [number, number, number];
  yaw = Math.PI * 0.75;
  pitch = -0.35;
  /** Flying speed, blocks a second; sprinting (Ctrl, or a double-tapped stick) is SPRINT times that. */
  speed = 20;
  /** Flying (free, through blocks) rather than walking. */
  flying = false;
  readonly body = new Body();
  /** Mouse captured (pointer lock). */
  locked = false;
  /** Playing with on-screen touch controls instead of pointer lock. */
  touchPlaying = false;
  /** Playing with a gamepad (started from it: no pointer lock needed). */
  padPlaying = false;
  readonly gamepad = new GamepadInput();
  /** The gamepad's walking, flying up / down, digging and running (clicked left stick, until it's let go). */
  private padMove = { right: 0, forward: 0 };
  private padVertical = 0;
  private padDig = false;
  private padRun = false;

  /** Touch stick, analog: strafe right / forward in [-1, 1]. */
  touchMove = { right: 0, forward: 0 };
  touchSprint = false;
  /** Touch Break button held. */
  touchDig = false;
  /** Touch fly buttons: +1 up, -1 down. */
  touchVertical = 0;

  /** Called when the browser refuses pointer lock (e.g. clicking again too soon after Esc). */
  onLockError?: (message: string) => void;
  /** Called when play starts or stops (pointer lock, touch or gamepad). */
  onPlayingChange?: (playing: boolean) => void;
  /** Called when a gamepad button starts play (as a click on the start screen would: start the sound). */
  onGamepadStart?: () => void;

  private keys = new Set<string>();
  /** Left mouse button held (with the mouse captured). */
  private mouseDig = false;
  /** Running from a double-tapped W (until W is let go), as in Minecraft: Ctrl+W closes the tab in browsers. */
  private runLatch = false;
  private lastWUp = -Infinity;
  /** Mouse buttons pressed since the last poll (touch break/place buttons push 0 / 2). */
  private clicks: number[] = [];
  private keyPresses: string[] = [];

  constructor(private readonly canvas: HTMLCanvasElement, start: [number, number, number]) {
    this.position = start;
    document.addEventListener('pointerlockerror', () => this.onLockError?.('the browser refused.'));
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === canvas;
      if (!this.locked) {
        this.keys.clear();
        this.mouseDig = false;
      }
      this.onPlayingChange?.(this.playing);
    });
    document.addEventListener('mousemove', (e) => {
      if (this.locked) this.rotate(e.movementX * MOUSE_LOOK_SPEED, e.movementY * MOUSE_LOOK_SPEED);
    });
    document.addEventListener('mousedown', (e) => {
      if (!this.locked) return;
      if (e.button === 0) this.mouseDig = true;
      else this.clicks.push(e.button);
    });
    document.addEventListener('mouseup', (e) => {
      if (e.button === 0) this.mouseDig = false;
    });
    window.addEventListener('keydown', (e) => {
      if (!this.playing) return;
      this.keys.add(e.code);
      if (!e.repeat) this.keyPresses.push(e.code);
      if (e.code === 'KeyW' && !e.repeat && e.timeStamp - this.lastWUp < 300) this.runLatch = true;
      if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => {
      this.keys.delete(e.code);
      if (e.code === 'KeyW') {
        this.runLatch = false;
        this.lastWUp = e.timeStamp;
      }
    });
  }

  /** Digging: the left mouse button, the touch Break button or the right trigger held down. */
  get digging(): boolean {
    return this.mouseDig || this.touchDig || this.padDig;
  }

  get playing(): boolean {
    return this.locked || this.touchPlaying || this.padPlaying;
  }

  /** Back to the start screen from however play was started. */
  stop(): void {
    if (this.locked) document.exitPointerLock();
    if (this.touchPlaying) this.stopTouch();
    if (this.padPlaying) {
      this.padPlaying = false;
      this.onPlayingChange?.(this.playing);
    }
  }

  /**
   * Read the gamepad: on the start screen any button starts play; while playing, the sticks walk
   * and look, the triggers dig and place, and the buttons jump, fly and pick hotbar blocks.
   */
  private pollGamepad(dt: number): void {
    const pads = typeof navigator !== 'undefined' && navigator.getGamepads ? navigator.getGamepads() : [];
    const pad = this.gamepad.read(pads);
    this.padMove = { right: 0, forward: 0 };
    this.padVertical = 0;
    this.padDig = false;
    if (!pad) return;
    if (!this.playing) {
      if (pad.pressed.size > 0) {
        this.padPlaying = true;
        this.onGamepadStart?.();
        this.onPlayingChange?.(this.playing);
      }
      return;
    }
    if (pad.pressed.has(Button.Start)) {
      this.stop();
      return;
    }
    this.padMove = pad.move;
    if (pad.pressed.has(Button.LeftStick)) this.padRun = true;
    if (pad.move.right === 0 && pad.move.forward === 0) this.padRun = false;
    this.rotate(pad.look.x * PAD_YAW_SPEED * dt, pad.look.y * PAD_PITCH_SPEED * dt);
    this.padVertical = (pad.held.has(Button.A) ? 1 : 0) - (pad.held.has(Button.B) ? 1 : 0);
    this.padDig = pad.held.has(Button.RT);
    if (pad.pressed.has(Button.LT)) this.clicks.push(2);
    if (pad.pressed.has(Button.Y)) this.keyPresses.push('KeyF');
    if (pad.pressed.has(Button.LB) || pad.pressed.has(Button.Left)) this.keyPresses.push('HotbarPrev');
    if (pad.pressed.has(Button.RB) || pad.pressed.has(Button.Right)) this.keyPresses.push('HotbarNext');
  }

  /** Start playing from the start screen: touch and pen get on-screen controls, a mouse gets pointer lock. */
  start(pointerType: string): void {
    if (pointerType === 'touch' || pointerType === 'pen') this.startTouch();
    else this.requestLock();
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

  startTouch(): void {
    this.touchPlaying = true;
    this.onPlayingChange?.(this.playing);
  }

  stopTouch(): void {
    this.touchPlaying = false;
    this.touchMove = { right: 0, forward: 0 };
    this.touchSprint = false;
    this.touchDig = false;
    this.touchVertical = 0;
    this.keys.clear();
    this.onPlayingChange?.(this.playing);
  }

  /** Turn the camera: positive yaw turns right, positive pitch looks down. */
  rotate(dYaw: number, dPitch: number): void {
    this.yaw -= dYaw;
    this.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, this.pitch - dPitch));
  }

  /** Queue a place (2) action, as from the right mouse button. */
  pushClick(button: number): void {
    this.clicks.push(button);
  }

  /** Queue a key press (e.g. 'Digit3', 'KeyG') from an on-screen control. */
  pushKey(code: string): void {
    this.keyPresses.push(code);
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

  /** Switch between walking and flying (F, or the touch fly button). */
  toggleFlying(): void {
    this.flying = !this.flying;
    this.body.velocity = [0, 0, 0];
  }

  /**
   * Move for `dt` seconds. Walking needs the blocks around the player (`typeAt`, from the
   * world's readBox); without them it waits.
   */
  update(dt: number, typeAt?: (x: number, y: number, z: number) => number): void {
    this.pollGamepad(dt);
    const k = this.keys;
    // Input in camera space: keys are digital, the touch and gamepad sticks analog.
    let right = this.touchMove.right + this.padMove.right, fwd = this.touchMove.forward + this.padMove.forward;
    let up = this.touchVertical + this.padVertical;
    if (k.has('KeyW')) fwd += 1;
    if (k.has('KeyS')) fwd -= 1;
    if (k.has('KeyD')) right += 1;
    if (k.has('KeyA')) right -= 1;
    if (k.has('Space')) up += 1;
    if (k.has('ShiftLeft') || k.has('ShiftRight')) up -= 1;
    const sprint = k.has('ControlLeft') || k.has('ControlRight') || this.runLatch || this.touchSprint || this.padRun;

    if (!this.flying) {
      if (!typeAt) return;
      const p = this.position, feet = [p[0], p[1] - EYE_HEIGHT, p[2]];
      this.body.step(feet, Math.min(dt, 0.1), this.yaw, { right, forward: fwd, jump: up > 0, run: sprint }, typeAt);
      this.position = [feet[0], feet[1] + EYE_HEIGHT, feet[2]];
      return;
    }
    // Full speed in any direction (diagonals aren't faster); a half-pushed stick moves at half speed.
    const len = Math.hypot(right, fwd, up);
    if (len === 0) return;
    const s = (this.speed * (sprint ? SPRINT : 1) * dt) / Math.max(1, len);
    const sin = Math.sin(this.yaw), cos = Math.cos(this.yaw);
    // forward = (-sin, 0, -cos), right = (cos, 0, -sin)
    this.position[0] += (-sin * fwd + cos * right) * s;
    this.position[1] += up * s;
    this.position[2] += (-cos * fwd - sin * right) * s;
  }
}
