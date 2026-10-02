import type { Controls } from './controls';
import { readStick } from './stick';

const STICK_RADIUS = 56;   // CSS px the knob can travel from where the thumb landed
const DEAD_ZONE = 0.12;    // fraction of the radius that reads as no movement
const LOOK_SPEED = 0.006;  // radians per CSS px dragged on the look side
const SPRINT_TAP_MS = 300; // touching the stick again this soon after letting go sprints

/**
 * On-screen controls for touch screens:
 * - left half: a dynamic movement stick that appears wherever the thumb lands
 *   (double-tap and hold to sprint)
 * - right half: drag to look around
 * - buttons for break / place and fly up / down, plus a small toolbar
 *
 * Every zone and button captures its own pointer, so several fingers work at once.
 */
export class TouchControls {
  private readonly root: HTMLElement;
  private readonly stick: HTMLElement;
  private readonly knob: HTMLElement;
  private readonly toggles = new Map<string, HTMLButtonElement>();
  private movePointer: number | null = null;
  private moveOrigin = { x: 0, y: 0 };
  private lastMoveEnd = -Infinity;
  private lookPointer: number | null = null;
  private lookLast = { x: 0, y: 0 };
  private flyUp = false;
  private flyDown = false;
  /** Release callbacks for buttons currently held down. */
  private held = new Set<() => void>();

  constructor(private readonly controls: Controls, parent: HTMLElement = document.body) {
    this.root = el('div', 'touch-ui');
    this.root.hidden = true;
    this.root.addEventListener('contextmenu', (e) => e.preventDefault());

    const moveZone = el('div', 'touch-zone touch-move');
    const lookZone = el('div', 'touch-zone touch-look');
    moveZone.append(el('span', 'touch-hint', 'drag to move'));
    lookZone.append(el('span', 'touch-hint', 'drag to look'));
    this.stick = el('div', 'stick');
    this.knob = el('div', 'knob');
    this.stick.append(this.knob);
    this.stick.style.setProperty('--radius', `${STICK_RADIUS}px`);
    this.stick.hidden = true;

    const actions = el('div', 'touch-actions');
    actions.append(
      this.holdButton('Break', 'break', 'Dig (hold)', () => { this.controls.touchDig = true; }, () => { this.controls.touchDig = false; }),
      this.holdButton('Place', 'place', 'Place block', () => this.controls.pushClick(2)),
      this.holdButton('▲', 'fly-up', 'Jump, swim or fly up', () => this.setFly('up', true), () => this.setFly('up', false)),
      this.holdButton('▼', 'fly-down', 'Fly down (when flying)', () => this.setFly('down', true), () => this.setFly('down', false)),
    );

    const toolbar = el('div', 'touch-toolbar');
    toolbar.append(
      this.toggleButton('Fly', 'KeyF', 'Fly through blocks, or walk'),
      this.toggleButton('Outlines', 'KeyG', 'Toggle chunk and ghost-halo outlines'),
      this.toggleButton('Pause', 'KeyP', 'Pause block updates'),
      this.tapButton('Menu', 'Show the start screen', () => this.controls.stopTouch()),
    );

    this.root.append(moveZone, lookZone, this.stick, actions, toolbar);
    parent.append(this.root);
    this.wireMoveZone(moveZone);
    this.wireLookZone(lookZone);
  }

  setVisible(visible: boolean): void {
    this.root.hidden = !visible;
    if (!visible) this.releaseAll();
  }

  /** Reflect toggle state (e.g. changed from the keyboard) on the toolbar buttons. */
  setToggle(key: string, on: boolean): void {
    this.toggles.get(key)?.setAttribute('aria-pressed', String(on));
  }

  private wireMoveZone(zone: HTMLElement): void {
    zone.addEventListener('pointerdown', (e) => {
      if (this.movePointer !== null) return;
      e.preventDefault();
      zone.setPointerCapture(e.pointerId);
      this.movePointer = e.pointerId;
      this.moveOrigin = { x: e.clientX, y: e.clientY };
      this.controls.touchSprint = performance.now() - this.lastMoveEnd < SPRINT_TAP_MS;
      this.stick.classList.toggle('sprint', this.controls.touchSprint);
      this.stick.style.left = `${e.clientX}px`;
      this.stick.style.top = `${e.clientY}px`;
      this.stick.hidden = false;
      this.root.classList.add('moved');
      this.updateStick(e);
    });
    zone.addEventListener('pointermove', (e) => {
      if (e.pointerId === this.movePointer) this.updateStick(e);
    });
    const end = (e: PointerEvent) => {
      if (e.pointerId !== this.movePointer) return;
      this.movePointer = null;
      this.lastMoveEnd = performance.now();
      this.stopMoving();
    };
    zone.addEventListener('pointerup', end);
    zone.addEventListener('pointercancel', end);
    zone.addEventListener('lostpointercapture', end);
  }

  private updateStick(e: PointerEvent): void {
    const r = readStick(e.clientX - this.moveOrigin.x, e.clientY - this.moveOrigin.y, STICK_RADIUS, DEAD_ZONE);
    this.controls.touchMove = { right: r.right, forward: r.forward };
    this.knob.style.transform = `translate(${r.knobX}px, ${r.knobY}px)`;
  }

  private stopMoving(): void {
    this.controls.touchMove = { right: 0, forward: 0 };
    this.controls.touchSprint = false;
    this.stick.hidden = true;
  }

  private wireLookZone(zone: HTMLElement): void {
    zone.addEventListener('pointerdown', (e) => {
      if (this.lookPointer !== null) return;
      e.preventDefault();
      zone.setPointerCapture(e.pointerId);
      this.lookPointer = e.pointerId;
      this.lookLast = { x: e.clientX, y: e.clientY };
      this.root.classList.add('looked');
    });
    zone.addEventListener('pointermove', (e) => {
      if (e.pointerId !== this.lookPointer) return;
      this.controls.rotate((e.clientX - this.lookLast.x) * LOOK_SPEED, (e.clientY - this.lookLast.y) * LOOK_SPEED);
      this.lookLast = { x: e.clientX, y: e.clientY };
    });
    const end = (e: PointerEvent) => {
      if (e.pointerId === this.lookPointer) this.lookPointer = null;
    };
    zone.addEventListener('pointerup', end);
    zone.addEventListener('pointercancel', end);
    zone.addEventListener('lostpointercapture', end);
  }

  private setFly(direction: 'up' | 'down', held: boolean): void {
    if (direction === 'up') this.flyUp = held;
    else this.flyDown = held;
    this.controls.touchVertical = (this.flyUp ? 1 : 0) - (this.flyDown ? 1 : 0);
  }

  /**
   * A button that acts while held. `press` may return an interval id to clear on release.
   */
  private holdButton(
    label: string, name: string, title: string, press: () => number | void, release?: () => void,
  ): HTMLButtonElement {
    const button = el('button', `touch-button ${name}`, label);
    button.setAttribute('aria-label', title);
    let pointer: number | null = null;
    let timer: number | undefined;
    const cancel = () => {
      pointer = null;
      button.classList.remove('pressed');
      window.clearInterval(timer);
      timer = undefined;
      this.held.delete(cancel);
      release?.();
    };
    button.addEventListener('pointerdown', (e) => {
      if (pointer !== null) return;
      e.preventDefault();
      button.setPointerCapture(e.pointerId);
      pointer = e.pointerId;
      button.classList.add('pressed');
      this.held.add(cancel);
      const t = press();
      if (typeof t === 'number') timer = t;
    });
    const end = (e: PointerEvent) => {
      if (e.pointerId === pointer) cancel();
    };
    button.addEventListener('pointerup', end);
    button.addEventListener('pointercancel', end);
    button.addEventListener('lostpointercapture', end);
    return button;
  }

  private tapButton(label: string, title: string, onTap: () => void): HTMLButtonElement {
    const button = el('button', 'touch-tool', label);
    button.setAttribute('aria-label', title);
    button.addEventListener('pointerdown', (e) => e.preventDefault());
    button.addEventListener('click', onTap);
    return button;
  }

  private toggleButton(label: string, key: string, title: string): HTMLButtonElement {
    const button = this.tapButton(label, title, () => this.controls.pushKey(key));
    button.setAttribute('aria-pressed', 'false');
    this.toggles.set(key, button);
    return button;
  }

  private releaseAll(): void {
    this.movePointer = null;
    this.lookPointer = null;
    this.stopMoving();
    for (const cancel of [...this.held]) cancel();
  }
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = className;
  if (text) e.textContent = text;
  return e;
}
