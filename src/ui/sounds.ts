import { Block } from '../constants';

/**
 * Sound effects, through the Web Audio API. Animal calls are short recordings
 * (public/sounds/<animal>.mp3); everything else is made on the spot from noise and tones,
 * so there's nothing to download or credit: digging, breaking and placing blocks,
 * footsteps, landing, splashing into water, and picking up a diamond.
 *
 * Sounds at a place in the world get quieter with distance and pan left or right of where
 * the player faces. Browsers only let a page start sound from a click or tap, so the audio
 * starts with the one that starts play (unlock). On or off is remembered between visits.
 */
const SOUNDS_KEY = 'block.sounds';
/** Recorded calls, per animal (the horse borrows the cow's, higher). */
const CALLS: Record<string, { file: string; rate: number }> = {
  cow: { file: 'cow', rate: 1 },
  pig: { file: 'pig', rate: 1 },
  sheep: { file: 'sheep', rate: 1 },
  chicken: { file: 'chicken', rate: 1 },
  horse: { file: 'cow', rate: 1.45 },
};
/** Sounds further away than this aren't heard. */
const HEARING = 24;

export interface Listener {
  position: readonly number[];
  yaw: number;
}

export class Sounds {
  enabled: boolean;
  private ctx?: AudioContext;
  private out?: GainNode;
  private noise?: AudioBuffer;
  private readonly samples = new Map<string, AudioBuffer>();
  private listener: Listener = { position: [0, 0, 0], yaw: 0 };

  constructor() {
    let stored: string | null = null;
    try { stored = localStorage.getItem(SOUNDS_KEY); } catch { /* storage blocked */ }
    this.enabled = stored !== '0';
  }

  /** Start the audio (call from a click or tap) and load the recordings. */
  unlock(): void {
    if (!this.ctx) {
      const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctx) return;
      this.ctx = new Ctx();
      this.out = this.ctx.createGain();
      this.out.gain.value = 0.8;
      this.out.connect(this.ctx.destination);
      const n = this.ctx.sampleRate;
      this.noise = this.ctx.createBuffer(1, n, n);
      const d = this.noise.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
      for (const file of new Set(Object.values(CALLS).map((c) => c.file))) {
        fetch(new URL(`sounds/${file}.mp3`, document.baseURI))
          .then((r) => r.arrayBuffer())
          .then((data) => this.ctx!.decodeAudioData(data))
          .then((buffer) => this.samples.set(file, buffer))
          .catch(() => { /* that animal stays quiet */ });
      }
    }
    void this.ctx.resume().catch(() => {});
  }

  /** Switch sound effects on or off (N, or the touch Sounds button); remembered. */
  toggle(): boolean {
    this.enabled = !this.enabled;
    try { localStorage.setItem(SOUNDS_KEY, this.enabled ? '1' : '0'); } catch { /* storage blocked */ }
    return this.enabled;
  }

  /** Where the player is and faces (for placing sounds left and right). */
  setListener(listener: Listener): void {
    this.listener = listener;
  }

  get ready(): boolean {
    return this.enabled && this.ctx?.state === 'running';
  }

  /** Gain and pan for a sound at `at` (or right here); undefined when too far to hear. */
  private place(at?: readonly number[]): { gain: number; pan: number } | undefined {
    if (!at) return { gain: 1, pan: 0 };
    const p = this.listener.position;
    const dx = at[0] - p[0], dy = at[1] - p[1], dz = at[2] - p[2], d = Math.hypot(dx, dy, dz);
    if (d > HEARING) return undefined;
    const right = [Math.cos(this.listener.yaw), -Math.sin(this.listener.yaw)];
    const pan = d > 0.5 ? (dx * right[0] + dz * right[1]) / d : 0;
    return { gain: (1 - d / HEARING) ** 2, pan: pan * 0.8 };
  }

  /** A gain node (with pan) for a sound starting now, connected to the output. */
  private chain(volume: number, at?: readonly number[]): GainNode | undefined {
    if (!this.ready) return undefined;
    const placed = this.place(at);
    if (!placed) return undefined;
    const ctx = this.ctx!;
    const gain = ctx.createGain();
    gain.gain.value = volume * placed.gain;
    if (placed.pan !== 0 && ctx.createStereoPanner) {
      const pan = ctx.createStereoPanner();
      pan.pan.value = placed.pan;
      gain.connect(pan).connect(this.out!);
    } else {
      gain.connect(this.out!);
    }
    return gain;
  }

  /** A burst of filtered noise with a quick attack and an exponential decay. */
  private burst(o: { volume: number; seconds: number; filter: BiquadFilterType; freq: number; q?: number; sweepTo?: number; at?: readonly number[] }): void {
    const g = this.chain(o.volume, o.at);
    if (!g) return;
    const ctx = this.ctx!, t = ctx.currentTime;
    const src = ctx.createBufferSource();
    src.buffer = this.noise!;
    const f = ctx.createBiquadFilter();
    f.type = o.filter;
    f.frequency.setValueAtTime(o.freq, t);
    if (o.sweepTo) f.frequency.exponentialRampToValueAtTime(o.sweepTo, t + o.seconds);
    f.Q.value = o.q ?? 1;
    const env = ctx.createGain();
    env.gain.setValueAtTime(0.0001, t);
    env.gain.exponentialRampToValueAtTime(1, t + 0.005);
    env.gain.exponentialRampToValueAtTime(0.0001, t + o.seconds);
    src.connect(f).connect(env).connect(g);
    src.start(t, Math.random() * 0.5, o.seconds + 0.05);
  }

  /** A tone gliding from one pitch to another, fading out. */
  private tone(o: { volume: number; seconds: number; from: number; to: number; type?: OscillatorType; delay?: number; at?: readonly number[] }): void {
    const g = this.chain(o.volume, o.at);
    if (!g) return;
    const ctx = this.ctx!, t = ctx.currentTime + (o.delay ?? 0);
    const osc = ctx.createOscillator();
    osc.type = o.type ?? 'sine';
    osc.frequency.setValueAtTime(o.from, t);
    osc.frequency.exponentialRampToValueAtTime(o.to, t + o.seconds);
    const env = ctx.createGain();
    env.gain.setValueAtTime(0.0001, t);
    env.gain.exponentialRampToValueAtTime(1, t + 0.006);
    env.gain.exponentialRampToValueAtTime(0.0001, t + o.seconds);
    osc.connect(env).connect(g);
    osc.start(t);
    osc.stop(t + o.seconds + 0.02);
  }

  /** One scrape of digging at a block (repeated while digging). */
  dig(type: Block, at: readonly number[]): void {
    const hard = type === Block.Stone || type === Block.Diamond;
    this.burst({ volume: 0.35, seconds: 0.07, filter: hard ? 'bandpass' : 'lowpass', freq: hard ? 2400 + Math.random() * 600 : 900 + Math.random() * 300, q: hard ? 1.6 : 0.7, at });
    if (type === Block.Diamond) this.tone({ volume: 0.05, seconds: 0.08, from: 2600, to: 2400, type: 'triangle', at });
  }

  /** A block breaking: a crunch and a thump. */
  breakBlock(type: Block, at: readonly number[]): void {
    const hard = type === Block.Stone || type === Block.Diamond;
    this.burst({ volume: 0.5, seconds: 0.18, filter: 'lowpass', freq: hard ? 3000 : 1400, sweepTo: 300, at });
    this.tone({ volume: 0.35, seconds: 0.12, from: hard ? 160 : 120, to: 60, at });
  }

  /** A block placed: a soft thud. */
  placeBlock(at: readonly number[]): void {
    this.tone({ volume: 0.4, seconds: 0.09, from: 190, to: 85, at });
    this.burst({ volume: 0.2, seconds: 0.06, filter: 'lowpass', freq: 700, at });
  }

  /** A footstep on a block of `type` (grass and dirt soft, stone sharper). */
  step(type: Block): void {
    const hard = type === Block.Stone || type === Block.Diamond;
    if (hard) this.burst({ volume: 0.16, seconds: 0.05, filter: 'bandpass', freq: 1700 + Math.random() * 500, q: 2 });
    else this.burst({ volume: 0.22, seconds: 0.07, filter: 'lowpass', freq: 600 + Math.random() * 300, q: 0.7 });
  }

  /** Landing after a fall, louder the harder. */
  land(speed: number): void {
    const v = Math.min(1, speed / 20);
    this.tone({ volume: 0.25 + 0.4 * v, seconds: 0.12, from: 140, to: 55 });
    this.burst({ volume: 0.2 + 0.3 * v, seconds: 0.1, filter: 'lowpass', freq: 800 });
  }

  /** Falling or walking into water. */
  splash(): void {
    this.burst({ volume: 0.45, seconds: 0.35, filter: 'bandpass', freq: 1800, sweepTo: 400, q: 0.8 });
  }

  /** A diamond picked up: a bright two-note chime. */
  pickup(): void {
    this.tone({ volume: 0.18, seconds: 0.12, from: 1319, to: 1319, type: 'triangle' });
    this.tone({ volume: 0.18, seconds: 0.2, from: 1976, to: 1976, type: 'triangle', delay: 0.07 });
  }

  /** Whether this animal has a call. */
  hasCall(animal: string): boolean {
    return animal in CALLS;
  }

  /** An animal's call, from where it stands (higher when it's been hit). */
  call(animal: string, at: readonly number[], hurt = false): void {
    const c = CALLS[animal];
    const buffer = c && this.samples.get(c.file);
    if (!buffer) return;
    const g = this.chain(hurt ? 0.9 : 0.6, at);
    if (!g) return;
    const src = this.ctx!.createBufferSource();
    src.buffer = buffer;
    src.playbackRate.value = c.rate * (hurt ? 1.3 : 0.92 + Math.random() * 0.16);
    src.connect(g);
    src.start();
  }
}
