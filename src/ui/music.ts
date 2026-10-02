/**
 * Background music: one track, looped, quiet, faded in and out. It plays while the game is
 * being played (browsers only let pages start sound from a click or tap, so it starts with
 * the one that starts play) and pauses on the start screen and while the tab is hidden. On or
 * off is remembered between visits. The file is fetched only once it first plays.
 */
const MUSIC_KEY = 'block.music';
const VOLUME = 0.45;
const FADE_MS = 1200;

export class Music {
  readonly audio: HTMLAudioElement;
  private fade?: number;
  private wanted = false;
  enabled: boolean;

  constructor(src: string) {
    this.audio = new Audio();
    this.audio.loop = true;
    this.audio.preload = 'none';
    this.audio.volume = 0;
    this.audio.src = new URL(src, document.baseURI).toString();
    let stored: string | null = null;
    try { stored = localStorage.getItem(MUSIC_KEY); } catch { /* storage blocked */ }
    this.enabled = stored !== '0';
    document.addEventListener('visibilitychange', () => this.apply());
  }

  /** Play (if music is on) or pause, as the game starts and stops being played. */
  setPlaying(playing: boolean): void {
    this.wanted = playing;
    this.apply();
  }

  /** Switch music on or off (M, or the touch Music button); remembered. */
  toggle(): boolean {
    this.enabled = !this.enabled;
    try { localStorage.setItem(MUSIC_KEY, this.enabled ? '1' : '0'); } catch { /* storage blocked */ }
    this.apply();
    return this.enabled;
  }

  private apply(): void {
    const on = this.enabled && this.wanted && !document.hidden;
    if (on && this.audio.paused) {
      // Rejected without a recent click or tap (autoplay rules): it starts with the next one.
      this.audio.play().catch(() => {});
    }
    this.fadeTo(on ? VOLUME : 0, () => { if (!on) this.audio.pause(); });
  }

  private fadeTo(target: number, done: () => void): void {
    window.clearInterval(this.fade);
    const from = this.audio.volume, start = performance.now();
    this.fade = window.setInterval(() => {
      const t = Math.min(1, (performance.now() - start) / FADE_MS);
      this.audio.volume = from + (target - from) * t;
      if (t === 1) {
        window.clearInterval(this.fade);
        done();
      }
    }, 50);
  }
}
