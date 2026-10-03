// Builds the game's sound effects, public/sounds/<name>-<n>.mp3, from Kenney's "Impact Sounds"
// and "RPG Audio" (CC0; the originals used are in assets/sounds/kenney/). Each is trimmed of
// leading silence, levelled to -1 dB peak, made mono and, if given a length, cut short with a
// fade. Run after changing them: node scripts/convert-sounds.mjs   (needs ffmpeg)
// (The animal calls, public/sounds/{cow,pig,chicken}.mp3, are separate recordings, not made
// here. So is the splash, public/sounds/splash.mp3, from assets/sounds/water-splosh.mp3: mono,
// leading silence trimmed, cut to 0.7 s with a fade.) Keep the names and counts in step with TAKES in src/ui/sounds.ts.
import { execFileSync, spawnSync } from 'node:child_process';

const SRC = new URL('../assets/sounds/kenney/', import.meta.url).pathname;
const OUT = new URL('../public/sounds/', import.meta.url).pathname;
const five = (name) => [0, 1, 2, 3, 4].map((i) => `${name}_00${i}`);

/** Game sound: its source takes (in order), and how long to keep of each (seconds), if cut. */
const SOUNDS = [
  { name: 'step-grass', from: five('footstep_grass') },
  { name: 'step-hard', from: five('footstep_concrete') },
  { name: 'step-gravel', from: five('footstep_snow') },
  { name: 'step-sand', from: [0, 1, 2, 3, 4].map((i) => `footstep0${i}`) },
  { name: 'dig-hard', from: five('impactMining'), seconds: 0.35 },
  { name: 'dig-soft', from: five('impactSoft_medium') },
  { name: 'dug-hard', from: five('impactMining') },
  { name: 'dug-soft', from: five('impactSoft_heavy') },
  { name: 'place', from: five('impactGeneric_light') },
];

/** A file's peak level (dB), from ffmpeg's volumedetect (which reports on stderr). */
const peak = (file) => Number(/max_volume: (-?[\d.]+) dB/.exec(
  spawnSync('ffmpeg', ['-i', file, '-af', 'volumedetect', '-f', 'null', '-'], { encoding: 'utf8' }).stderr)?.[1] ?? 0);

for (const { name, from, seconds } of SOUNDS) {
  from.forEach((source, i) => {
    const file = `${SRC}${source}.ogg`, gain = -1 - peak(file);
    const cut = seconds ? `,atrim=0:${seconds},afade=t=out:st=${seconds * 0.5}:d=${seconds * 0.5}` : '';
    execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', file, '-af', `silenceremove=start_periods=1:start_threshold=-50dB,volume=${gain}dB${cut}`,
      '-ac', '1', '-ar', '44100', '-b:a', '64k', `${OUT}${name}-${i + 1}.mp3`]);
  });
  console.log(`${name}: ${from.length} takes`);
}
