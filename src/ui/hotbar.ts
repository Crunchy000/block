import { CONCRETE_COLOURS } from '../constants';

export interface Hotbar {
  /** Highlight the concrete colour being placed (an index into CONCRETE_COLOURS). */
  setSelected(colour: number): void;
  /** Show how many diamonds you've collected. */
  setDiamonds(count: number): void;
  /** Flash the diamond count (one just picked up). */
  flashDiamonds(): void;
}

/**
 * The building blocks along the bottom of the screen, pastel concrete in each colour, with
 * the diamonds you've collected at the end. Tapping a slot selects it; keys 1–8 do the same.
 */
export function createHotbar(onSelect: (colour: number) => void, parent: HTMLElement = document.body): Hotbar {
  const bar = document.createElement('div');
  bar.className = 'hotbar';
  const slots = CONCRETE_COLOURS.map(({ name, rgb }, i) => {
    const slot = document.createElement('button');
    slot.className = 'slot';
    slot.setAttribute('aria-label', `Place ${name} concrete`);
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = `rgb(${rgb.map((v) => Math.round(v * 255)).join(' ')})`;
    const key = document.createElement('span');
    key.className = 'key';
    key.textContent = String(i + 1);
    slot.append(swatch, key);
    slot.addEventListener('pointerdown', (e) => e.preventDefault());
    slot.addEventListener('click', () => onSelect(i));
    bar.append(slot);
    return slot;
  });
  // Collected diamonds: an item, not a block to place.
  const gems = document.createElement('div');
  gems.className = 'slot item item-diamond';
  const gem = document.createElement('span');
  gem.className = 'swatch';
  const count = document.createElement('span');
  count.className = 'count';
  gems.append(gem, count);
  bar.append(gems);
  parent.append(bar);
  return {
    setSelected(colour) {
      slots.forEach((slot, i) => slot.classList.toggle('selected', i === colour));
    },
    setDiamonds(n) {
      count.textContent = String(n);
      gems.classList.toggle('empty', n === 0);
      gems.setAttribute('aria-label', `${n} diamond${n === 1 ? '' : 's'}`);
    },
    flashDiamonds() {
      gems.classList.remove('flash');
      void gems.offsetWidth; // restart the animation
      gems.classList.add('flash');
    },
  };
}
