import { BUILDING_BLOCKS } from '../constants';

export interface Hotbar {
  /** Highlight the block being placed (an index into BUILDING_BLOCKS). */
  setSelected(index: number): void;
  /** Show how many diamonds you've collected. */
  setDiamonds(count: number): void;
  /** Flash the diamond count (one just picked up). */
  flashDiamonds(): void;
}

/**
 * The building blocks along the bottom of the screen (pastel concrete in each colour, sand and
 * gravel), with the diamonds you've collected at the end. Tapping a slot selects it; keys 1–9
 * and 0 do the same.
 */
export function createHotbar(onSelect: (index: number) => void, parent: HTMLElement = document.body): Hotbar {
  const bar = document.createElement('div');
  bar.className = 'hotbar';
  const slots = BUILDING_BLOCKS.map(({ name, swatch: look }, i) => {
    const slot = document.createElement('button');
    slot.className = 'slot';
    slot.setAttribute('aria-label', `Place ${name}`);
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = look;
    const key = document.createElement('span');
    key.className = 'key';
    key.textContent = String((i + 1) % 10);
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
    setSelected(index) {
      slots.forEach((slot, i) => slot.classList.toggle('selected', i === index));
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
