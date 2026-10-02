import { BLOCK_NAMES, Block } from '../constants';

/** Placeable blocks, in hotbar order (keys 1–6). Wheat is planted as a seedling on dirt or grass. */
export const HOTBAR_BLOCKS = [Block.Dirt, Block.Stone, Block.Water, Block.Lava, Block.Grass, Block.Wheat] as const;

export interface Hotbar {
  setSelected(block: Block): void;
  /** Show how many diamonds you've collected. */
  setDiamonds(count: number): void;
  /** Flash the diamond count (one just picked up). */
  flashDiamonds(): void;
}

/**
 * Block picker along the bottom of the screen, with the diamonds you've collected at its
 * end. Tapping a slot selects it; keys 1–6 do the same.
 */
export function createHotbar(onSelect: (block: Block) => void, parent: HTMLElement = document.body): Hotbar {
  const bar = document.createElement('div');
  bar.className = 'hotbar';
  const slots = HOTBAR_BLOCKS.map((block, i) => {
    const slot = document.createElement('button');
    slot.className = `slot block-${BLOCK_NAMES[block]}`;
    slot.setAttribute('aria-label', `Place ${BLOCK_NAMES[block]}`);
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    const key = document.createElement('span');
    key.className = 'key';
    key.textContent = String(i + 1);
    slot.append(swatch, key);
    slot.addEventListener('pointerdown', (e) => e.preventDefault());
    slot.addEventListener('click', () => onSelect(block));
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
    setSelected(block) {
      slots.forEach((slot, i) => slot.classList.toggle('selected', HOTBAR_BLOCKS[i] === block));
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
