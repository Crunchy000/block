import { BLOCK_NAMES, Block } from '../constants';

/**
 * Placeable blocks, in hotbar order (keys 1–7). Wheat is planted as a seedling on dirt or
 * grass. Diamond ore is placed from the diamonds you've mined, one each.
 */
export const HOTBAR_BLOCKS = [Block.Dirt, Block.Stone, Block.Water, Block.Lava, Block.Grass, Block.Wheat, Block.Diamond] as const;

export interface Hotbar {
  setSelected(block: Block): void;
  /** Show how many of a block you have (the diamond slot's count). */
  setCount(block: Block, count: number): void;
  /** Flash a slot (a diamond mined, or none left to place). */
  flash(block: Block): void;
}

/** Block picker along the bottom of the screen. Tapping a slot selects it; keys 1–7 do the same. */
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
    const count = document.createElement('span');
    count.className = 'count';
    slot.append(swatch, key, count);
    slot.addEventListener('pointerdown', (e) => e.preventDefault());
    slot.addEventListener('click', () => onSelect(block));
    bar.append(slot);
    return slot;
  });
  parent.append(bar);
  return {
    setSelected(block) {
      slots.forEach((slot, i) => slot.classList.toggle('selected', HOTBAR_BLOCKS[i] === block));
    },
    setCount(block, n) {
      const slot = slots[HOTBAR_BLOCKS.indexOf(block as (typeof HOTBAR_BLOCKS)[number])];
      if (!slot) return;
      slot.querySelector('.count')!.textContent = String(n);
      slot.classList.toggle('empty', n === 0);
      slot.setAttribute('aria-label', `Place ${BLOCK_NAMES[block]} (${n} left)`);
    },
    flash(block) {
      const slot = slots[HOTBAR_BLOCKS.indexOf(block as (typeof HOTBAR_BLOCKS)[number])];
      if (!slot) return;
      slot.classList.remove('flash');
      void slot.offsetWidth; // restart the animation
      slot.classList.add('flash');
    },
  };
}
