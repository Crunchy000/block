import { BLOCK_NAMES, Block } from '../constants';

/** Placeable blocks, in hotbar order (keys 1–4). */
export const HOTBAR_BLOCKS = [Block.Dirt, Block.Stone, Block.Water, Block.Lava] as const;

export interface Hotbar {
  setSelected(block: Block): void;
}

/** Block picker along the bottom of the screen. Tapping a slot selects it; keys 1–4 do the same. */
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
  parent.append(bar);
  return {
    setSelected(block) {
      slots.forEach((slot, i) => slot.classList.toggle('selected', HOTBAR_BLOCKS[i] === block));
    },
  };
}
