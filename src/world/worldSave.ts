/**
 * Edited chunks kept between visits, in the browser's IndexedDB: one record per chunk that
 * differs from the generated world (its cells as bytes), under the name of the world it
 * belongs to.
 * Chunks that were never changed aren't stored: they regenerate the same.
 */
const DB = 'block-world';
const STORE = 'chunks';
/** Bump when generation changes so that old saves no longer fit the world around them. */
const VERSION = 'v2'; // v2: the worldgen hash that's the same on every device

function request<T>(r: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

export class WorldSave {
  /** The chunk keys ("cx,cz") this world has saved. */
  readonly keys: Set<string>;

  private constructor(private readonly db: IDBDatabase, private readonly prefix: string, keys: string[]) {
    this.keys = new Set(keys.map((k) => k.slice(prefix.length)));
  }

  /** The save of the world called `name`, or undefined where the browser has no IndexedDB (or blocks it). */
  static async open(name: string): Promise<WorldSave | undefined> {
    try {
      const db = await openDb();
      const prefix = `${VERSION}/${name}/`;
      const keys = await request(db.transaction(STORE).objectStore(STORE).getAllKeys(IDBKeyRange.bound(prefix, `${prefix}￿`)));
      return new WorldSave(db, prefix, keys as string[]);
    } catch {
      return undefined;
    }
  }

  /** Forget everything saved for the world called `name`. */
  static async clear(name: string): Promise<void> {
    const db = await openDb();
    const prefix = `${VERSION}/${name}/`;
    await request(db.transaction(STORE, 'readwrite').objectStore(STORE).delete(IDBKeyRange.bound(prefix, `${prefix}￿`)));
    db.close();
  }

  /** A saved chunk's cells, if there are any. */
  async load(key: string): Promise<Uint8Array | undefined> {
    const cells = await request(this.db.transaction(STORE).objectStore(STORE).get(this.prefix + key)) as Uint8Array | undefined;
    return cells;
  }

  save(key: string, cells: Uint8Array): Promise<void> {
    this.keys.add(key);
    return request(this.db.transaction(STORE, 'readwrite').objectStore(STORE).put(cells, this.prefix + key)).then(() => {});
  }
}
