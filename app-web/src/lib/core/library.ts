import { t } from "../i18n/i18n";
import { newId } from "./id";
import { normalizeBaseUrl } from "./client";
export type ItemType = "chat" | "image" | "speech" | "music" | "sound" | "video";

export type LibraryItem = { id: string; type: ItemType; model: string; server: string; createdAt: number; prompt?: string; blob: Blob; };

export type LibraryInput = Omit<LibraryItem, "id" | "createdAt"> & {
  createdAt?: number;
};

export type LibraryFilter = { type?: ItemType; model?: string; server?: string; from?: number; to?: number; };

class Library {
  private name: string;
  private db: IDBDatabase | undefined;
  private opening: Promise<IDBDatabase> | undefined;
  constructor(name = "mlx-serve-studio") {
    this.name = name;
  }
  private open(): Promise<IDBDatabase> {
    if (this.db) return Promise.resolve(this.db);
    if (this.opening) return this.opening;
    this.opening = new Promise((resolve, reject) => {
      const request = indexedDB.open(this.name, 1);
      request.onupgradeneeded = () => {
        request.result.createObjectStore("items", { keyPath: "id" });
      };
      request.onerror = () => reject(request.error);
      request.onblocked = () =>
        reject(new Error(t("Library upgrade is blocked by another tab.")));
      request.onsuccess = () => {
        this.db = request.result;
        this.db.onversionchange = () => this.close();
        resolve(this.db);
      };
    });
    return this.opening.finally(() => {
      this.opening = undefined;
    });
  }
  private async transaction<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction("items", mode),
        req = run(tx.objectStore("items"));
      tx.oncomplete = () => resolve(req.result);
      tx.onerror = () => reject(tx.error ?? req.error);
      tx.onabort = () =>
        reject(tx.error ?? new Error(t("Library write aborted.")));
    });
  }
  async add(input: LibraryInput): Promise<string> {
    const id = newId(),
      item: LibraryItem = {
        id,
        type: input.type,
        model: input.model,
        server: normalizeBaseUrl(input.server),
        createdAt: input.createdAt ?? Date.now(),
        prompt: input.prompt,
        blob: input.blob,
      };
    if (!Number.isFinite(item.createdAt) || !(item.blob instanceof Blob))
      throw new Error(t("Invalid library item."));
    await this.transaction("readwrite", (s) => s.add(item));
    return id;
  }
  /** Atomically replace a known-ID item; the session layer supplies a UUID. */
  async put(id: string, input: LibraryInput) {
    const item = {
      id,
      type: input.type,
      model: input.model,
      server: normalizeBaseUrl(input.server),
      createdAt: input.createdAt ?? Date.now(),
      prompt: input.prompt,
      blob: input.blob,
    };
    if (!id || !Number.isFinite(item.createdAt) || !(item.blob instanceof Blob))
      throw new Error(t("Invalid library item."));
    await this.transaction("readwrite", (s) => s.put(item));
  }
  async list(filter: LibraryFilter = {}): Promise<Omit<LibraryItem, "blob">[]> {
    // Cursor filters without materializing every media blob into one JS array.
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction("items"),
        req = tx.objectStore("items").openCursor(),
        out: Omit<LibraryItem, "blob">[] = [];
      req.onsuccess = () => {
        const cursor = req.result;
        if (!cursor) return;
        const { blob: _, ...item } = (cursor.value as LibraryItem);
        if (
          (!filter.type || item.type === filter.type) &&
          (!filter.model || item.model === filter.model) &&
          (!filter.server || item.server === filter.server) &&
          (filter.from === undefined || item.createdAt >= filter.from) &&
          (filter.to === undefined || item.createdAt <= filter.to)
        )
          out.push(item);
        cursor.continue();
      };
      tx.oncomplete = () =>
        resolve(out.sort((a, b) => b.createdAt - a.createdAt));
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  }
  get(id: string): Promise<LibraryItem | undefined> {
    return this.transaction("readonly", (s) => s.get(id));
  }
  async delete(id: string) {
    await this.transaction("readwrite", (s) => s.delete(id));
  }
  async export(id: string) {
    const item = await this.get(id);
    if (!item) throw new Error(t("Library item not found."));
    const ext =
      ({
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "audio/wav": "wav",
        "audio/mpeg": "mp3",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
        "video/mp4": "mp4",
        "video/webm": "webm",
        "application/json": "json",
        "text/plain": "txt",
      } as Record<string, string>)[item.blob.type] ?? "bin";
    return { blob: item.blob, filename: `${id}.${ext}` };
  }
  /**
   * Append a fully validated archive in one transaction; a quota failure rolls back all rows.
   */
  async append(items: LibraryItem[]) {
    const db = await this.open();
    await new Promise((resolve, reject) => {
      const tx = db.transaction("items", "readwrite");
      tx.oncomplete = () => resolve(undefined);
      tx.onabort = () =>
        reject(tx.error ?? new Error(t("Library import aborted.")));
      for (const item of items) tx.objectStore("items").add(item);
    });
  }
  close() {
    this.db?.close();
    this.db = undefined;
  }
}

export { Library };
