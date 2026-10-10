import type { Library, LibraryFilter, LibraryInput, LibraryItem } from "../../src/lib/core/library";

/** An in-memory stand-in for the IndexedDB Library, same surface the app calls. */
export class MemoryLibrary {
  items = new Map<string, LibraryItem>();
  failWrites = false;
  async add(input: LibraryInput): Promise<string> {
    const id = crypto.randomUUID();
    this.items.set(id, { ...input, id, createdAt: input.createdAt ?? Date.now() });
    return id;
  }
  async put(id: string, input: LibraryInput) {
    if (this.failWrites) throw new Error("quota");
    this.items.set(id, { ...input, id, createdAt: input.createdAt ?? Date.now() });
  }
  async list(filter: LibraryFilter = {}) {
    return [...this.items.values()]
      .filter((i) => (!filter.type || i.type === filter.type) && (!filter.server || i.server === filter.server) && (!filter.model || i.model === filter.model))
      .map(({ blob: _, ...rest }) => rest)
      .sort((a, b) => b.createdAt - a.createdAt);
  }
  async get(id: string) {
    return this.items.get(id);
  }
  async delete(id: string) {
    this.items.delete(id);
  }
  async append(items: LibraryItem[]) {
    for (const item of items) this.items.set(item.id, item);
  }
  get asLibrary() {
    return this as unknown as Library;
  }
}
