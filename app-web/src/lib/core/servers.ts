import { t } from "../i18n/i18n";
import { newId } from "./id";

import { Client, normalizeBaseUrl, record } from "./client";
import { parseModels } from "./models";
import type { ClientOptions } from "./client";
import type { RequestOptions } from "./client";
import type { Model } from "./models";
export type Server = { id: string; url: string; name: string; apiKey?: string; rememberKey: boolean; };

export type StorageAdapter = { getItem(key: string): string | null; setItem(key: string, value: string): void; };

class ServerStore {
  onRequestFinished: (id: string, path: string, method: string) => void = (): void => {};
  private storage: StorageAdapter | undefined;
  private entries: Server[] = [];
  private catalogues = (new Map() as Map<string, Model[]>);
  constructor(storage?: StorageAdapter) {
    this.storage = storage;
    try {
      const saved: unknown = JSON.parse(storage?.getItem("studio.servers") ?? "[]");
      if (Array.isArray(saved))
        for (const value of saved) {
          const v = record(value);
          if (typeof v.id !== "string" || typeof v.url !== "string") continue;
          try {
            this.entries.push({
              id: v.id,
              url: normalizeBaseUrl(v.url),
              name: typeof v.name === "string" ? v.name : v.url,
              rememberKey: v.rememberKey === true,
              apiKey:
                v.rememberKey === true && typeof v.apiKey === "string"
                  ? v.apiKey
                  : undefined,
            });
          } catch {
            /* Skip an invalid saved URL. */
          }
        }
    } catch {
      /* A corrupt setting must not block startup. */
    }
  }
  private save() {
    this.storage?.setItem(
      "studio.servers",
      JSON.stringify(
        this.entries.map(({ apiKey, ...s }) => ({
          ...s,
          ...(s.rememberKey ? { apiKey } : {}),
        })),
      ),
    );
  }
  list(): Server[] {
    return this.entries.map((s) => ({ ...s }));
  }
  get(id: string) {
    const s = this.entries.find((s) => s.id === id);
    return s ? { ...s } : undefined;
  }
  add(input: {
      url: string;
      name?: string;
      apiKey?: string;
      rememberKey?: boolean;
    }) {
    const url = normalizeBaseUrl(input.url),
      id = newId();
    this.entries.push({
      id,
      url,
      name: input.name ?? url,
      apiKey: input.apiKey,
      rememberKey: input.rememberKey ?? false,
    });
    this.save();
    return id;
  }
  update(id: string, patch: Partial<Omit<Server, "id">>) {
    const i = this.entries.findIndex((s) => s.id === id);
    if (i < 0) throw new Error(t("Unknown server."));
    this.entries[i] = {
      ...this.entries[i],
      ...patch,
      ...(patch.url ? { url: normalizeBaseUrl(patch.url) } : {}),
    };
    this.catalogues.delete(id);
    this.save();
  }
  remove(id: string) {
    this.entries = this.entries.filter((s) => s.id !== id);
    this.catalogues.delete(id);
    this.save();
  }
  export(): string {
    return JSON.stringify(
      this.entries.map(({ id, url, name }) => ({ id, url, name })),
      null,
      2,
    );
  }
  client(id: string, options: ClientOptions = {}) {
    const s = this.get(id);
    if (!s) throw new Error(t("Unknown server."));
    return new Client(s.url, {
      ...options,
      apiKey: s.apiKey,
      onRequestFinished: (path, method) => {
        this.onRequestFinished(id, path, method);
        options.onRequestFinished?.(path, method);
      },
    });
  }
  health(id: string, options: ClientOptions = {}) {
    return this.client(id, options).json("/health", options);
  }
  async models(id: string, options: ClientOptions & RequestOptions = {}) {
    const models = parseModels(
      await this.client(id, options).json("/v1/models", options),
    );
    this.catalogues.set(id, models);
    return models;
  }
  catalogue(id: string) {
    return structuredClone(this.catalogues.get(id) ?? []);
  }
}

export { ServerStore };
