import { record, type ClientOptions } from "../core/client";
import type { Model } from "../core/models";
import { ServerStore, type Server, type StorageAdapter } from "../core/servers";
import { t } from "../i18n/i18n";

export type Status = "idle" | "checking" | "online" | "error";
type Check = { status: Status; failure: string };

/** A finished request of these kinds changes what the server has resident. */
const CHANGES_MODELS = /^\/v1\/(chat\/completions|images\/(generations|edits)|video\/generations|audio\/(speech|music-generations|sound-generations)|load-model|unload-model)$/;

export const defaultModel = (models: Model[]) => models.find((m) => m.loaded) ?? models[0];
export const loadNotice = (model: Model | undefined, server: string) =>
  model && !model.loaded ? t("This loads %@ on %@ first.", [model.id, server]) : "";

/** Selected-server state. Only reads health/catalogue; never manages host models. */
export class Connection {
  readonly store: ServerStore;
  models = $state<Model[]>([]);
  serverVersion = $state("");
  residentBytes = $state<number | null>(null);
  status = $state<Status>("idle");
  failure = $state("");
  selectedModel = $state("");
  servers = $state<Server[]>([]);
  checks = $state<Record<string, Check>>({});
  activeId = $state("");
  active = $derived(this.servers.find((s) => s.id === this.activeId) as Server);
  message = $derived(
    this.status === "checking" ? t("Checking connection…") : this.status === "online" ? t("Connected") : this.status === "error" ? this.failure || t("Connection failed.") : t("Not checked"),
  );
  pending: AbortController | undefined;
  infoPending: AbortController | undefined;
  readonly origin: string;
  #storage: StorageAdapter;
  #options: ClientOptions;

  /** @param pageKey the page's own `?api_key=`, for this origin only */
  constructor(origin: string, storage: StorageAdapter, options: ClientOptions = {}, pageKey?: string) {
    this.#storage = storage;
    this.#options = options;
    this.origin = origin;
    this.store = new ServerStore(storage);
    this.store.onRequestFinished = (id, path, method) => {
      if (id === this.activeId && method === "POST" && CHANGES_MODELS.test(path)) void this.refresh();
    };
    const home = this.store.list().find((s) => s.url === origin)?.id ?? this.store.add({ url: origin, name: "This server" });
    if (pageKey) this.store.update(home, { apiKey: pageKey });
    const saved = storage.getItem("studio.activeServer");
    this.activeId = saved && this.store.get(saved) ? saved : home;
    this.sync();
  }

  /** Display the built-in label without changing the saved server name. */
  serverName(name: string = this.active.name) {
    return name === "This server" ? t("This server") : name;
  }

  /** A client for a server of the list (the selected one by default). */
  client(id: string = this.activeId) {
    return this.store.client(id, this.#options);
  }

  private sync() {
    this.servers = this.store.list();
  }

  select(id: string) {
    if (!this.store.get(id)) return;
    this.pending?.abort();
    this.pending = undefined;
    this.infoPending?.abort();
    this.infoPending = undefined;
    this.serverVersion = "";
    this.residentBytes = null;
    this.activeId = id;
    this.models = [];
    this.selectedModel = "";
    this.status = "idle";
    this.failure = "";
    this.#storage.setItem("studio.activeServer", id);
  }

  add(input: Parameters<ServerStore["add"]>[0]) {
    const id = this.store.add(input);
    this.sync();
    return id;
  }

  update(id: string, patch: Parameters<ServerStore["update"]>[1]) {
    this.store.update(id, patch);
    this.sync();
  }

  remove(id: string) {
    this.store.remove(id);
    delete this.checks[id];
    this.sync();
    if (id === this.activeId) this.select(this.store.list()[0]?.id ?? this.add({ url: this.origin, name: "This server" }));
  }

  chooseModel(id: string) {
    if (this.models.some((m) => m.id === id)) this.selectedModel = id;
  }

  async refreshInfo() {
    if (this.infoPending) return;
    const controller = new AbortController();
    this.infoPending = controller;
    const client = this.client();
    const options = { signal: controller.signal, timeoutMs: 5000 };
    const [props, version] = await Promise.allSettled([client.json("/props", options), client.json("/api/version", options)]);
    if (this.infoPending !== controller || controller.signal.aborted) return;
    this.infoPending = undefined;
    const bytes = props.status === "fulfilled" ? record(record(props.value).memory).active_bytes : null;
    this.residentBytes = typeof bytes === "number" && Number.isFinite(bytes) && bytes >= 0 ? bytes : null;
    const v = version.status === "fulfilled" ? record(version.value).version : undefined;
    this.serverVersion = typeof v === "string" ? v.slice(0, 100) : "";
  }

  async refresh() {
    this.pending?.abort();
    const controller = new AbortController();
    this.pending = controller;
    const id = this.activeId;
    this.status = "checking";
    this.checks[id] = { status: "checking", failure: "" };
    const options = { ...this.#options, signal: controller.signal, timeoutMs: 5000 };
    const [health, models] = await Promise.allSettled([this.store.health(id, options), this.store.models(id, options)]);
    if (this.pending !== controller || controller.signal.aborted) return;
    this.pending = undefined;
    this.models = models.status === "fulfilled" ? models.value : [];
    if (!this.models.some((m) => m.id === this.selectedModel))
      this.selectedModel = defaultModel(this.models.filter((m) => m.capabilities.includes("chat") || !m.capabilities.length))?.id ?? "";
    const reason = health.status === "rejected" ? health.reason : models.status === "rejected" ? models.reason : undefined;
    this.failure = reason instanceof Error ? reason.message : "";
    this.status = reason ? "error" : "online";
    this.checks[id] = { status: this.status, failure: this.failure };
  }
}
