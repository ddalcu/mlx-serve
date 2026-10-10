import { record, StudioError } from "../core/client";
import { pageServer } from "../core/console";
import { MetricsPoller } from "../core/metrics";
import { fetchMonitorFeed } from "../core/monitor-feed";
import {
  emptyDoc,
  HistoryStorage,
  liveRates,
  makeSample,
  mergeDocs,
  resetBetween,
  trackRequests,
  type Doc,
  type Feed,
  type LiveSample,
  type StorageEnv,
} from "../core/monitor-history";
import type { Server } from "../core/servers";
import { N } from "../i18n/i18n";
import type { Connection } from "./connection.svelte";

type LiveChartPoint = { t: number; decode: number | null; prefill: number | null };
type Rates = ReturnType<typeof liveRates>;

export const RANGES: [string, number][] = [
  [N("Live"), 60000],
  ["10m", 600000],
  ["30m", 1800000],
  ["1h", 3600000],
  ["24h", 86400000],
];

/**
 * Metrics collection belongs to the page, not to the mounted Monitoring pane: it polls the selected server,
 * keeps a thinned history in this browser, and carries on while other panes are open.
 */
export class MonitorWorkspace {
  doc = $state.raw<Doc>(emptyDoc());
  feed = $state.raw<Feed | null>(null);
  rates = $state.raw<Rates | null>(null);
  liveChart = $state.raw<LiveChartPoint[]>([]);
  models = $state.raw<Record<string, unknown>[] | null>(null);
  /** English keys: the pane translates them as it draws. */
  status = $state<string>(N("Connecting to metrics…"));
  where = $state<string>(N("Checking storage…"));
  range = $state(60000);
  now = $state(Date.now());
  server = $state.raw<Server | undefined>(undefined);
  #connection: Connection;
  #env: StorageEnv;
  #live: LiveSample[] = [];
  #poller: MetricsPoller<Feed> | undefined;
  #modelPoller: MetricsPoller<Record<string, unknown>[]> | undefined;
  #storage: HistoryStorage | undefined;
  #generation = 0;
  #lastSave = 0;
  #saving = false;

  constructor(connection: Connection, env: StorageEnv = window) {
    this.#connection = connection;
    this.#env = env;
    document.addEventListener("visibilitychange", () => document.hidden && void this.save());
    window.addEventListener("pagehide", () => void this.save());
    this.connectionChanged();
  }

  /** Point collection at the selected server; nothing happens unless it is a different server or key. */
  connectionChanged() {
    const next = this.#connection.active;
    if (next.id === this.server?.id && next.url === this.server?.url && next.apiKey === this.server?.apiKey) return;
    void this.save();
    this.#poller?.stop();
    this.#modelPoller?.stop();
    const generation = ++this.#generation;
    this.server = { ...next };
    this.doc = emptyDoc();
    this.feed = null;
    this.rates = null;
    this.#live = [];
    this.liveChart = [];
    this.models = null;
    this.#lastSave = 0;
    this.#saving = false;
    this.where = N("Checking storage…");
    this.status = N("Connecting to metrics…");
    const storage = (this.#storage = new HistoryStorage(next.url, this.#env));
    void storage.load().then((doc) => {
      if (generation === this.#generation) this.doc = mergeDocs(doc, this.doc, Date.now());
    });
    const client = this.#connection.client(next.id);
    this.#poller = new MetricsPoller(
      (signal) => fetchMonitorFeed(client, signal),
      (feed) => this.accept(feed),
      (error) => this.failed(error),
    );
    this.#modelPoller = new MetricsPoller(
      async (signal) => {
        const d = record(await client.json("/v1/models", { signal, timeoutMs: 5000 }));
        return (Array.isArray(d.data) ? d.data : []).map(record).filter((m) => typeof m.id === "string");
      },
      (models) => (this.models = models),
      () => (this.models = null),
      15000,
    );
    this.#poller.setVisible(true);
    this.#modelPoller.setVisible(true);
  }

  async save() {
    if (!this.#storage || this.#saving) return;
    const generation = this.#generation,
      storage = this.#storage;
    this.#saving = true;
    try {
      const saved = await storage.save(this.doc, Date.now());
      if (generation !== this.#generation) return;
      this.doc = mergeDocs(saved, this.doc, Date.now());
      this.where = saved.where;
      this.#lastSave = Date.now();
    } finally {
      if (generation === this.#generation) this.#saving = false;
    }
  }

  /** One polled feed: advance the live rates and the history. */
  accept(feed: Feed) {
    const now = Date.now(),
      sample = makeSample(now, feed),
      last = this.doc.samples.at(-1);
    const reset = !!last && resetBetween(last, sample);
    if (reset || (last && now - last.t > 300000)) this.#live = [];
    const rates = liveRates(this.#live, feed, now);
    this.#live.push({ t: now, live: rates.live, pre: rates.pre, req: feed.counters.requests_success_total ?? null });
    this.#live = this.#live.filter((s) => now - s.t <= 120000);
    this.liveChart = [...this.liveChart, { t: sample.t, decode: rates.decode, prefill: rates.prefill }].filter((s) => now - s.t <= 60000);
    this.doc = mergeDocs({ ...this.doc, rows: trackRequests(this.doc.rows, feed.sessions, now, reset) }, { samples: [sample], rows: [] }, now);
    this.rates = rates;
    this.feed = feed;
    this.status = "";
    this.now = now;
    if (now - this.#lastSave >= 15000) void this.save();
  }

  /** One failed poll: explain it in `status` and drop the live numbers. */
  failed(error: unknown) {
    this.feed = null;
    this.rates = null;
    this.#live = [];
    this.now = Date.now();
    const status = error instanceof StudioError ? error.status : undefined;
    const kind = error instanceof StudioError ? error.kind : undefined;
    const home = this.server?.url === pageServer(location);
    if (status === 403)
      this.status = this.server?.apiKey ? N("Monitoring access was denied. Check the API key and server access policy in Settings.") : N("Monitoring needs an API key on this server (Settings → Server → API key).");
    else if (status === 401) this.status = N("Monitoring authentication failed. Check the API key in Settings.");
    else if (kind === "unsupported" || [404, 405, 501, 503].includes(status ?? 0)) {
      // Only the server that serves this page knows its own --metrics launch setting, and it cannot change without a restart.
      this.status = home ? N("Metrics are disabled on this server. Start mlx-serve with --metrics to enable Monitoring.") : N("Metrics are disabled or unavailable on this server.");
      if (home) this.#poller?.stop();
    } else this.status = N("Cannot reach server metrics. Check the server connection. Retrying every second.");
  }
}
