import { t } from "../i18n/i18n";
// Missing values stay unknown; gap edges survive thinning.
import { normalizeBaseUrl, record } from "./client";
export type Numbers = Record<string, number | null>;
export type Session = { request_id: number; model: string; phase: string; client?: string; context_tokens?: number; context_length?: number; cached_tokens?: number; generated_tokens?: number; state_bytes?: number; };
export type Feed = { counters: Numbers; gauges: Numbers; histograms: Record<string, { sum: number | null; count: number | null; }>; sessions: Session[] | null; };
export type Sample = { t: number; p: number | null; m: string | null; c: Numbers; };
export type Row = { key: string; model: string; client: string; phase: string; startT: number; endT: number | null; lastT: number; ctx: number | null; cached: number | null; generated: number | null; };
export type Doc = { samples: Sample[]; rows: Row[]; };
export type LiveSample = { t: number; live: number | null; pre: number | null; req: number | null; };
const HISTORY_KEYS = [
  "requests_success_total",
  "requests_failed_total",
  "requests_rejected_total",
  "requests_cancelled_total",
  "prompt_tokens_total",
  "prefill_tokens_total",
  "generation_tokens_total",
  "prefix_cache_queries_total",
  "prefix_cache_hits_total",
  "prefix_cache_tokens_total",
];
const HOUR = 3600000,
  DAY = 24 * HOUR,
  GAP = 300000;
const number = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
const emptyDoc = (): Doc => ({ samples: [], rows: [] });
function makeSample(now: number, d: Feed): Sample {
  const models = new Set(
    (d.sessions ?? []).filter((s) => s.phase !== "cached").map((s) => s.model),
  );
  return {
    t: Math.floor(now / 1000) * 1000,
    p: number(d.gauges.process_start_time_seconds),
    m: models.size === 1 ? [...models][0] : null,
    c: Object.fromEntries(HISTORY_KEYS.map((k) => [k, number(d.counters[k])])),
  };
}
function resetBetween(a: Sample, b: Sample) {
  return (
    (a.p !== null && b.p !== null && a.p !== b.p) ||
    HISTORY_KEYS.some(
      (k) => a.c[k] != null && b.c[k] != null && b.c[k] < a.c[k],
    )
  );
}
const gap = (a: Sample, b: Sample) => b.t - a.t > GAP || b.t <= a.t || resetBetween(a, b);
function compact(samples: Sample[], now: number) {
  let bucket = -1;
  return samples.filter((s, i) => {
    if (s.t < now - DAY || s.t > now + 1000) return false;
    const next = Math.floor(s.t / 60000);
    const edge =
      (i > 0 && gap(samples[i - 1], s)) ||
      (i + 1 < samples.length && gap(s, samples[i + 1]));
    const keep = s.t >= now - HOUR || next !== bucket || edge;
    bucket = next;
    return keep;
  });
}
function pairs(samples: Sample[], from: number, to: number) {
  return samples
    .slice(1)
    .map((b, i) => ({ a: samples[i], b }))
    .filter(({ b }) => b.t > from && b.t <= to);
}
const delta = (a: Sample, b: Sample, key: string) =>
  gap(a, b) || a.c[key] == null || b.c[key] == null
    ? null
    : b.c[key] - a.c[key];
/** End-of-interval bucketing; no interpolation over missing polls. */
function rateSeries(samples: Sample[], key: string, from: number, to: number, n: number) {
  const sums = new Array(n).fill(0),
    time = new Array(n).fill(0),
    cut = new Array(n).fill(false);
  for (const { a, b } of pairs(samples, from, to)) {
    const i = Math.min(
      n - 1,
      Math.max(0, Math.ceil((b.t - from) / ((to - from) / n)) - 1),
    );
    const d = delta(a, b, key);
    if (d === null) {
      cut[i] = true;
      continue;
    }
    sums[i] += d;
    time[i] += (b.t - a.t) / 1000;
  }
  return sums.map((v, i) => (cut[i] || !time[i] ? null : v / time[i]));
}
function totals(samples: Sample[], from: number, to: number, byModel: boolean = false) {
   const out: Record<string, Numbers> = Object.create(null);
  for (const { a, b } of pairs(samples, from, to)) {
    if (gap(a, b)) continue;
    const key = byModel ? a.m || b.m || "unattributed" : "all";
    const row = (out[key] ??= Object.fromEntries(
      HISTORY_KEYS.map((k) => [k, 0]),
    ));
    for (const k of HISTORY_KEYS) {
      const d = delta(a, b, k);
      row[k] = d === null || row[k] === null ? null : row[k] + d;
    }
  }
  return out;
}
const windowTotals = (samples: Sample[], from: number, to: number) =>
  totals(samples, from, to).all ??
  Object.fromEntries(HISTORY_KEYS.map((k) => [k, 0]));
const modelTotals = (samples: Sample[], from: number, to: number) =>
  totals(samples, from, to, true);
function liveRates(samples: LiveSample[], d: Feed, now: number) {
  const c = d.counters,
    g = d.gauges,
    live = g.generation_tokens_live ?? c.generation_tokens_total ?? null,
    pre = g.prefill_tokens_live ?? null;
  const rate = (window: number, key: 'live' | 'pre' | 'req', value: number | null) => {
    let first = samples[0];
    for (const s of samples) {
      if (now - s.t >= window) first = s;
      else break;
    }
    return !first ||
      value == null ||
      first[key] == null ||
      now <= first.t ||
      now - first.t > GAP ||
      value < first[key]
      ? null
      : (value - first[key]) / ((now - first.t) / 1000);
  };
  const sum = d.histograms.prefill_time_seconds?.sum;
  return {
    decode: g.requests_running === 0 ? 0 : rate(4000, "live", live),
    prefill: pre === 0 ? 0 : rate(30000, "pre", pre),
    requests: rate(60000, "req", c.requests_success_total),
    averagePrefill:
      sum != null &&
      sum > 1e-6 &&
      c.prefill_tokens_total != null &&
      c.prefill_tokens_total > 0
        ? c.prefill_tokens_total / sum
        : null,
    live,
    pre,
  };
}
function trackRequests(rows: Row[], sessions: Session[] | null, now: number, reset: boolean = false) {
  const out = rows.map((r) => ({ ...r }));
  // A lost observation is not proof of completion; end at last observed time.
  for (const r of out)
    if (r.endT === null && (reset || now - r.lastT > GAP)) r.endT = r.lastT;
  if (sessions === null) return out;
  const seen = new Set();
  for (const s of sessions) {
    if (s.phase === "cached") continue;
    const key = String(s.request_id);
    seen.add(key);
    let row = out.find((r) => r.key === key && r.endT === null);
    if (!row) {
      row = {
        key,
        model: s.model,
        client: s.client ?? "—",
        phase: s.phase,
        startT: now,
        endT: null,
        lastT: now,
        ctx: null,
        cached: null,
        generated: null,
      };
      out.push(row);
    }
    Object.assign(row, {
      phase: s.phase,
      lastT: now,
      ctx: number(s.context_tokens),
      cached: number(s.cached_tokens),
      generated: number(s.generated_tokens),
    });
  }
  for (const r of out) if (r.endT === null && !seen.has(r.key)) r.endT = now;
  const closed = out.filter((r) => r.endT !== null),
    drop = new Set(closed.slice(0, Math.max(0, closed.length - 200)));
  return out.filter((r) => !drop.has(r) && r.lastT >= now - DAY);
}
/** Requests observed during the range, including ones that began earlier. */
const rangeRows = (rows: Row[], from: number, to: number) =>
  rows
    .filter((r) => r.startT <= to && (r.endT ?? r.lastT) >= from)
    .slice()
    .reverse()
    .slice(0, 50);
function mergeDocs(a: Doc, b: Doc, now: number): Doc {
  const samples = new Map([...a.samples, ...b.samples].map((s) => [s.t, s]));
   const rows: Row[] = [];
  for (const r of [...a.rows, ...b.rows].sort((x, y) => x.startT - y.startT)) {
    const twin = rows.find(
      (x) =>
        x.key === r.key &&
        x.model === r.model &&
        !(x.endT !== null && r.startT > x.endT) &&
        Math.abs(x.startT - r.startT) <= 2000,
    );
    if (!twin) rows.push({ ...r });
    else {
      const end =
        twin.endT === null
          ? r.endT
          : r.endT === null
            ? twin.endT
            : Math.max(twin.endT, r.endT);
      if (r.lastT > twin.lastT) Object.assign(twin, r);
      twin.endT = end;
    }
  }
  const kept = rows.filter((r) => r.lastT >= now - DAY && r.startT <= now);
  const closed = kept.filter((r) => r.endT !== null),
    drop = new Set(closed.slice(0, Math.max(0, closed.length - 200)));
  return {
    samples: compact(
      [...samples.values()].sort((x, y) => x.t - y.t),
      now,
    ),
    rows: kept.filter((r) => !drop.has(r)),
  };
}
function sanitize(raw: unknown): Doc {
  const d = record(raw);
  const samples = (Array.isArray(d.samples) ? d.samples : []).flatMap((s) => {
    if (number(s?.t) === null) return [];
    return [
      {
        t: s.t,
        p: number(s.p),
        m: typeof s.m === "string" ? s.m : null,
        c: Object.fromEntries(HISTORY_KEYS.map((k) => [k, number(s.c?.[k])])),
      },
    ];
  });
  const rows = (Array.isArray(d.rows) ? d.rows : []).flatMap((r) => {
    if (
      !r ||
      typeof r.key !== "string" ||
      typeof r.model !== "string" ||
      number(r.startT) === null ||
      number(r.lastT) === null
    )
      return [];
    return [
      {
        key: r.key,
        model: r.model,
        client: typeof r.client === "string" ? r.client : "—",
        phase: typeof r.phase === "string" ? r.phase : "",
        startT: r.startT,
        lastT: r.lastT,
        endT: number(r.endT),
        ctx: number(r.ctx),
        cached: number(r.cached),
        generated: number(r.generated),
      },
    ];
  });
  return mergeDocs({ samples, rows }, emptyDoc(), Date.now());
}
export type StorageEnv = { indexedDB?: IDBFactory; localStorage?: Pick<Storage, 'getItem' | 'setItem'>; };
class HistoryStorage {
  env: StorageEnv;
  key: string;
  constructor(url: string, env: StorageEnv) {
    this.key = "studio.monitor:" + normalizeBaseUrl(url);
    this.env = env;
  }
  /** One read/write transaction merges concurrent tabs atomically. */
  idb(doc?: Doc, now: number = Date.now()): Promise<Doc> {
    return new Promise((resolve, reject) => {
      const idb = this.env.indexedDB;
      if (!idb) {
        reject(new Error(t("No IndexedDB")));
        return;
      }
      const open = idb.open("studio.monitor", 1);
      let settled = false;
      let pending: IDBTransaction | undefined;
      const timer = setTimeout(() => {
        settled = true;
        pending?.abort();
        reject(new Error(t("Storage timeout")));
      }, 2000);
      open.onupgradeneeded = () => open.result.createObjectStore("history");
      open.onerror = open.onblocked = () => {
        clearTimeout(timer);
        settled = true;
        reject(new Error(t("Storage unavailable")));
      };
      open.onsuccess = () => {
        const db = open.result;
        if (settled) {
          db.close();
          return;
        }
        try {
          const tx = (pending = db.transaction(
              "history",
              doc ? "readwrite" : "readonly",
            )),
            store = tx.objectStore("history"),
            get = store.get(this.key);
          let result = emptyDoc();
          get.onsuccess = () => {
            result = sanitize(get.result);
            if (doc) {
              result = mergeDocs(result, doc, now);
              store.put(result, this.key);
            }
          };
          tx.oncomplete = () => {
            clearTimeout(timer);
            db.close();
            resolve(result);
          };
          tx.onerror = tx.onabort = () => {
            clearTimeout(timer);
            db.close();
            reject(new Error(t("Storage transaction failed")));
          };
        } catch (error) {
          clearTimeout(timer);
          db.close();
          reject(error);
        }
      };
    });
  }
  local() {
    try {
      return sanitize(
        JSON.parse(this.env.localStorage?.getItem(this.key) ?? "null"),
      );
    } catch {
      return emptyDoc();
    }
  }
  async load() {
    const local = this.local();
    try {
      return mergeDocs(local, await this.idb(), Date.now());
    } catch {
      return local;
    }
  }
  async save(doc: Doc, now: number) {
    const merged = mergeDocs(this.local(), doc, now);
    try {
      return { ...(await this.idb(merged, now)), where: "IndexedDB" };
    } catch {}
    try {
      if (!this.env.localStorage) throw new Error(t("No storage"));
      this.env.localStorage.setItem(this.key, JSON.stringify(merged));
      return { ...merged, where: "localStorage" };
    } catch {
      return { ...merged, where: "memory" };
    }
  }
}

export { HISTORY_KEYS, number, emptyDoc, makeSample, resetBetween, rateSeries, windowTotals, modelTotals, liveRates, trackRequests, rangeRows, mergeDocs, HistoryStorage };
