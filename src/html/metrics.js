'use strict';
// Live metrics panel — the markup AND the polling logic in one file. The panel
// is injected into the `#mlx-metrics` mount on the index page (only present when
// the server ran with --metrics), then this polls the open /metrics.json feed
// once a second. The server keeps no time series: the history below the live
// tiles is held in this browser (IndexedDB, else localStorage, else memory) and
// every rate in it is a delta of the cumulative counters, as Prometheus computes
// rate().
//
// Decode & prefill "tok/s" are ACTIVE speeds — Δtokens ÷ Δ(phase time) over a
// trailing window. The phase-time sums (decode_time_seconds / prefill_time_
// seconds) and token counters only advance when a request COMPLETES, so this is
// the true generation/prefill speed of recently-finished requests — NOT the
// idle-averaged counter rate (which reads ~0 between requests).
// ── Pure rate math (no DOM, no module state) ─────────────────────────────────
//
// Every displayed rate is derived FROM THE CURRENT DATA on every tick. Nothing
// is carried between ticks. A value stashed in a module-level `let` outlives the
// condition that produced it: the Prefill tile used to keep showing the last
// prefill speed for the whole of a long decode, and only a page refresh cleared
// it — which is exactly the signature of state that isn't derived from the feed.
// Exported for `tests/metrics_panel_test.mjs`.

// Newest sample that is at least winMs old (tightest window >= winMs); the
// oldest retained sample while still warming up.
function panelAt(now, samples, winMs) {
  let s = samples[0];
  for (const x of samples) { if (now - x.t >= winMs) s = x; else break; }
  return s;
}

// `apiPrefix` (the mount the page was served under) comes from `api.js`, the
// boot script that this panel's own script is rendered after.

function computeRates(now, samples, c, g, psum) {
  const liveTok = (g.generation_tokens_live != null) ? g.generation_tokens_live : c.generation_tokens_total;
  const livePre = (g.prefill_tokens_live != null) ? g.prefill_tokens_live : 0;
  const prefilling = (g.requests_prefilling || 0) > 0;

  // Decode tok/s — LIVE decode speed while a request runs, 0 when idle. Tokens
  // only accrue during decode, so this is flat through prefill.
  let decodeTps = 0;
  if (g.requests_running > 0) {
    const wl = panelAt(now, samples, 4000);
    if (wl) {
      const dt = (now - wl.t) / 1000;
      if (dt > 0) decodeTps = Math.max(0, (liveTok - wl.live) / dt);
    }
  }

  // Prefill tok/s — LIVE prefill speed, 0 when no prefill is running. Same
  // no-carry-forward rule as decode: the big number answers "what is happening
  // NOW". Progress is published once per prefill CHUNK (8192 tokens), so the
  // window is wide enough to span one chunk even on a slow model.
  let prefillTps = 0;
  if (livePre > 0) {
    const wl = panelAt(now, samples, 30000);
    if (wl) {
      const dt = (now - wl.t) / 1000;
      if (dt > 0) prefillTps = Math.max(0, (livePre - wl.pre) / dt);
    }
  }

  // "How fast does this machine prefill?" — the stable answer, shown in the
  // sub-line where it can't be mistaken for a live rate. Cumulative, so it never
  // goes stale. Numerator is FORWARDED tokens (`prefill_tokens_total`), never
  // `prompt_tokens_total`: with the prefix cache warm most billed tokens are
  // restored, not computed, and dividing them by prefill time overstates
  // throughput by prompt/(prompt-cached) — measured 10.6x on a 35B MoE.
  const avgPrefillTps = (psum > 1e-6 && c.prefill_tokens_total > 0)
    ? c.prefill_tokens_total / psum
    : null;

  // Requests per second over a ~60s window.
  let reqRate = null;
  const wp = panelAt(now, samples, 60000);
  if (wp) {
    const dt = (now - wp.t) / 1000;
    if (dt > 0) reqRate = Math.max(0, (c.requests_success_total - wp.req) / dt);
  }

  return { decodeTps, prefillTps, avgPrefillTps, reqRate, prefilling, liveTok, livePre };
}

// ── Browser-held history ─────────────────────────────────────────────────────
//
// A history document is {samples, rows}. A sample is {t, p, m, c}: its time
// (whole seconds, so two tabs polling the same second agree on the key), the
// server's process start time or null, the one model with an active session or
// null, and the cumulative counters. Everything below is pure except the
// storage functions, which take their environment as an argument.

const HISTORY_KEYS = [
  'requests_success_total', 'requests_failed_total', 'requests_rejected_total',
  'requests_cancelled_total', 'prompt_tokens_total', 'prefill_tokens_total',
  'generation_tokens_total', 'prefix_cache_queries_total', 'prefix_cache_hits_total',
  'prefix_cache_tokens_total',
];
const RAW_MS = 3600000;          // samples newer than this are kept as polled
const KEEP_MS = 86400000;        // nothing older than this is kept
const BUCKET_MS = 60000;         // older samples are thinned to one per minute
const GAP_MS = 300000;           // two samples further apart than this share no rate
const MAX_ROWS = 200;
const ROW_JOIN_MS = 2000;        // tabs see a request open up to a second apart

const finite = (v) => typeof v === 'number' && Number.isFinite(v);

function activeModel(sessions) {
  const live = new Set();
  for (const s of Array.isArray(sessions) ? sessions : []) {
    if (s && s.phase !== 'cached' && typeof s.model === 'string') live.add(s.model);
  }
  return live.size === 1 ? [...live][0] : null;
}

function makeSample(now, d) {
  const c = (d && d.counters) || {};
  const g = (d && d.gauges) || {};
  const out = {
    t: Math.floor(now / 1000) * 1000,
    p: finite(g.process_start_time_seconds) ? g.process_start_time_seconds : null,
    m: activeModel(d && d.sessions),
    c: {},
  };
  for (const k of HISTORY_KEYS) out.c[k] = c[k];
  return out;
}

// A new process start time, or any counter going down, means the server restarted.
function resetBetween(a, b) {
  if (a.p !== null && b.p !== null && a.p !== b.p) return true;
  return HISTORY_KEYS.some((k) => b.c[k] < a.c[k]);
}

const noRate = (a, b) => b.t - a.t > GAP_MS || resetBetween(a, b);

function compact(samples, now) {
  const latest = now === undefined ? (samples.length ? samples[samples.length - 1].t : 0) : now;
  const rawFrom = latest - RAW_MS, keepFrom = latest - KEEP_MS;
  const out = [];
  let lastBucket = null;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.t < keepFrom) continue;
    const bucket = Math.floor(s.t / BUCKET_MS);
    const edge = (i > 0 && resetBetween(samples[i - 1], s)) ||
      (i + 1 < samples.length && resetBetween(s, samples[i + 1]));
    if (s.t >= rawFrom || bucket !== lastBucket || edge) out.push(s);
    lastBucket = bucket;
  }
  return out;
}

function mergeStores(a, b, now) {
  const all = a.concat(b).sort((x, y) => x.t - y.t);
  const uniq = [];
  for (const s of all) if (!uniq.length || uniq[uniq.length - 1].t !== s.t) uniq.push(s);
  return compact(uniq, now);
}

function appendSample(store, sample, now) {
  const at = now === undefined ? sample.t : now;
  if (!store.length || sample.t > store[store.length - 1].t) return compact(store.concat([sample]), at);
  return mergeStores(store, [sample], at);
}

// Consecutive sample pairs ending inside (fromT, toT]; `gap` marks the pairs a
// rate must not span.
function pairsIn(store, fromT, toT) {
  const out = [];
  for (let i = 1; i < store.length; i++) {
    const a = store[i - 1], b = store[i];
    if (b.t > fromT && b.t <= toT) out.push({ a, b, dt: (b.t - a.t) / 1000, gap: noRate(a, b) });
  }
  return out;
}

// Delta per second over the trailing window, from the current process only.
function rateOver(store, key, windowMs, now) {
  const inWin = store.filter((s) => s.t >= now - windowMs && s.t <= now);
  let start = inWin.length - 1;
  while (start > 0 && !noRate(inWin[start - 1], inWin[start])) start--;
  const reset = start > 0;
  if (inWin.length - start < 2) return { rate: null, reset };
  const first = inWin[start], last = inWin[inWin.length - 1];
  return { rate: (last.c[key] - first.c[key]) / ((last.t - first.t) / 1000), reset };
}

// n buckets across (fromT, toT]; null where there is no data or the server restarted.
function rateSeries(store, key, fromT, toT, n) {
  const step = (toT - fromT) / n;
  const num = new Array(n).fill(0), den = new Array(n).fill(0), cut = new Array(n).fill(false);
  for (const { a, b, dt, gap } of pairsIn(store, fromT, toT)) {
    const i = Math.min(n - 1, Math.max(0, Math.ceil((b.t - fromT) / step) - 1));
    if (gap) { cut[i] = resetBetween(a, b) || cut[i]; continue; }
    num[i] += b.c[key] - a.c[key];
    den[i] += dt;
  }
  return num.map((v, i) => (cut[i] || den[i] === 0 ? null : v / den[i]));
}

function sumDeltas(store, fromT, toT, bucketOf) {
  const out = {};
  for (const { a, b, gap } of pairsIn(store, fromT, toT)) {
    if (gap) continue;
    const name = bucketOf(a, b);
    const into = out[name] || (out[name] = Object.fromEntries(HISTORY_KEYS.map((k) => [k, 0])));
    for (const k of HISTORY_KEYS) into[k] += b.c[k] - a.c[k];
  }
  return out;
}

function windowTotals(store, fromT, toT) {
  return sumDeltas(store, fromT, toT, () => 'all').all ||
    Object.fromEntries(HISTORY_KEYS.map((k) => [k, 0]));
}

// A delta belongs to the model whose session was live at the start of the
// interval, since a counter only moves after the session has left the list.
function modelTotals(store, fromT, toT) {
  return sumDeltas(store, fromT, toT, (a, b) => a.m || b.m || '');
}

// ── Request table ────────────────────────────────────────────────────────────

function mkRow(r) {
  const n = (v) => (finite(v) ? v : 0);
  return {
    key: String(r.key), id: finite(r.id) ? r.id : null,
    client: typeof r.client === 'string' && r.client ? r.client : null,
    model: typeof r.model === 'string' ? r.model : '',
    phase: typeof r.phase === 'string' ? r.phase : '',
    startT: r.startT, endT: finite(r.endT) ? r.endT : null, lastT: finite(r.lastT) ? r.lastT : r.startT,
    ctx: n(r.ctx), ctxLen: n(r.ctxLen), cached: n(r.cached), generated: n(r.generated),
  };
}

const requestKey = (s) => 'r' + s.request_id;

function capClosed(rows) {
  const closed = rows.filter((r) => r.endT !== null);
  const drop = new Set(closed.slice(0, Math.max(0, closed.length - MAX_ROWS)));
  return rows.filter((r) => !drop.has(r));
}

// A row opens when its session appears and closes when it leaves.
function trackRequests(rows, sessions, now) {
  const out = rows.map((r) => Object.assign({}, r));
  const seen = new Set();
  for (const s of Array.isArray(sessions) ? sessions : []) {
    if (!s || s.phase === 'cached') continue;
    const key = requestKey(s);
    seen.add(key);
    let row = out.find((r) => r.endT === null && r.key === key);
    if (!row) { row = mkRow({ key, id: s.request_id, client: s.client, model: s.model, startT: now }); out.push(row); }
    Object.assign(row, {
      phase: s.phase, lastT: now, ctx: s.context_tokens || 0, ctxLen: s.context_length || 0,
      cached: s.cached_tokens || 0, generated: s.generated_tokens || 0,
    });
  }
  for (const r of out) if (r.endT === null && !seen.has(r.key)) r.endT = now;
  return capClosed(out);
}

function mergeRows(a, b) {
  const out = [];
  for (const r of a.concat(b).sort((x, y) => x.startT - y.startT)) {
    const twin = out.find((o) => o.key === r.key && Math.abs(o.startT - r.startT) <= ROW_JOIN_MS);
    if (!twin) { out.push(mkRow(r)); continue; }
    twin.endT = twin.endT === null || r.endT === null ? (twin.endT ?? r.endT) : Math.max(twin.endT, r.endT);
    if (r.lastT > twin.lastT) {
      Object.assign(twin, { phase: r.phase, lastT: r.lastT, ctx: r.ctx, ctxLen: r.ctxLen, cached: r.cached, generated: r.generated });
    }
  }
  return capClosed(out);
}

// ── Persistence ──────────────────────────────────────────────────────────────
//
// Every read and write is best effort: a store that throws, is missing, or
// holds something unreadable is an empty history, never an error in the panel.

const DB_NAME = 'mlx-serve-monitor', DB_STORE = 'kv', DB_KEY = 'history', LS_KEY = 'mlx-serve-history';
const emptyDoc = () => ({ samples: [], rows: [] });

function sanitizeDoc(raw) {
  if (!raw || typeof raw !== 'object') return emptyDoc();
  const samples = [];
  for (const s of Array.isArray(raw.samples) ? raw.samples : []) {
    if (!s || !finite(s.t) || !s.c || typeof s.c !== 'object') continue;
    const c = {};
    for (const k of HISTORY_KEYS) c[k] = finite(s.c[k]) ? s.c[k] : 0;
    samples.push({ t: s.t, p: finite(s.p) ? s.p : null, m: typeof s.m === 'string' ? s.m : null, c });
  }
  const rows = (Array.isArray(raw.rows) ? raw.rows : [])
    .filter((r) => r && typeof r.key === 'string' && finite(r.startT)).map(mkRow);
  return { samples: mergeStores(samples, []), rows };
}

function idbRun(idb, mode, fn) {
  return new Promise((resolve, reject) => {
    const open = idb.open(DB_NAME, 1);
    open.onupgradeneeded = () => open.result.createObjectStore(DB_STORE);
    open.onerror = () => reject(open.error || new Error('indexedDB open failed'));
    open.onsuccess = () => {
      const db = open.result;
      try {
        const tx = db.transaction(DB_STORE, mode);
        const req = fn(tx.objectStore(DB_STORE));
        const fin = (v) => { try { db.close(); } catch (e) { /* closing is best effort */ } resolve(v); };
        if (mode === 'readonly') req.onsuccess = () => fin(req.result);
        else tx.oncomplete = () => fin();
        req.onerror = tx.onerror = tx.onabort = () => reject(new Error('indexedDB transaction failed'));
      } catch (e) { reject(e); }
    };
  });
}

async function loadDoc(env) {
  let raw;
  try { raw = await idbRun(env.indexedDB, 'readonly', (st) => st.get(DB_KEY)); } catch (e) { raw = undefined; }
  if (raw === undefined) {
    try { raw = JSON.parse(env.localStorage.getItem(LS_KEY)); } catch (e) { raw = null; }
  }
  try { return sanitizeDoc(raw); } catch (e) { return emptyDoc(); }
}

// Resolves to where the history landed: 'indexedDB', 'localStorage' or 'memory'.
async function saveDoc(env, doc) {
  try { await idbRun(env.indexedDB, 'readwrite', (st) => st.put(doc, DB_KEY)); return 'indexedDB'; } catch (e) { /* next backend */ }
  try { env.localStorage.setItem(LS_KEY, JSON.stringify(doc)); return 'localStorage'; } catch (e) { /* next backend */ }
  return 'memory';
}

const mergeDocs = (a, b, now) => ({ samples: mergeStores(a.samples, b.samples, now), rows: mergeRows(a.rows, b.rows) });

// Folds in what another tab saved before writing, so tabs add to one history.
async function persistDoc(env, doc, now) {
  const merged = mergeDocs(await loadDoc(env), doc, now);
  merged.where = await saveDoc(env, { samples: merged.samples, rows: merged.rows });
  return merged;
}

// Node (tests) sees no `document`; the browser sees no `globalThis.__mlxPanel`
// consumer. Either way the IIFE below only runs in a real page.
if (typeof globalThis !== 'undefined') globalThis.__mlxPanel = {
  computeRates, panelAt, makeSample, appendSample, mergeStores, rateOver, rateSeries, windowTotals,
  modelTotals, trackRequests, loadDoc, saveDoc, persistDoc, mergeDocs, HISTORY_KEYS, apiPrefix,
};

if (typeof document !== 'undefined') (function () {
  // Panel markup, injected into the page. A template literal, so the CSS/HTML
  // braces need no escaping — the reason this lives here and not inline in the
  // std.fmt-formatted index.html.
  const PANEL_HTML = `
<style>
.mhead{display:flex;align-items:center;gap:10px;margin:24px 0 10px}
.mhead h2{margin:0}
#m-status{font-size:0.6875rem;font-weight:600;letter-spacing:.02em;padding:2px 9px;border-radius:999px;background:#1a1e25;color:#7d8794}
#m-status.live{background:#0f2a17;color:#4ade80}
#m-status.err{background:#2a0f14;color:#ff95a8}
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
@media(max-width:640px){.mgrid{grid-template-columns:repeat(2,1fr)}}
.mtile{background:#0f1216;border:1px solid #1f242c;border-radius:8px;padding:12px 14px}
.mlbl{font-size:0.625rem;text-transform:uppercase;letter-spacing:.07em;color:#7d8794;margin-bottom:7px}
.mval{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.625rem;font-weight:700;line-height:1;color:#e6e9ee}
.munit{font-size:0.75rem;font-weight:400;color:#7d8794;margin-left:4px}
.msub{font-size:0.6875rem;color:#5b6470;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mbar{height:5px;background:#1f242c;border-radius:3px;margin-top:10px;overflow:hidden}
.mfill{height:100%;width:0;border-radius:3px;background:#3b82f6;transition:width .5s}
.mfill.warn{background:#f59e0b}.mfill.crit{background:#ef4444}
.mspark{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
@media(max-width:640px){.mspark{grid-template-columns:1fr}}
.msparkbox{background:#0f1216;border:1px solid #1f242c;border-radius:8px;padding:10px 12px}
.msparkhead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.msparkval{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.9375rem;font-weight:700;color:#e6e9ee}
.msparkbox svg{width:100%;height:44px;display:block}
:root[data-theme=light] .mbase{stroke:var(--line)}
:root[data-theme=light] .mtile,:root[data-theme=light] .msparkbox{background:#fff;border-color:#e1e2e6}
:root[data-theme=light] .mval,:root[data-theme=light] .msparkval{color:#1e1f22}
:root[data-theme=light] .mlbl,:root[data-theme=light] .munit{color:#5b616b}
:root[data-theme=light] .msub{color:#878d96}
:root[data-theme=light] .mbar{background:#e7e8ec}
:root[data-theme=light] #m-status{background:#ececf0;color:#5b616b}
:root[data-theme=light] #m-status.live{background:#e3f5ee;color:#0f7b5f}
:root[data-theme=light] #m-status.err{background:#fdeceb;color:#b3261e}
.mwin{display:flex;gap:6px;margin-left:auto}
.mwin button{font:inherit;font-size:0.6875rem;padding:2px 9px;border-radius:999px;border:1px solid #1f242c;background:transparent;color:#7d8794;cursor:pointer}
.mwin button.on{background:#1a1e25;color:#e6e9ee}
.mhist{margin-top:16px}
.mhist .mhead{margin:0 0 10px}
.mhist .mnote{font-size:0.6875rem;color:#5b6470;margin-top:8px}
.mtot{font-size:0.75rem;color:#7d8794;margin-top:10px}
:root[data-theme=light] .mwin button{border-color:#e1e2e6;color:#5b616b}
:root[data-theme=light] .mwin button.on{background:#ececf0;color:#1e1f22}
:root[data-theme=light] .mhist .mnote{color:#878d96}
.msess{margin-top:12px;overflow-x:auto}
.msess table{width:100%;border-collapse:collapse;font-size:0.75rem}
.msess th{text-align:left;font-weight:600;font-size:0.625rem;text-transform:uppercase;letter-spacing:.07em;color:#7d8794;padding:0 8px 6px 0}
.msess td{padding:6px 8px 6px 0;border-top:1px solid #1f242c;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#e6e9ee;white-space:nowrap}
.msess td.mmodel{font-family:inherit;max-width:260px;overflow:hidden;text-overflow:ellipsis}
.msess td.mctx{width:34%}
.msess .mbar{margin-top:4px}
.msess .mempty{color:#5b6470;font-size:0.75rem}
:root[data-theme=light] .msess td{color:#1e1f22;border-color:#e1e2e6}
</style>
<div class=mhead><h2 style="margin:0" data-i18n="Live metrics">Live metrics</h2><span id=m-status data-i18n="connecting…">connecting…</span></div>
<div class=card style="padding:16px">
<div class=mgrid>
<div class=mtile><div class=mlbl data-i18n="Decode">Decode</div><div class=mval><span id=m-decode-tps>—</span><span class=munit>tok/s</span></div><div class=msub id=m-decode-ms data-i18n="— ms avg">— ms avg</div></div>
<div class=mtile><div class=mlbl data-i18n="Prefill">Prefill</div><div class=mval><span id=m-prefill-tps>—</span><span class=munit>tok/s</span></div><div class=msub id=m-prefill-ms data-i18n="— ms avg">— ms avg</div><div class=mbar><div class=mfill id=m-prefillbar></div></div></div>
<div class=mtile><div class=mlbl data-i18n="Requests">Requests</div><div class=mval><span id=m-running>0</span><span class=munit data-i18n="running">running</span></div><div class=msub id=m-waiting data-i18n="0 waiting · — req/s">0 waiting · — req/s</div></div>
<div class=mtile><div class=mlbl data-i18n="Avg TTFT">Avg TTFT</div><div class=mval><span id=m-ttft>—</span><span class=munit>ms</span></div><div class=msub id=m-e2e data-i18n="— ms e2e">— ms e2e</div></div>
<div class=mtile><div class=mlbl data-i18n="Cache hit rate">Cache hit rate</div><div class=mval><span id=m-cache>—</span><span class=munit>%</span></div><div class=msub id=m-cachedetail data-i18n="— / — queries">— / — queries</div></div>
<div class=mtile><div class=mlbl>GPU</div><div class=mval><span id=m-gpu>0</span><span class=munit>%</span></div><div class=mbar><div class=mfill id=m-gpubar></div></div></div>
<div class=mtile><div class=mlbl data-i18n="Memory">Memory</div><div class=mval><span id=m-mem>0</span><span class=munit>MB</span></div><div class=msub id=m-memdetail data-i18n="physical footprint">physical footprint</div></div>
<div class=mtile><div class=mlbl data-i18n="Generated">Generated</div><div class=mval><span id=m-gen>0</span><span class=munit>tok</span></div><div class=msub id=m-success data-i18n="0 requests">0 requests</div></div>
</div>
<div class=mspark>
<div class=msparkbox><div class=msparkhead><span class=mlbl data-i18n="Decode tok/s · last 60s">Decode tok/s · last 60s</span><span class=msparkval id=m-spark-decode-val>—</span></div><svg id=m-spark-decode viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
<div class=msparkbox><div class=msparkhead><span class=mlbl data-i18n="Prefill tok/s · last 60s">Prefill tok/s · last 60s</span><span class=msparkval id=m-spark-prefill-val>—</span></div><svg id=m-spark-prefill viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
</div>
<div class="msparkbox msess"><div class=mlbl data-i18n="Sessions">Sessions</div><div id=m-sessions></div></div>
</div>
<div class=mhist>
<div class=mhead><h2 style="margin:0" data-i18n="History">History</h2><div class=mwin id=m-win><button data-win=3600000 class=on data-i18n="1 hour">1 hour</button><button data-win=21600000 data-i18n="6 hours">6 hours</button><button data-win=86400000 data-i18n="24 hours">24 hours</button></div></div>
<div class=mspark>
<div class=msparkbox><div class=msparkhead><span class=mlbl data-i18n="Generated tok/s">Generated tok/s</span><span class=msparkval id=m-h-gen-val>—</span></div><svg id=m-h-gen viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
<div class=msparkbox><div class=msparkhead><span class=mlbl data-i18n="Prefill tok/s">Prefill tok/s</span><span class=msparkval id=m-h-pre-val>—</span></div><svg id=m-h-pre viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
<div class=msparkbox><div class=msparkhead><span class=mlbl data-i18n="Requests / min">Requests / min</span><span class=msparkval id=m-h-req-val>—</span></div><svg id=m-h-req viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
<div class=msparkbox><div class=msparkhead><span class=mlbl data-i18n="Failed + rejected + cancelled / min">Failed + rejected + cancelled / min</span><span class=msparkval id=m-h-bad-val>—</span></div><svg id=m-h-bad viewBox="0 0 300 44" preserveAspectRatio="none"></svg></div>
</div>
<div class=mtot id=m-h-tot></div>
<div class=mtot id=m-h-since></div>
<div class="msparkbox msess"><div class=mlbl data-i18n="By model">By model</div><div id=m-h-models></div></div>
<div class="msparkbox msess"><div class=mlbl data-i18n="Request history">Request history</div><div id=m-h-rows></div></div>
<div class=mnote data-i18n="Requests shorter than one poll (1 s) appear only in the totals. Gaps in a chart mean the server restarted or was unreachable.">Requests shorter than one poll (1 s) appear only in the totals. Gaps in a chart mean the server restarted or was unreachable.</div>
<div class=mnote id=m-h-where></div>
</div>
</div>`;

  // The panel brings its own markup, so the language boot cannot know about it:
  // translate it here, and follow later switches.
  const I18N = (typeof window !== 'undefined' && window.mlxI18n) ? window.mlxI18n : null;
  const t = (key, params) => (I18N ? I18N.t(key, params) : key);

  const mount = document.getElementById('mlx-metrics');
  if (mount) mount.innerHTML = PANEL_HTML;
  if (I18N && mount) I18N.applyMarkup(mount);

  const $ = (id) => document.getElementById(id);
  const samples = [];              // {t, gen, dsum, prompt, psum, req} ring buffer
  const RETAIN_MS = 120000;        // keep 2 min of samples for the 60s window
  const SPARK_N = 60;              // sparkline points (≈60s at 1 Hz)
  const decodeHist = [], prefillHist = [];
  const hover = { decode: null, prefill: null };  // hovered point index per chart

  function fmt(v, d) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    return v.toFixed(d === undefined ? 1 : d);
  }

  // The status text is never table copy (it carries a fetch error verbatim), so
  // the slot's data-i18n must go with it — otherwise a language switch would
  // put "connecting…" back over a live feed.
  // A value slot also carries a static placeholder in the markup ("— ms avg"),
  // so writing a number has to take that key away: a language switch must
  // re-render data by re-running the tick, not by restoring the placeholder.
  function setVal(id, txt) {
    const e = $(id);
    if (e) { e.removeAttribute('data-i18n'); e.textContent = txt; }
  }

  function setStatus(cls, txt) {
    const e = $('m-status');
    if (e) { e.removeAttribute('data-i18n'); e.className = cls; e.textContent = txt; }
  }


  // Draw a sparkline (auto-scaled) into an <svg>, set its value label, and — when
  // the mouse is over that chart (hover[key] set) — draw a marker at the hovered
  // point and show that point's value in the label instead of the latest.
  function spark(id, data, color, valId, key, dec) {
    const svg = $(id), label = $(valId);
    if (!svg) return;
    if (data.length < 2) {
      svg.innerHTML = '';
      if (label) label.textContent = data.length ? fmt(data[0], dec) : '—';
      return;
    }
    const max = Math.max.apply(null, data.concat([0.001]));
    const n = data.length, W = 300, H = 44, p = 3;
    const xs = new Array(n), ys = new Array(n);
    let pts = '';
    for (let i = 0; i < n; i++) {
      xs[i] = p + (i / (n - 1)) * (W - 2 * p);
      ys[i] = H - p - (data[i] / max) * (H - 2 * p);
      pts += (i ? ' ' : '') + xs[i].toFixed(1) + ',' + ys[i].toFixed(1);
    }
    let m =
      '<polyline points="' + pts + '" fill="none" stroke="' + color +
      '" stroke-width="1.5" stroke-linejoin="round"/>' +
      '<line class=mbase x1="' + p + '" y1="' + (H - 1) + '" x2="' + (W - p) + '" y2="' + (H - 1) +
      '" stroke="#1f242c" stroke-width="1"/>';
    const hi = hover[key];
    if (hi !== null && hi >= 0 && hi < n) {
      m += '<line x1="' + xs[hi].toFixed(1) + '" y1="' + p + '" x2="' + xs[hi].toFixed(1) +
        '" y2="' + (H - 1) + '" stroke="' + color + '" stroke-width="1" opacity="0.35"/>' +
        '<circle cx="' + xs[hi].toFixed(1) + '" cy="' + ys[hi].toFixed(1) + '" r="2.5" fill="' + color + '"/>';
      if (label) label.textContent = fmt(data[hi], dec);
    } else if (label) {
      label.textContent = fmt(data[n - 1], dec);
    }
    svg.innerHTML = m;
  }

  // Wire mouse hover on a sparkline: map cursor x → nearest data point, mark it,
  // show that value in the chart's label; mouseleave restores the latest value.
  function attachSparkHover(id, key, getData, color, valId, dec) {
    const svg = $(id);
    if (!svg) return;
    svg.style.cursor = 'crosshair';
    svg.addEventListener('mousemove', function (e) {
      const data = getData();
      if (data.length < 2) return;
      const rect = svg.getBoundingClientRect();
      const frac = rect.width ? (e.clientX - rect.left) / rect.width : 0;
      hover[key] = Math.max(0, Math.min(data.length - 1, Math.round(frac * (data.length - 1))));
      spark(id, data, color, valId, key, dec);
    });
    svg.addEventListener('mouseleave', function () {
      hover[key] = null;
      spark(id, getData(), color, valId, key, dec);
    });
  }

  function fmtBytes(b) {
    if (!b) return '0';
    return b >= 1073741824 ? (b / 1073741824).toFixed(1) + ' GB' : (b / 1048576).toFixed(0) + ' MB';
  }

  // Built with DOM nodes, never innerHTML: the model id is a folder name.
  function renderSessions(list) {
    const box = $('m-sessions');
    if (!box) return;
    box.textContent = '';
    if (!list || list.length === 0) {
      const e = document.createElement('div');
      e.className = 'mempty';
      e.textContent = t('No sessions');
      box.appendChild(e);
      return;
    }
    const table = document.createElement('table');
    const head = table.insertRow();
    for (const h of ['Model', 'Phase', 'Context', 'Cached', 'Generated', 'KV + state']) {
      const th = document.createElement('th');
      th.textContent = t(h);
      head.appendChild(th);
    }
    const idle = (s) => (s.phase === 'cached' ? 1 : 0);
    const rows = list.slice().sort((a, b) => a.model.localeCompare(b.model) || idle(a) - idle(b));
    for (const s of rows) {
      const tr = table.insertRow();
      const cell = (txt, cls) => { const td = tr.insertCell(); td.textContent = txt; if (cls) td.className = cls; return td; };
      cell(s.model, 'mmodel').title = s.model;
      cell(t({ prefill: 'prefilling', decode: 'decoding', cached: 'in cache' }[s.phase] || s.phase));
      const pct = s.context_length > 0 ? Math.min(100, (s.context_tokens / s.context_length) * 100) : null;
      const ctx = cell(fmt(s.context_tokens, 0) + (s.context_length > 0 ? ' / ' + fmt(s.context_length, 0) + ' · ' + pct.toFixed(0) + '%' : ''), 'mctx');
      if (pct !== null) {
        const bar = document.createElement('div'); bar.className = 'mbar';
        const fill = document.createElement('div');
        fill.className = 'mfill' + (pct >= 90 ? ' crit' : pct >= 70 ? ' warn' : '');
        fill.style.width = pct + '%';
        bar.appendChild(fill); ctx.appendChild(bar);
      }
      cell(fmt(s.cached_tokens, 0));
      cell(s.phase === 'cached' ? '—' : fmt(s.generated_tokens, 0));
      cell(fmtBytes(s.state_bytes));
    }
    box.appendChild(table);
  }

  // ── History view ───────────────────────────────────────────────────────────
  let doc = { samples: [], rows: [] };
  let winMs = 3600000, storedIn = null;
  const HIST_N = 60;

  // Draws a series with gaps: a null breaks the line.
  function drawSeries(id, vals, color, valId, scale, dec) {
    const svg = $(id), label = $(valId);
    if (!svg) return;
    const W = 300, H = 44, p = 3, n = vals.length;
    const max = Math.max.apply(null, vals.map((v) => (v === null ? 0 : v * scale)).concat([0.001]));
    let m = '<line class=mbase x1="' + p + '" y1="' + (H - 1) + '" x2="' + (W - p) + '" y2="' + (H - 1) + '" stroke="#1f242c" stroke-width="1"/>';
    let run = '', last = null;
    const flush = () => { if (run) m += '<polyline points="' + run + '" fill="none" stroke="' + color + '" stroke-width="1.5" stroke-linejoin="round"/>'; run = ''; };
    for (let i = 0; i < n; i++) {
      if (vals[i] === null) { flush(); continue; }
      last = vals[i] * scale;
      const x = p + (i / Math.max(1, n - 1)) * (W - 2 * p), y = H - p - (last / max) * (H - 2 * p);
      run += (run ? ' ' : '') + x.toFixed(1) + ',' + y.toFixed(1);
    }
    flush();
    svg.innerHTML = m;
    if (label) label.textContent = last === null ? '—' : fmt(last, dec);
  }

  function renderHistoryTable(box, heads, rows, emptyKey) {
    box.textContent = '';
    if (!rows.length) {
      const e = document.createElement('div');
      e.className = 'mempty';
      e.textContent = t(emptyKey);
      box.appendChild(e);
      return;
    }
    const table = document.createElement('table');
    const head = table.insertRow();
    for (const h of heads) { const th = document.createElement('th'); th.textContent = t(h); head.appendChild(th); }
    for (const cells of rows) {
      const tr = table.insertRow();
      cells.forEach((txt, i) => { const td = tr.insertCell(); td.textContent = txt; if (heads[i] === 'Model') td.className = 'mmodel'; });
    }
    box.appendChild(table);
  }

  const clock = (ts) => new Date(ts).toLocaleTimeString();
  const secs = (ms) => (ms < 60000 ? Math.max(1, Math.round(ms / 1000)) + ' s' : Math.round(ms / 60000) + ' min');

  function renderHistory(now) {
    const from = now - winMs, st = doc.samples;
    const gen = rateSeries(st, 'generation_tokens_total', from, now, HIST_N);
    const pre = rateSeries(st, 'prefill_tokens_total', from, now, HIST_N);
    const req = rateSeries(st, 'requests_success_total', from, now, HIST_N);
    const bad = ['requests_failed_total', 'requests_rejected_total', 'requests_cancelled_total']
      .map((k) => rateSeries(st, k, from, now, HIST_N))
      .reduce((a, b) => a.map((v, i) => (v === null || b[i] === null ? null : v + b[i])));
    drawSeries('m-h-gen', gen, '#22c55e', 'm-h-gen-val', 1, 1);
    drawSeries('m-h-pre', pre, '#3b82f6', 'm-h-pre-val', 1, 0);
    drawSeries('m-h-req', req, '#a78bfa', 'm-h-req-val', 60, 1);
    drawSeries('m-h-bad', bad, '#ef4444', 'm-h-bad-val', 60, 1);

    const w = windowTotals(st, from, now);
    setVal('m-h-tot', t('In this window: %@ ok · %@ failed · %@ rejected · %@ cancelled',
      [w.requests_success_total, w.requests_failed_total, w.requests_rejected_total, w.requests_cancelled_total]));
    const last = st.length ? st[st.length - 1].c : null;
    setVal('m-h-since', last ? t('Since startup: %@ ok · %@ failed · %@ rejected · %@ cancelled',
      [last.requests_success_total, last.requests_failed_total, last.requests_rejected_total, last.requests_cancelled_total]) : '');

    const mt = modelTotals(st, from, now);
    const names = Object.keys(mt).sort();
    renderHistoryTable($('m-h-models'), ['Model', 'Requests', 'Generated', 'Prefill'],
      names.map((n) => [n || t('unattributed'), fmt(mt[n].requests_success_total, 0), fmt(mt[n].generation_tokens_total, 0), fmt(mt[n].prefill_tokens_total, 0)]),
      'No requests yet');

    const rows = doc.rows.slice().reverse().slice(0, 50);
    renderHistoryTable($('m-h-rows'), ['Started', 'Client', 'Model', 'Phase', 'Duration', 'Context', 'Cached', 'Generated'],
      rows.map((r) => [clock(r.startT), r.client, r.model,
        r.endT === null ? t({ prefill: 'prefilling', decode: 'decoding' }[r.phase] || r.phase) : t('finished'),
        secs((r.endT === null ? now : r.endT) - r.startT), fmt(r.ctx, 0), fmt(r.cached, 0), fmt(r.generated, 0)]),
      'No requests yet');

    const where = $('m-h-where');
    if (where && storedIn) {
      where.removeAttribute('data-i18n');
      where.textContent = storedIn === 'memory'
        ? t('History is kept in memory only and is lost on reload.')
        : t('History is kept in this browser (%@).', [storedIn]);
    }
  }

  let lastSave = 0, saving = false;
  async function save(now) {
    if (saving) return;
    saving = true;
    try {
      const merged = await persistDoc(window, doc, now);
      storedIn = merged.where;
      doc = { samples: mergeStores(doc.samples, merged.samples, now), rows: mergeRows(doc.rows, merged.rows) };
    } catch (e) { storedIn = 'memory'; }
    saving = false;
    lastSave = now;
  }

  const winBox = $('m-win');
  if (winBox) winBox.addEventListener('click', function (e) {
    const b = e.target.closest('button[data-win]');
    if (!b) return;
    winMs = Number(b.getAttribute('data-win'));
    for (const x of winBox.querySelectorAll('button')) x.className = x === b ? 'on' : '';
    renderHistory(Date.now());
  });
  document.addEventListener('visibilitychange', function () { if (document.visibilityState === 'hidden') save(Date.now()); });
  loadDoc(window).then(function (stored) {
    doc = mergeDocs(stored, doc, Date.now());
    renderHistory(Date.now());
  });

  const histSum = (hist) => (hist && typeof hist.sum === 'number') ? hist.sum : 0;

  let ticking = false;
  async function tick() {
    // A hanging /metrics.json must not pile up requests: each new fetch occupies
    // one per-origin TCP connection, and exhausting the limit (6-8) freezes all
    // network I/O to the server origin including chat/completions. Never start a
    // new fetch while one is in flight, and bound each one with a timeout.
    if (ticking) return;
    ticking = true;
    let d;
    try {
      const r = await fetch(apiPrefix(location.pathname) + '/metrics.json', {
        cache: 'no-store',
        signal: AbortSignal.timeout(5000),
      });
      if (r.status === 503) { setStatus('err', t('metrics disabled')); return; }
      if (!r.ok) throw new Error('HTTP ' + r.status);
      d = await r.json();
    } catch (e) { setStatus('err', t('error: %@', [e.message])); return; }
    finally { ticking = false; }

    setStatus('live', t('● live'));
    const c = d.counters, g = d.gauges, h = d.histograms;
    const now = Date.now();

    const psum = histSum(h.prefill_time_seconds);
    const liveTok = (g.generation_tokens_live != null) ? g.generation_tokens_live : c.generation_tokens_total;
    const livePre = (g.prefill_tokens_live != null) ? g.prefill_tokens_live : 0;
    samples.push({ t: now, live: liveTok, pre: livePre, pretok: c.prefill_tokens_total, psum: psum, req: c.requests_success_total });
    while (samples.length > 2 && now - samples[0].t > RETAIN_MS) samples.shift();

    // Everything displayed is derived here, from THIS tick's data. Nothing is
    // remembered between ticks — see the note above `computeRates`.
    const r = computeRates(now, samples, c, g, psum);
    const { decodeTps, prefillTps, avgPrefillTps, reqRate, prefilling } = r;

    // Sparkline history: both series dip to 0 when their phase is idle.
    decodeHist.push(decodeTps); if (decodeHist.length > SPARK_N) decodeHist.shift();
    prefillHist.push(prefillTps); if (prefillHist.length > SPARK_N) prefillHist.shift();

    // Average latency from each histogram's sum/count (seconds → ms).
    const avgMs = (hist) => (hist && hist.count > 0) ? (hist.sum / hist.count) * 1000 : null;
    const ttft = avgMs(h.time_to_first_token_seconds);
    const e2e = avgMs(h.e2e_request_latency_seconds);
    const decodeMs = avgMs(h.decode_time_seconds);
    const prefillMs = avgMs(h.prefill_time_seconds);

    const cq = c.prefix_cache_queries_total, ch = c.prefix_cache_hits_total;
    const cachePct = cq > 0 ? Math.round((ch / cq) * 100) : null;
    // Token-level reuse: what fraction of billed prompt tokens never reached the
    // GPU. This is the number that explains a low prefill tok/s on warm turns.
    const tokTotal = c.prompt_tokens_total;
    const tokPct = tokTotal > 0 ? Math.round((c.prefix_cache_tokens_total / tokTotal) * 100) : null;

    $('m-decode-tps').textContent = fmt(decodeTps, 1);
    setVal('m-decode-ms', t('%@ ms avg', [decodeMs !== null ? fmt(decodeMs, 0) : '—']));
    // Big number = live prefill speed, 0 when not prefilling (mirrors Decode).
    $('m-prefill-tps').textContent = fmt(prefillTps, 0);
    // Sub-line doubles as the phase indicator AND carries the stable average, so
    // "0 tok/s while decoding" never means "I don't know how fast prefill is".
    // The phase flag flips at prefill START; the token count appears once the
    // first chunk lands (and never for ds4/llama, which prefill elsewhere).
    setVal('m-prefill-ms', prefilling
      ? (t('prefilling') + (r.livePre > 0 ? ' · ' + fmt(r.livePre, 0) + ' tok' : ''))
      : (avgPrefillTps !== null
          ? t('%@ tok/s avg · %@ ms', [fmt(avgPrefillTps, 0), prefillMs !== null ? fmt(prefillMs, 0) : '—'])
          : t('%@ ms avg', ['—'])));
    const pexp = g.prefill_tokens_expected || 0;
    $('m-prefillbar').style.width = (prefilling && pexp > 0 ? Math.min(100, (r.livePre / pexp) * 100) : 0) + '%';
    $('m-running').textContent = g.requests_running;
    setVal('m-waiting', t('%@ waiting · %@ req/s', [g.requests_waiting, reqRate !== null ? fmt(reqRate, 2) : '—'])
      + (g.batched_group_size > 1 ? ' · ' + t('batch of %@', [g.batched_group_size]) : '')
      + (c.requests_cancelled_total > 0 ? ' · ' + t('%@ cancelled', [c.requests_cancelled_total]) : ''));
    $('m-ttft').textContent = ttft !== null ? fmt(ttft, 0) : '—';
    setVal('m-e2e', t('%@ ms e2e', [e2e !== null ? fmt(e2e, 0) : '—']));
    $('m-cache').textContent = cachePct !== null ? cachePct : '—';
    setVal('m-cachedetail', t('%@ / %@ queries', [ch, cq])
      + (tokPct !== null ? ' · ' + t('%@% tokens reused', [tokPct]) : ''));

    const gp = g.gpu_utilization_pct;
    $('m-gpu').textContent = gp;
    const bar = $('m-gpubar');
    bar.style.width = gp + '%';
    bar.className = 'mfill' + (gp >= 90 ? ' crit' : gp >= 70 ? ' warn' : '');

    $('m-mem').textContent = g.memory_mb;
    setVal('m-memdetail', t('MLX %@ active · %@ pool', [fmtBytes(g.mlx_active_bytes), fmtBytes(g.mlx_cache_bytes)]));
    // Live count (completed + in-flight) so it moves during a running request.
    $('m-gen').textContent = fmt(liveTok, 0);
    setVal('m-success', t('%@ requests', [fmt(c.requests_success_total, 0)]));

    spark('m-spark-decode', decodeHist, '#22c55e', 'm-spark-decode-val', 'decode', 1);
    spark('m-spark-prefill', prefillHist, '#3b82f6', 'm-spark-prefill-val', 'prefill', 0);
    renderSessions(d.sessions);

    const sample = makeSample(now, d);
    doc = {
      samples: appendSample(doc.samples, sample, now),
      rows: trackRequests(doc.rows, d.sessions, sample.t),
    };
    renderHistory(now);
    if (now - lastSave >= 15000) save(now);
  }

  attachSparkHover('m-spark-decode', 'decode', function () { return decodeHist; }, '#22c55e', 'm-spark-decode-val', 1);
  attachSparkHover('m-spark-prefill', 'prefill', function () { return prefillHist; }, '#3b82f6', 'm-spark-prefill-val', 0);
  if (I18N) I18N.onChange(function () { if (mount) I18N.applyMarkup(mount); tick(); });

  tick();
  setInterval(tick, 1000);
})();
