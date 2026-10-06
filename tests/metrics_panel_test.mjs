// Unit tests for the index panel's rate math (`src/html/metrics.js`).
//
// The panel is an untestable surface (DOM + polling), so the decision logic is
// factored into a pure `computeRates(now, samples, counters, gauges, psum)` and
// tested here. Run via `node tests/metrics_panel_test.mjs`; skipped when node
// is absent.
//
// Regression (2026-07-09): `lastPrefillTps` was a module-level `let` that was
// only ASSIGNED when the 60s window contained prefill work, and never reset. So
// the Prefill tile kept displaying the last prefill speed for the whole of a
// long decode — and a page refresh cleared it, which is the signature of state
// living in a variable rather than being derived from the current data.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
// The page's boot slot evaluates api.js before this script; the panel has no
// prefix logic of its own, so the harness keeps that order.
new Function(readFileSync(join(here, '..', 'src', 'html', 'api.js'), 'utf8'))();
const src = readFileSync(join(here, '..', 'src', 'html', 'metrics.js'), 'utf8');

// The file guards its IIFE on `typeof document`, so in node only the top-level
// helpers evaluate. It hands them back through `globalThis.__mlxPanel`.
new Function(src)();
const { computeRates, apiPrefix } = globalThis.__mlxPanel ?? {};
assert.ok(computeRates, 'metrics.js must expose computeRates for tests');

const counters = (over = {}) => ({
  prompt_tokens_total: 10000,
  prefill_tokens_total: 4000,
  prefix_cache_tokens_total: 6000,
  generation_tokens_total: 500,
  requests_success_total: 4,
  requests_cancelled_total: 0,
  requests_failed_total: 0,
  requests_rejected_total: 0,
  prefix_cache_queries_total: 4,
  prefix_cache_hits_total: 2,
  ...over,
});
const gauges = (over = {}) => ({
  requests_running: 0,
  requests_waiting: 0,
  gpu_utilization_pct: 0,
  memory_mb: 0,
  generation_tokens_live: 500,
  prefill_tokens_live: 0,
  requests_prefilling: 0,
  ...over,
});

let failures = 0;
const pending = [];
function test(name, fn) {
  pending.push((async () => {
    try { await fn(); console.log(`  PASS ${name}`); }
    catch (e) { failures++; console.log(`  FAIL ${name}\n       ${e.message}`); }
  })());
}

// ── The bug ──────────────────────────────────────────────────────────────────
test('prefill tok/s is 0 while decoding, with no carry-forward from an earlier prefill', () => {
  const now0 = 1_000_000;
  const samples = [];

  // t0: a prefill is running, 8192 tokens forwarded.
  samples.push({ t: now0, live: 100, pre: 0, pretok: 4000, psum: 5.0, req: 4 });
  // t1 (+4s): prefill has advanced to 16384.
  samples.push({ t: now0 + 4000, live: 100, pre: 8192, pretok: 4000, psum: 5.0, req: 4 });
  const mid = computeRates(now0 + 8000, samples, counters(), gauges({
    requests_running: 1, requests_prefilling: 1, prefill_tokens_live: 16384,
  }), 5.0);
  assert.ok(mid.prefillTps > 0, `expected a live prefill rate, got ${mid.prefillTps}`);
  assert.equal(mid.prefilling, true);

  // t2 (+10s): prefill ended, the request is now DECODING. The server has
  // already zeroed both prefill gauges (verified in test_metrics.sh Phase 4).
  samples.push({ t: now0 + 8000, live: 100, pre: 16384, pretok: 4000, psum: 5.0, req: 4 });
  const dec = computeRates(now0 + 10000, samples, counters(), gauges({
    requests_running: 1, requests_prefilling: 0, prefill_tokens_live: 0,
    generation_tokens_live: 900,
  }), 5.0);

  assert.equal(dec.prefillTps, 0,
    `Prefill must read 0 while decoding; got ${dec.prefillTps} (carry-forward)`);
  assert.equal(dec.prefilling, false);
  assert.ok(dec.decodeTps > 0, `expected a live decode rate, got ${dec.decodeTps}`);
});

test('a page refresh and a live tick agree (no hidden state across ticks)', () => {
  const now = 2_000_000;
  const c = counters(), g = gauges({ requests_running: 1, generation_tokens_live: 900 });

  // "Warm" panel: a long history including a finished prefill burst.
  const warm = [
    { t: now - 60000, live: 0,   pre: 0,     pretok: 0,    psum: 0.0, req: 0 },
    { t: now - 30000, live: 100, pre: 8192,  pretok: 2000, psum: 2.0, req: 2 },
    { t: now - 2000,  live: 500, pre: 0,     pretok: 4000, psum: 5.0, req: 4 },
  ];
  // "Fresh" panel, as after F5: only the samples gathered since load.
  const fresh = [{ t: now - 2000, live: 500, pre: 0, pretok: 4000, psum: 5.0, req: 4 }];

  const a = computeRates(now, warm, c, g, 5.0);
  const b = computeRates(now, fresh, c, g, 5.0);
  assert.equal(a.prefillTps, b.prefillTps,
    `refresh changed the prefill reading (${b.prefillTps}) vs live (${a.prefillTps})`);
  assert.equal(a.prefillTps, 0);
});

// ── The number that replaces it ──────────────────────────────────────────────
test('average prefill speed = forwarded tokens / seconds spent prefilling', () => {
  const r = computeRates(3_000_000, [{ t: 2_999_000, live: 0, pre: 0, pretok: 0, psum: 0, req: 0 }],
    counters({ prefill_tokens_total: 4000 }), gauges(), 5.0);
  assert.equal(r.avgPrefillTps, 800);   // 4000 forwarded / 5.0 s
});

test('average prefill speed excludes prefix-cache restores', () => {
  // 10000 billed, 6000 restored -> only 4000 were forwarded. Using the billed
  // total would report 2000 tok/s (the 10.6x class of bug).
  const r = computeRates(3_000_000, [{ t: 2_999_000, live: 0, pre: 0, pretok: 0, psum: 0, req: 0 }],
    counters(), gauges(), 5.0);
  assert.equal(r.avgPrefillTps, 800);
  assert.notEqual(r.avgPrefillTps, 2000);
});

test('average is null before any request has completed (shows an em dash)', () => {
  const r = computeRates(4_000_000, [{ t: 3_999_000, live: 0, pre: 0, pretok: 0, psum: 0, req: 0 }],
    counters({ prefill_tokens_total: 0, requests_success_total: 0 }), gauges(), 0);
  assert.equal(r.avgPrefillTps, null);
});

test('decode tok/s is 0 when nothing is running', () => {
  const now = 5_000_000;
  const samples = [
    { t: now - 4000, live: 100, pre: 0, pretok: 4000, psum: 5.0, req: 4 },
    { t: now,        live: 500, pre: 0, pretok: 4000, psum: 5.0, req: 4 },
  ];
  const r = computeRates(now, samples, counters(), gauges({ requests_running: 0 }), 5.0);
  assert.equal(r.decodeTps, 0);
});

// ── Browser-held history (the console keeps what Prometheus would) ───────────
const P = globalThis.__mlxPanel;
const T0 = 1_700_000_000_000;
const poll = (t, over = {}, g = {}, sessions = []) =>
  P.makeSample(t, { counters: counters(over), gauges: gauges(g), sessions });
const rowsOf = (...a) => P.trackRequests(...a);

// In-memory IndexedDB with the one object store and the request/transaction
// shape the store uses, so a "reload" is a second env over the same data.
function fakeIdb(shared = new Map(), { failOpen = false } = {}) {
  return {
    shared,
    open() {
      const rq = {};
      setTimeout(() => {
        if (failOpen) { rq.error = new Error('denied'); rq.onerror?.(); return; }
        const db = {
          createObjectStore() {},
          transaction() {
            const tx = {
              objectStore: () => ({
                get: (k) => { const r = {}; setTimeout(() => { r.result = shared.get(k); r.onsuccess?.(); }); return r; },
                put: (v, k) => { const r = {}; shared.set(k, structuredClone(v)); setTimeout(() => { r.onsuccess?.(); tx.oncomplete?.(); }); return r; },
              }),
            };
            return tx;
          },
          close() {},
        };
        rq.result = db;
        rq.onupgradeneeded?.();
        rq.onsuccess?.();
      });
      return rq;
    },
  };
}
const fakeLs = (init = {}, { throwSet = false } = {}) => {
  const m = new Map(Object.entries(init));
  return { m, getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => { if (throwSet) throw new Error('quota'); m.set(k, v); } };
};
const EMPTY = { samples: [], rows: [] };

test('row 10: a server restart draws a gap, never a negative rate or a delta across the reset', () => {
  for (const withStart of [true, false]) {
    const g = (p) => (withStart ? { process_start_time_seconds: p } : {});
    const store = [
      poll(T0 + 0, { generation_tokens_total: 1000 }, g(100)),
      poll(T0 + 1000, { generation_tokens_total: 1100 }, g(100)),
      poll(T0 + 2000, { generation_tokens_total: 1200 }, g(100)),
      poll(T0 + 3000, { generation_tokens_total: 0 }, g(200)),
      poll(T0 + 4000, { generation_tokens_total: 50 }, g(200)),
      poll(T0 + 5000, { generation_tokens_total: 150 }, g(200)),
    ];
    const series = P.rateSeries(store, 'generation_tokens_total', T0, T0 + 5000, 5);
    assert.deepEqual(series, [100, 100, null, 50, 100], `withStart=${withStart}`);
    assert.ok(series.every((v) => v === null || v >= 0));
    assert.equal(P.rateOver(store, 'generation_tokens_total', 60000, T0 + 5000).rate, 75);
    assert.equal(P.windowTotals(store, T0, T0 + 5000).generation_tokens_total, 350);
  }
});

test('row 10: a restart that out-counts the old process is still a gap once downsampled', () => {
  let store = [];
  const now = T0 + 3 * 3600_000;
  store = P.appendSample(store, poll(T0, { requests_success_total: 500 }), now);
  store = P.appendSample(store, poll(T0 + 1000, { requests_success_total: 3 }), now);
  store = P.appendSample(store, poll(T0 + 2000, { requests_success_total: 900 }), now);
  store = P.appendSample(store, poll(T0 + 60_000, { requests_success_total: 901 }), now);
  assert.equal(P.windowTotals(store, T0, T0 + 60_000).requests_success_total, 898, 'the 500 to 3 step is dropped, the rest is kept');
});

test('row 11: a reload restores the history that a tab saved', async () => {
  const idb = fakeIdb();
  const doc = {
    samples: [poll(T0, { requests_success_total: 4 }), poll(T0 + 1000, { requests_success_total: 5 })],
    rows: rowsOf([], [{ model: 'm', phase: 'decode', request_id: 7, client: 'codex' }], T0),
  };
  await P.saveDoc({ indexedDB: idb }, doc);
  const back = await P.loadDoc({ indexedDB: fakeIdb(idb.shared) });
  assert.deepEqual(back, doc);
});

test('row 11: a corrupt or truncated store loads as an empty history', async () => {
  const garbageIdb = fakeIdb(new Map([['history', 'not an object']]));
  assert.deepEqual(await P.loadDoc({ indexedDB: garbageIdb }), EMPTY);
  const truncated = fakeLs({ 'mlx-serve-history': '{"samples":[{"t":1' });
  assert.deepEqual(await P.loadDoc({ localStorage: truncated }), EMPTY);
  const mixed = fakeIdb(new Map([['history', { samples: [null, { t: 'x' }, { t: 5, c: {} }, { t: 6 }], rows: 7 }]]));
  const back = await P.loadDoc({ indexedDB: mixed });
  assert.deepEqual(back.samples.map((s) => s.t), [5]);
  assert.deepEqual(back.rows, []);
});

test('row 11: IndexedDB that fails to open falls back to localStorage', async () => {
  const ls = fakeLs();
  const env = { indexedDB: fakeIdb(new Map(), { failOpen: true }), localStorage: ls };
  const doc = { samples: [poll(T0)], rows: [] };
  assert.equal(await P.saveDoc(env, doc), 'localStorage');
  assert.deepEqual(await P.loadDoc(env), doc);
});

test('row 12: with no storage at all the history stays in memory and nothing throws', async () => {
  const env = {};
  Object.defineProperty(env, 'localStorage', { get() { throw new Error('SecurityError'); } });
  assert.deepEqual(await P.loadDoc(env), EMPTY);
  const doc = { samples: [poll(T0)], rows: [] };
  assert.equal(await P.saveDoc(env, doc), 'memory');
  const full = { indexedDB: { open() { throw new Error('blocked'); } }, localStorage: fakeLs({}, { throwSet: true }) };
  assert.equal(await P.saveDoc(full, doc), 'memory');
  assert.deepEqual(await P.loadDoc(full), EMPTY);
});

test('row 13: two tabs dedupe by timestamp and the store stays bounded', async () => {
  assert.equal(poll(5001).t, poll(5999).t, 'two tabs polling in the same second share a timestamp');
  const idb = fakeIdb();
  const envA = { indexedDB: fakeIdb(idb.shared) }, envB = { indexedDB: fakeIdb(idb.shared) };
  const step = 10_000, end = T0 + 30 * 3600_000;
  let a = [], b = [];
  for (let t = T0; t <= end; t += step) {
    a = P.appendSample(a, poll(t, { requests_success_total: (t - T0) / step }), t);
    if (t % 20_000 === 0) b = P.appendSample(b, poll(t, { requests_success_total: (t - T0) / step }), t);
  }
  const mergedA = await P.persistDoc(envA, { samples: a, rows: [] }, end);
  const mergedB = await P.persistDoc(envB, { samples: b, rows: [] }, end);
  const ts = mergedB.samples.map((s) => s.t);
  assert.equal(new Set(ts).size, ts.length, 'no duplicate timestamps');
  assert.deepEqual(ts, [...ts].sort((x, y) => x - y), 'ordered');
  assert.ok(ts.length <= 3600 + 24 * 60 + 1, `bounded, got ${ts.length}`);
  assert.ok(ts[0] >= end - 24 * 3600_000, 'nothing older than 24h');
  const old = ts.filter((t) => t < end - 3600_000);
  assert.ok(old.every((t, i) => i === 0 || t - old[i - 1] >= 60_000), 'older than an hour is one per minute');
  assert.deepEqual(mergedB.samples, mergedA.samples);
});

test('row 14: a request shorter than one poll is in the totals and not in the table', () => {
  const before = poll(T0, { requests_success_total: 4 });
  let rows = rowsOf([], [], T0);
  rows = rowsOf(rows, [], T0 + 1000);
  const after = poll(T0 + 1000, { requests_success_total: 5 });
  assert.deepEqual(rows, []);
  assert.equal(P.windowTotals([before, after], T0, T0 + 1000).requests_success_total, 1);
});

test('row 14: a request that spans polls opens on its request_id and closes when it leaves', () => {
  const s = (over = {}) => ({ model: 'm', phase: 'prefill', request_id: 9, client: 'claude-code', context_tokens: 10, context_length: 100, cached_tokens: 0, generated_tokens: 0, ...over });
  let rows = rowsOf([], [s()], T0);
  rows = rowsOf(rows, [s({ phase: 'decode', generated_tokens: 5 })], T0 + 1000);
  assert.equal(rows.length, 1, 'a phase change on the same request_id is the same row');
  assert.equal(rows[0].endT, null);
  assert.equal(rows[0].generated, 5);
  rows = rowsOf(rows, [], T0 + 2000);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].endT, T0 + 2000);
  assert.equal(rows[0].client, 'claude-code');
});

test('row 14: idle cached sessions are not requests', () => {
  assert.deepEqual(rowsOf([], [{ model: 'm', phase: 'cached', context_tokens: 9 }], T0), []);
});

test('failed, rejected and cancelled move as counter deltas', () => {
  const a = poll(T0, { requests_failed_total: 1, requests_rejected_total: 2, requests_cancelled_total: 3 });
  const b = poll(T0 + 2000, { requests_failed_total: 4, requests_rejected_total: 2, requests_cancelled_total: 8 });
  const tot = P.windowTotals([a, b], T0, T0 + 2000);
  assert.deepEqual([tot.requests_failed_total, tot.requests_rejected_total, tot.requests_cancelled_total], [3, 0, 5]);
});

test('per-model totals attribute a delta to the model that was active', () => {
  const act = (model) => [{ model, phase: 'decode', generated_tokens: 1 }];
  const s = [
    poll(T0, { generation_tokens_total: 0, requests_success_total: 0 }, {}, act('alpha')),
    poll(T0 + 1000, { generation_tokens_total: 100, requests_success_total: 1 }, {}, []),
    poll(T0 + 2000, { generation_tokens_total: 100, requests_success_total: 1 }, {}, act('beta')),
    poll(T0 + 3000, { generation_tokens_total: 130, requests_success_total: 2 }, {}, []),
  ];
  const m = P.modelTotals(s, T0, T0 + 3000);
  assert.equal(m.alpha.generation_tokens_total, 100);
  assert.equal(m.beta.generation_tokens_total, 30);
  assert.equal(m.alpha.requests_success_total, 1);
});

// The panel polls through the prefix the page was served under, not the proxy's
// root — and through the page's ONE implementation (a twin is the next
// divergence waiting to happen).
test('the metrics poll resolves against the page prefix', () => {
  assert.equal(apiPrefix, globalThis.apiPrefix);
  assert.equal(apiPrefix('/mount') + '/metrics.json', '/mount/metrics.json');
  assert.equal(apiPrefix('/') + '/metrics.json', '/metrics.json');
});

await Promise.all(pending);
console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
