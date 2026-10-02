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
import { runInNewContext } from 'node:vm';
import assert from 'node:assert/strict';

const here = dirname(fileURLToPath(import.meta.url));
const src = readFileSync(join(here, '..', 'src', 'html', 'metrics.js'), 'utf8');

// The file guards its IIFE on `typeof document`, so in node only the top-level
// helpers evaluate. It hands them back through `globalThis.__mlxPanel`.
new Function(src)();
const { computeRates } = globalThis.__mlxPanel ?? {};
assert.ok(computeRates, 'metrics.js must expose computeRates for tests');

const counters = (over = {}) => ({
  prompt_tokens_total: 10000,
  prefill_tokens_total: 4000,
  prefix_cache_tokens_total: 6000,
  generation_tokens_total: 500,
  requests_success_total: 4,
  requests_cancelled_total: 0,
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
function test(name, fn) {
  try { fn(); console.log(`  PASS ${name}`); }
  catch (e) { failures++; console.log(`  FAIL ${name}\n       ${e.message}`); }
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

const { monitorWindow, monitorHistory, monitorValidPair, monitorWindowCounters, monitorWindowActiveRate, monitorWindowGauge, monitorWindowMemory, monitorLifetime, monitorWindowTTFT, monitorIntervalMeans, monitorGaugeSeries, monitorSeries, monitorNearestPoint, monitorEscape, monitorFormatTTFT, monitorChartTimeLabel } = globalThis.__mlxPanel;
const sample = (at_ms, values) => ({ at_ms, ...values });
test('chart labels include local dates when the window crosses midnight', () => {
  const start = new Date(2026, 8, 28, 23, 30).getTime();
  const end = new Date(2026, 8, 29, 1, 30).getTime();
  const day = new Date(start).toLocaleString(undefined, { month:'short', day:'numeric' });
  const nextDay = new Date(end).toLocaleString(undefined, { month:'short', day:'numeric' });
  assert.ok(monitorChartTimeLabel(start,start,end).includes(day));
  assert.ok(monitorChartTimeLabel(end,start,end).includes(nextDay));
  assert.ok(monitorChartTimeLabel(start,start,end,true).includes(day));
  assert.equal(monitorChartTimeLabel(start,start,end,true).includes(new Date(start).getFullYear().toString()),false);
});
test('same-day chart labels keep time without a redundant date', () => {
  const start = new Date(2026, 8, 29, 10, 15).getTime();
  const end = start + 5 * 60_000;
  const day = new Date(start).toLocaleString(undefined, { month:'short', day:'numeric' });
  assert.equal(monitorChartTimeLabel(start,start,end).includes(day),false);
  assert.equal(monitorChartTimeLabel(start,start,end,true).includes(day),false);
  assert.match(monitorChartTimeLabel(start,start,end,true),/15/);
});
test('chart labels include years across New Year', () => {
  const start = new Date(2025, 11, 31, 23, 30).getTime();
  const end = new Date(2026, 0, 1, 1, 30).getTime();
  assert.ok(monitorChartTimeLabel(start,start,end).includes('2025'));
  assert.ok(monitorChartTimeLabel(end,start,end,true).includes('2026'));
});
test('TTFT card changes from ms to seconds at one second', () => {
  assert.equal(monitorFormatTTFT(null),'—');
  assert.equal(monitorFormatTTFT(NaN),'—');
  assert.equal(monitorFormatTTFT(Infinity),'—');
  assert.equal(monitorFormatTTFT(0),'0 ms');
  assert.match(monitorFormatTTFT(999.9),/^999[.,]9 ms$/);
  assert.equal(monitorFormatTTFT(1000),'1 s');
  assert.match(monitorFormatTTFT(101403.9),/^101[.,]4 s$/);
});
test('counter window includes idle sampled time', () => {
  const history = [sample(0, { generation_tokens_live: 0 }),
    sample(1000, { generation_tokens_live: 100 }),
    sample(4000, { generation_tokens_live: 100 })];
  const result = monitorWindowCounters(history, ['generation_tokens_live'], 0, 4000);
  assert.equal(result.deltas.generation_tokens_live, 100);
  assert.equal(result.seconds, 4);
  assert.equal(result.deltas.generation_tokens_live / result.seconds, 25);
  const recent = monitorWindowCounters(history, ['generation_tokens_live'], 1000, 4000);
  assert.equal(recent.deltas.generation_tokens_live / recent.seconds, 0);
});
test('processing average stays stable as idle samples append', () => {
  const history = [sample(0,{generation_tokens_live:0,decode_active_ns_total:0}),
    sample(2000,{generation_tokens_live:100,decode_active_ns_total:1_000_000_000}),
    sample(4000,{generation_tokens_live:100,decode_active_ns_total:1_000_000_000}),
    sample(6000,{generation_tokens_live:100,decode_active_ns_total:1_000_000_000})];
  const during=monitorWindowActiveRate(history.slice(0,2),'generation_tokens_live','decode_active_ns_total',0,2000);
  const after=monitorWindowActiveRate(history,'generation_tokens_live','decode_active_ns_total',0,6000);
  assert.equal(during.rate,100);
  assert.equal(after.rate,100);
  assert.equal(after.activeSeconds,1);
  assert.equal(after.coverageMs,6000);
  assert.equal(monitorWindowActiveRate(history,'generation_tokens_live','decode_active_ns_total',4000,6000).rate,null);
});
test('processing average weights unequal busy durations by active time', () => {
  const history = [sample(0,{generation_tokens_live:0,decode_active_ns_total:0}),
    sample(2000,{generation_tokens_live:100,decode_active_ns_total:1_000_000_000}),
    sample(4000,{generation_tokens_live:200,decode_active_ns_total:1_500_000_000})];
  const result=monitorWindowActiveRate(history,'generation_tokens_live','decode_active_ns_total',0,4000);
  assert.ok(Math.abs(result.rate-200/1.5)<1e-9);
  assert.equal(result.activeSeconds,1.5);
});
test('long active prefill with no published tokens lowers average', () => {
  const history = [sample(0,{prefill_tokens_forwarded_live_total:0,prefill_active_ns_total:0}),
    sample(2000,{prefill_tokens_forwarded_live_total:0,prefill_active_ns_total:1_000_000_000}),
    sample(4000,{prefill_tokens_forwarded_live_total:100,prefill_active_ns_total:2_000_000_000})];
  const first=monitorWindowActiveRate(history.slice(0,2),'prefill_tokens_forwarded_live_total','prefill_active_ns_total',0,2000);
  const full=monitorWindowActiveRate(history,'prefill_tokens_forwarded_live_total','prefill_active_ns_total',0,4000);
  assert.equal(first.rate,0);
  assert.equal(full.rate,50);
});
test('processing average excludes boundary, resets, missing samples and long gaps', () => {
  const history = [sample(0,{generation_tokens_live:0,decode_active_ns_total:0}),
    sample(2000,{generation_tokens_live:20,decode_active_ns_total:1_000_000_000}),
    sample(4000,{generation_tokens_live:5,decode_active_ns_total:100_000_000}),
    sample(6000,{generation_tokens_live:15,decode_active_ns_total:600_000_000}),
    sample(8000,{generation_tokens_live:20,decode_active_ns_total:null}),
    sample(10000,{generation_tokens_live:25,decode_active_ns_total:1_000_000_000}),
    sample(20000,{generation_tokens_live:125,decode_active_ns_total:6_000_000_000})];
  const result=monitorWindowActiveRate(history,'generation_tokens_live','decode_active_ns_total',2000,20000);
  assert.equal(result.rate,20);
  assert.equal(result.coverageMs,2000);
  assert.equal(monitorWindowActiveRate(history,'generation_tokens_live','decode_active_ns_total',0,0).rate,null);
});
test('nullable active time is missing data, not zero or a reset', () => {
  const history=[sample(0,{generation_tokens_live:0,decode_active_ns_total:0}),
    sample(2000,{generation_tokens_live:20,decode_active_ns_total:1_000_000_000}),
    sample(4000,{generation_tokens_live:30,decode_active_ns_total:null}),
    sample(6000,{generation_tokens_live:40,decode_active_ns_total:2_000_000_000})];
  const result=monitorWindowActiveRate(history,'generation_tokens_live','decode_active_ns_total',0,6000);
  assert.equal(result.rate,20);
  assert.equal(result.activeSeconds,1);
  assert.equal(result.coverageMs,2000);
  assert.equal(monitorWindowActiveRate(history,'generation_tokens_live','decode_active_ns_total',2000,6000).rate,null);
});
test('selected-window boundary uses only whole observed intervals', () => {
  const history = [sample(0, { prefill_tokens_forwarded_live_total: 0 }),
    sample(4000, { prefill_tokens_forwarded_live_total: 40 }),
    sample(6000, { prefill_tokens_forwarded_live_total: 50 })];
  const short = monitorWindowCounters(history, ['prefill_tokens_forwarded_live_total'], 2000, 6000);
  assert.equal(short.deltas.prefill_tokens_forwarded_live_total, 10);
  assert.equal(short.seconds, 2);
  const long = monitorWindowCounters(history, ['prefill_tokens_forwarded_live_total'], 0, 6000);
  assert.equal(long.deltas.prefill_tokens_forwarded_live_total / long.seconds, 50 / 6);
});
test('startup uses measured coverage, and empty or zero-duration history has no speed', () => {
  const h = [sample(8000, { generation_tokens_live: 10 }), sample(10000, { generation_tokens_live: 14 })];
  const partial = monitorWindowCounters(h, ['generation_tokens_live'], 0, 10000);
  assert.equal(partial.seconds, 2);
  assert.equal(partial.coverageMs, 2000);
  assert.equal(partial.deltas.generation_tokens_live / partial.seconds, 2);
  for (const invalid of [[], [h[0]], [h[0], sample(8000, { generation_tokens_live: 20 })]]) {
    const result = monitorWindowCounters(invalid, ['generation_tokens_live'], 0, 10000);
    assert.equal(result.seconds, 0);
  }
});
test('reset, missing counter and long sampling gap do not create a false rate', () => {
  const history = [sample(0, { generation_tokens_live: 20 }),
    sample(2000, { generation_tokens_live: 30 }),
    sample(4000, { generation_tokens_live: 2 }),
    sample(6000, { generation_tokens_live: 4 }),
    sample(8000, { generation_tokens_live: null }),
    sample(10000, { generation_tokens_live: 10 }),
    sample(18000, { generation_tokens_live: 100 }),
    sample(20000, { generation_tokens_live: 104 })];
  const result = monitorWindowCounters(history, ['generation_tokens_live'], 0, 20000);
  assert.equal(result.deltas.generation_tokens_live, 16);
  assert.equal(result.seconds, 6);
});
test('cache card uses hit and lookup deltas on the same selected intervals', () => {
  const history = [sample(0, { cache_hits_total: 900, cache_queries_total: 1000 }),
    sample(2000, { cache_hits_total: 901, cache_queries_total: 1010 }),
    sample(4000, { cache_hits_total: 901, cache_queries_total: 1010 }),
    sample(6000, { cache_hits_total: 906, cache_queries_total: 1020 })];
  const earlier = monitorWindowCounters(history, ['cache_hits_total', 'cache_queries_total'], 0, 4000);
  assert.equal(earlier.deltas.cache_hits_total, 1);
  assert.equal(earlier.deltas.cache_queries_total, 10);
  assert.equal(100 * earlier.deltas.cache_hits_total / earlier.deltas.cache_queries_total, 10);
  const recent = monitorWindowCounters(history, ['cache_hits_total', 'cache_queries_total'], 4000, 6000);
  assert.equal(recent.deltas.cache_hits_total, 5);
  assert.equal(recent.deltas.cache_queries_total, 10);
  assert.equal(100 * recent.deltas.cache_hits_total / recent.deltas.cache_queries_total, 50);
});
test('zero cache lookups remains distinct from a measured zero-percent hit rate', () => {
  const history = [sample(0, { cache_hits_total: 3, cache_queries_total: 10 }),
    sample(2000, { cache_hits_total: 3, cache_queries_total: 10 }),
    sample(4000, { cache_hits_total: 3, cache_queries_total: 15 })];
  const idle = monitorWindowCounters(history, ['cache_hits_total', 'cache_queries_total'], 0, 2000);
  assert.equal(idle.deltas.cache_queries_total, 0);
  const misses = monitorWindowCounters(history, ['cache_hits_total', 'cache_queries_total'], 2000, 4000);
  assert.equal(misses.deltas.cache_queries_total, 5);
  assert.equal(misses.deltas.cache_hits_total, 0);
  assert.equal(100 * misses.deltas.cache_hits_total / misses.deltas.cache_queries_total, 0);
});
test('cache pairs discard intervals with a missing hit counter', () => {
  const history = [sample(0, { cache_hits_total: 2, cache_queries_total: 10 }),
    sample(2000, { cache_hits_total: null, cache_queries_total: 20 }),
    sample(4000, { cache_hits_total: 3, cache_queries_total: 30 }),
    sample(6000, { cache_hits_total: 4, cache_queries_total: 40 })];
  const result = monitorWindowCounters(history, ['cache_hits_total', 'cache_queries_total'], 0, 6000);
  assert.equal(result.deltas.cache_hits_total, 1);
  assert.equal(result.deltas.cache_queries_total, 10);
  assert.equal(result.seconds, 2);
});
test('TTFT average weights by successful request count instead of intervals', () => {
  const history = [sample(0, { ttft_ns_sum: 0, ttft_count: 0 }),
    sample(1000, { ttft_ns_sum: 1_000_000_000, ttft_count: 10 }),
    sample(4000, { ttft_ns_sum: 2_000_000_000, ttft_count: 11 })];
  const total = monitorWindowTTFT(history, 0, 4000);
  assert.equal(total.count, 11);
  assert.equal(total.average, 2000 / 11);
  const recent = monitorWindowTTFT(history, 1000, 4000);
  assert.equal(recent.count, 1);
  assert.equal(recent.average, 1000);
  assert.deepEqual(monitorIntervalMeans(history, 'ttft_ns_sum', 'ttft_count', 1e6).map(p => p.value), [null, 100, 1000]);
});
test('TTFT has no mean without an observation, and invalid intervals leave graph gaps', () => {
  const history = [sample(0, { ttft_ns_sum: 100_000_000, ttft_count: 1 }),
    sample(2000, { ttft_ns_sum: 100_000_000, ttft_count: 1 }),
    sample(4000, { ttft_ns_sum: null, ttft_count: 2 }),
    sample(6000, { ttft_ns_sum: 200_000_000, ttft_count: 2 }),
    sample(14000, { ttft_ns_sum: 300_000_000, ttft_count: 3 }),
    sample(16000, { ttft_ns_sum: 50_000_000, ttft_count: 1 })];
  const total = monitorWindowTTFT(history, 0, 16000);
  assert.equal(total.count, 0);
  assert.equal(total.average, null);
  assert.deepEqual(monitorIntervalMeans(history, 'ttft_ns_sum', 'ttft_count', 1e6).map(p => p.value), [null, null, null, null, null, null]);
});
test('TTFT rejects a window containing incoherent sampling, then rebases graph', () => {
  const history = [sample(0, { ttft_ns_sum: 0, ttft_count: 0 }),
    sample(2000, { ttft_ns_sum: 100_000_000, ttft_count: 0 }),
    sample(4000, { ttft_ns_sum: 200_000_000, ttft_count: 1 }),
    sample(6000, { ttft_ns_sum: 500_000_000, ttft_count: 3 })];
  const result = monitorWindowTTFT(history, 0, 6000);
  assert.equal(result.count, 0);
  assert.equal(result.coverageMs, 0);
  assert.equal(result.average, null);
  assert.deepEqual(monitorIntervalMeans(history, 'ttft_ns_sum', 'ttft_count', 1e6).map(p => p.value), [null, null, null, 150]);
  assert.equal(monitorWindowTTFT(history, 2000, 4000).average, null);
});
test('TTFT atomic sampling skew never fabricates a zero-ms observation', () => {
  const history = [sample(0, { ttft_ns_sum: 0, ttft_count: 0 }),
    sample(2000, { ttft_ns_sum: 100_000_000, ttft_count: 0 }),
    sample(4000, { ttft_ns_sum: 100_000_000, ttft_count: 1 })];
  assert.equal(monitorWindowTTFT(history, 0, 4000).average, null);
  assert.deepEqual(monitorIntervalMeans(history, 'ttft_ns_sum', 'ttft_count', 1e6).map(p => p.value), [null, null, null]);
});
test('memory card weights gauge by elapsed time with trapezoids and retains idle periods', () => {
  const GiB = 1073741824;
  const history = [sample(0, { process_bytes: 2 * GiB }),
    sample(1000, { process_bytes: 2 * GiB }),
    sample(4000, { process_bytes: 4 * GiB })];
  const all = monitorWindowGauge(history, 'process_bytes', 0, 4000);
  assert.equal(all.seconds, 4);
  assert.equal(all.average / GiB, 2.75);
  const recent = monitorWindowGauge(history, 'process_bytes', 1000, 4000);
  assert.equal(recent.average / GiB, 3);
});
test('memory skips missing and long gaps without using lifetime or the latest gauge', () => {
  const GiB = 1073741824;
  const history = [sample(0, { process_bytes: 2 * GiB }),
    sample(2000, { process_bytes: 2 * GiB }),
    sample(4000, { process_bytes: null }),
    sample(6000, { process_bytes: 3 * GiB }),
    sample(14000, { process_bytes: 9 * GiB }),
    sample(16000, { process_bytes: 4 * GiB })];
  const result = monitorWindowGauge(history, 'process_bytes', 0, 16000);
  assert.equal(result.seconds, 4);
  assert.equal(result.coverageMs, 4000);
  assert.equal(result.average / GiB, 4.25);
  const graph = monitorGaugeSeries(history, 'process_bytes').map(p => p.value);
  assert.equal(graph[2], null);
  assert.equal(graph[4], null);
  assert.equal(monitorWindowGauge([], 'process_bytes', 0, 16000).average, null);
});
test('window percentiles exclude failures, null timing and other models', () => {
  const row = (model, finished_at_ms, ttft_ms, outcome='success') => ({model,finished_at_ms,ttft_ms,outcome});
  const d={monitor:{recent_requests:[row('a',10,1),row('a',100,10),row('a',110,null),row('a',120,500,'failed'),row('a',130,20),row('b',130,900)]}};
  const w=monitorWindow(d,150,100,'a');
  assert.equal(w.requests.length,4);
  assert.equal(w.percentile('ttft_ms',.5),10);
  assert.equal(w.percentile('ttft_ms',.95),20);
  assert.equal(w.percentile('queue_ms',.5),null);
});
test('history rates handle irregular timestamps, resets and missing measurements', () => {
  const h=[{at_ms:1000,n:4},{at_ms:3000,n:10},{at_ms:4000,n:1},{at_ms:5000,n:null}];
  assert.deepEqual(monitorSeries(h,'n',true).map(p=>p.value),[null,3,null,null]);
  assert.deepEqual(monitorSeries(h,'n').map(p=>p.value),[4,10,1,null]);
  assert.equal(monitorWindow({monitor:{history:[{at_ms:1000,n:100},{at_ms:2000,n:1}]}},2000,2000,'').rate('n'),null);
});
test('retention overflow marks partial request coverage only when window exceeds retained records', () => {
  const d={monitor:{retention:{requests_dropped:5},recent_requests:[{finished_at_ms:500}]}};
  assert.equal(monitorWindow(d,1000,1000,'').partial,true);
  assert.equal(monitorWindow(d,1000,100,'').partial,false);
});
test('model and error metadata are escaped before entering table markup', () => {
  assert.equal(monitorEscape('<img src=x onerror="x">&'), '&lt;img src=x onerror=&quot;x&quot;&gt;&amp;');
});

test('live prefill history advances before request completion and does not count completion twice', () => {
  const h=[{at_ms:1000,prefill_tokens_total:0,prefill_tokens_forwarded_live_total:0},
    {at_ms:3000,prefill_tokens_total:0,prefill_tokens_forwarded_live_total:8192},
    {at_ms:5000,prefill_tokens_total:0,prefill_tokens_forwarded_live_total:16384},
    {at_ms:7000,prefill_tokens_total:16384,prefill_tokens_forwarded_live_total:16384}];
  assert.deepEqual(monitorSeries(h,'prefill_tokens_forwarded_live_total',true).map(p=>p.value),[null,4096,4096,0]);
  assert.equal(monitorWindow({monitor:{history:h.slice(0,2)}},3000,30000,'').rate('prefill_tokens_forwarded_live_total'),4096);
});

test('point picker maps first, middle and last samples at SVG edges', () => {
  const series = [{ name: 'Decode', points: [
    { t: 1000, value: 0 }, { t: 6000, value: 50 }, { t: 11000, value: 100 },
  ] }];
  for (const [x, y, expected] of [[30, 108, 0], [310, 63, 1], [590, 18, 2]]) {
    const hit = monitorNearestPoint(series, 1000, 11000, x, y, 100);
    assert.equal(hit?.pointIndex, expected);
    assert.equal(hit?.point, series[0].points[expected]);
  }
});

test('point picker uses the supplied scale for each separate rate graph', () => {
  const prefill = [{ name: 'Prefill', points: [{ t: 6000, value: 3000 }] }];
  const decode = [{ name: 'Decode', points: [{ t: 6000, value: 30 }] }];
  assert.equal(monitorNearestPoint(prefill, 1000, 11000, 310, 18, 3000)?.point.value, 3000);
  assert.equal(monitorNearestPoint(decode, 1000, 11000, 310, 18, 30)?.point.value, 30);
});

test('point picker never invents a value for a gap, reset or empty period', () => {
  const history = [
    { at_ms: 1000, n: 10 }, { at_ms: 2000, n: 20 },
    { at_ms: 3000, n: 1 }, { at_ms: 4000, n: null },
  ];
  const points = monitorSeries(history, 'n', true);
  assert.deepEqual(points.map(p => p.value), [null, 10, null, null]);
  const series = [{ name: 'Rate', points }];
  assert.equal(monitorNearestPoint(series, 1000, 4000, 30, 18, 10), null);
  assert.equal(monitorNearestPoint(series, 1000, 4000, 590, 18, 10), null);
  assert.equal(monitorNearestPoint(series, 1000, 4000, 310, 18, 10), null);
  assert.equal(monitorNearestPoint([], 1000, 4000, 310, 18, 10), null);
});

test('range controls render defaults and change the monitor scope', () => {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {
      dataset: {}, innerHTML: '', textContent: '', value: id === 'm-window' ? '300000' : '',
      listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; },
      setAttribute() {},
      get selectedOptions() {
        const markup = elements.get('mlx-metrics').innerHTML;
        const options = /<select id="m-window"[^>]*>(.*?)<\/select>/.exec(markup)?.[1] || '';
        const choice = [...options.matchAll(/<option value="([^"]+)"[^>]*>([^<]+)<\/option>/g)]
          .find(([, value]) => value === this.value);
        return choice ? [{ textContent: choice[2] }] : [];
      },
    });
    return elements.get(id);
  };
  const document = {
    hidden: false, getElementById: element, addEventListener() {},
    createTreeWalker: () => ({ nextNode: () => false }),
  };
  runInNewContext(src, {
    document, NodeFilter: { SHOW_TEXT: 4 }, location: { search: '', hash: '#monitor' },
    window: { addEventListener() {}, mlxI18n: {
      t: (key, args = []) => key.replace(/%@/g, () => args.shift()), onChange() {},
    } }, fetch: () => new Promise(() => {}),
    setTimeout: () => 0, AbortSignal, URLSearchParams,
  });
  const markup = element('mlx-metrics').innerHTML;
  const options = /<select id="m-window"[^>]*>(.*?)<\/select>/.exec(markup)?.[1] || '';
  assert.deepEqual([...options.matchAll(/<option value="([^"]+)"/g)].map(m => m[1]),
    ['60000','300000','900000','1800000','3600000','10800000','21600000','43200000','86400000','startup']);
  assert.match(options, /<option value="300000" selected>5m<\/option>/);
  assert.match(options, /<option value="startup">Since startup<\/option>/);
  assert.match(element('m-overview-scope').textContent, /5m/);
  assert.match(element('m-coverage').textContent, /0 \/ 256 retained requests.*0 \/ 128 retained events/);
  const range = element('m-window');
  range.value = '86400000';
  range.listeners.change({ target: range });
  assert.match(element('m-overview-scope').textContent, /24h/);
  range.value = 'startup';
  range.listeners.change({ target: range });
  assert.match(element('m-overview-scope').textContent, /Since startup.*Start time unavailable/);
});

test('24h and startup use distinct actual boundaries with archived history', () => {
  const hour=3_600_000, now=30*hour, server={started_at_ms:0};
  const old={at_ms:2*hour,continuity_id:1,generation_tokens_live:20,decode_active_ns_total:1_000_000_000};
  const mid={at_ms:7*hour,continuity_id:1,generation_tokens_live:70,decode_active_ns_total:2_000_000_000};
  const raw={at_ms:now,continuity_id:1,generation_tokens_live:300,decode_active_ns_total:3_000_000_000};
  const monitor={server,history_archive:[old,mid],history:[raw],recent_requests:[{finished_at_ms:3*hour,outcome:'success'},{finished_at_ms:8*hour,outcome:'success'}]};
  assert.deepEqual(monitorWindow({monitor},now,24*hour,'').history,[mid,raw]);
  assert.equal(monitorWindow({monitor},now,24*hour,'').requests.length,1);
  assert.deepEqual(monitorWindow({monitor},now,'startup','').history,[old,mid,raw]);
  assert.equal(monitorWindow({monitor},now,'startup','').requests.length,2);
  assert.equal(monitorWindow({monitor:{...monitor,server:{}}},now,'startup','').history.length,0);
});

test('archive/raw overlap is removed and coarse valid intervals retain token progress', () => {
  const m={history_archive:[{at_ms:0,continuity_id:4,n:0},{at_ms:20000,continuity_id:4,n:100},{at_ms:40000,continuity_id:4,n:200}],history:[{at_ms:40000,continuity_id:4,n:200},{at_ms:42000,continuity_id:4,n:210}]};
  const history=monitorHistory(m);
  assert.deepEqual(history.map(s=>s.at_ms),[0,20000,40000,42000]);
  assert.equal(monitorWindowCounters(history,['n'],0,42000).deltas.n,210);
  assert.deepEqual(monitorSeries(history,'n',true).map(p=>p.value),[null,5,5,5]);
});

test('continuity excludes real raw gaps even when compacted endpoints are distant', () => {
  const h=[{at_ms:0,continuity_id:1,n:0},{at_ms:20000,continuity_id:1,n:100},{at_ms:40000,continuity_id:2,n:1000},{at_ms:60000,continuity_id:2,n:1100}];
  assert.equal(monitorValidPair(h[0],h[1],2000),true);
  assert.equal(monitorValidPair(h[1],h[2],2000),false);
  assert.equal(monitorWindowCounters(h,['n'],0,60000).deltas.n,200);
  assert.deepEqual(monitorSeries(h,'n',true).map(p=>p.value),[null,5,null,5]);
  assert.equal(monitorValidPair({at_ms:0},{at_ms:20000},2000),false);
});

test('startup cards use cumulative processing, TTFT, cache, and observed memory', () => {
  const lifetime=monitorLifetime({generation_tokens_live:900,decode_active_ns_total:3_000_000_000,prefill_tokens_forwarded_live_total:600,prefill_active_ns_total:2_000_000_000,cache_queries_total:10,cache_hits_total:4,ttft_ns_sum:2_000_000_000,ttft_count:4,process_memory_byte_seconds_total:6000,process_memory_observed_seconds_total:2});
  assert.equal(lifetime.decode,300);
  assert.equal(lifetime.prefill,300);
  assert.equal(lifetime.cache,40);
  assert.equal(lifetime.ttft,500);
  assert.equal(lifetime.memory,3000);
  assert.equal(lifetime.memorySeconds,2);
});

test('startup unavailable fields and idle-only phases stay unavailable', () => {
  const idle=monitorLifetime({generation_tokens_live:0,decode_active_ns_total:0,prefill_tokens_forwarded_live_total:0,prefill_active_ns_total:0,cache_queries_total:0,cache_hits_total:0,ttft_ns_sum:0,ttft_count:0,process_memory_byte_seconds_total:0,process_memory_observed_seconds_total:0});
  assert.equal(idle.decode,null);
  assert.equal(idle.prefill,null);
  assert.equal(idle.cache,null);
  assert.equal(idle.ttft,null);
  assert.equal(idle.memory,null);
  assert.equal(idle.ttftCount,0);
  assert.equal(monitorLifetime({decode_active_ns_total:null,generation_tokens_live:10,cache_queries_total:2,cache_hits_total:null}).decode,null);
  assert.equal(monitorLifetime({decode_active_ns_total:null,generation_tokens_live:10,cache_queries_total:2,cache_hits_total:null}).cache,null);
});

test('compacted memory uses observed area and time, excluding missing intervals', () => {
  const h=[{at_ms:0,continuity_id:1,process_memory_byte_seconds_total:0,process_memory_observed_seconds_total:0},{at_ms:20000,continuity_id:1,process_memory_byte_seconds_total:4000,process_memory_observed_seconds_total:2},{at_ms:40000,continuity_id:1,process_memory_byte_seconds_total:12000,process_memory_observed_seconds_total:4}];
  const result=monitorWindowMemory(h,0,40000);
  assert.equal(result.average,3000);
  assert.equal(result.seconds,4);
  assert.equal(monitorWindowMemory(h,10000,40000).average,4000);
});

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
