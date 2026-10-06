// Monitoring calculations from the shipped bundle; no DOM, GPU or dependencies.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import { test } from 'node:test';
const context = vm.createContext({ EventTarget, URL, atob });
for (const file of ['api.js', 'app.js'])
  vm.runInContext(readFileSync(new URL('../src/html/' + file, import.meta.url), 'utf8'), context);
const P = context.__mlxConsole;
assert.ok(P, 'Monitoring uses the console test hook');
const plain = (v) => JSON.parse(JSON.stringify(v));
const feed = (n, gauges = {}) => ({
  counters: { generation_tokens_total: n, requests_success_total: n, prefill_tokens_total: n },
  gauges: {
    process_start_time_seconds: 1,
    generation_tokens_live: n,
    prefill_tokens_live: 0,
    requests_running: 1,
    ...gauges
  },
  histograms: { prefill_time_seconds: { sum: 2, count: 1 } },
  sessions: []
});
const sample = (t, n, g = {}) => P.makeSample(t, feed(n, g));

test('live rates use elapsed seconds, active counters and completed prefill time', () => {
  const current = feed(100, { prefill_tokens_live: 300 });
  assert.deepEqual(plain(P.liveRates([{ t: 0, live: 20, pre: 100, req: 60 }], current, 4000)), {
    decode: 20,
    prefill: 50,
    requests: 10,
    averagePrefill: 50,
    live: 100,
    pre: 300
  });
});

test('a page refresh and a live tick agree on idle and prefill completion', () => {
  const old = [{ t: 0, live: 20, pre: 100, req: 1 }];
  P.liveRates(old, feed(100, { prefill_tokens_live: 300 }), 4000);
  const idle = feed(100, { requests_running: 0, prefill_tokens_live: 0 });
  for (const samples of [old, []]) {
    const rates = P.liveRates(samples, idle, 5000);
    assert.equal(rates.decode, 0);
    assert.equal(rates.prefill, 0);
    assert.equal(rates.averagePrefill, 50);
  }
  // A fresh page has no delta baseline for an active decode; unknown is not a remembered rate.
  assert.equal(P.liveRates([], feed(100), 5000).decode, null);
  assert.equal(P.liveRates(old, feed(100), 5000).prefill, 0);
  const history = [sample(0, 0), sample(1000, 10), sample(4000, 40)];
  assert.deepEqual(
    plain(P.rateSeries(JSON.parse(JSON.stringify(history)), 'generation_tokens_total', 0, 4000, 4)),
    plain(P.rateSeries(history, 'generation_tokens_total', 0, 4000, 4))
  );
});

test('zero work yields zero; missing samples, zero elapsed and counter reset stay unknown', () => {
  const before = [{ t: 1000, live: 20, pre: 10, req: 5 }];
  assert.equal(P.liveRates(before, feed(20), 2000).decode, 0);
  assert.equal(P.liveRates(before, feed(10), 2000).decode, null);
  assert.equal(P.liveRates(before, feed(30), 1000).decode, null);
  assert.equal(P.liveRates(before, feed(30), 302000).decode, null);
  assert.equal(P.liveRates([], feed(30), 2000).decode, null);
});

test('history buckets never interpolate through gaps, restarts or decreases', () => {
  const before = sample(0, 10);
  for (const after of [
    sample(1000, 20, { process_start_time_seconds: 2 }),
    sample(1000, 1),
    sample(301000, 20)
  ]) {
    assert.deepEqual(
      plain(P.rateSeries([before, after], 'generation_tokens_total', 0, after.t, 1)),
      [null]
    );
    assert.equal(P.windowTotals([before, after], 0, after.t).generation_tokens_total, 0);
  }
  assert.equal(
    P.rateSeries([before, sample(300000, 20)], 'generation_tokens_total', 0, 300000, 1)[0],
    10 / 300
  );
  const missing = sample(1000, 20);
  missing.c.generation_tokens_total = null;
  assert.deepEqual(plain(P.rateSeries([before, missing], 'generation_tokens_total', 0, 1000, 1)), [
    null
  ]);
});

test('history rates use counter deltas and measured time, not poll counts', () => {
  const samples = [sample(0, 0), sample(1000, 10), sample(4000, 40)];
  assert.deepEqual(plain(P.rateSeries(samples, 'generation_tokens_total', 0, 4000, 4)), [
    10,
    null,
    null,
    10
  ]);
  assert.equal(P.windowTotals(samples, 1000, 4000).generation_tokens_total, 30);
});

test('merged history deduplicates tabs, retains 24 hours and preserves gap/reset edges', () => {
  const now = 86400000;
  const samples = [
    sample(-1000, 0),
    sample(1000, 1),
    sample(2000, 2),
    sample(3000, 0),
    sample(4000, 1),
    sample(5000, 2),
    sample(600000, 3),
    sample(now - 1000, 4),
    sample(now, 5)
  ];
  const merged = P.mergeDocs({ samples, rows: [] }, { samples: [sample(now, 5)], rows: [] }, now);
  assert.deepEqual(plain(merged.samples.map((s) => s.t)), [
    1000,
    2000,
    3000,
    5000,
    600000,
    now - 1000,
    now
  ]);
});

test('Sessions track observed request ids, close departures, and exclude cached slots', () => {
  const s = {
    model: 'm',
    phase: 'prefill',
    request_id: 9,
    client: 'codex',
    context_tokens: 10,
    context_length: 100,
    cached_tokens: 0,
    generated_tokens: 0
  };
  let rows = P.trackRequests([], [s, { ...s, request_id: 0, phase: 'cached' }], 1000);
  rows = P.trackRequests(rows, [{ ...s, phase: 'decode', generated_tokens: 5 }], 2000);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].generated, 5);
  assert.equal(rows[0].endT, null);
  rows = P.trackRequests(rows, [], 3000);
  assert.equal(rows[0].endT, 3000);
  assert.equal(P.rangeRows(rows, 1500, 2500).length, 1);
  assert.equal(P.rangeRows(rows, 3001, 4000).length, 0);
});

test('per-model totals attribute completed deltas to the departing model', () => {
  const a = sample(0, 0),
    b = sample(1000, 10),
    c = sample(2000, 20);
  a.m = 'alpha';
  c.m = 'beta';
  assert.equal(P.modelTotals([a, b, c], 0, 2000).alpha.generation_tokens_total, 10);
  assert.equal(P.modelTotals([a, b, c], 0, 2000).beta.generation_tokens_total, 10);
});

test('console and Monitoring share mount-prefix resolution, without query keys', () => {
  for (const [path, prefix] of [
    ['/', ''],
    ['/mount', '/mount'],
    ['/mount/index.html', '/mount']
  ]) {
    assert.equal(P.apiPrefix(path), prefix);
    assert.equal(
      P.pageServer(new URL('https://host' + path + '?api_key=ignored')),
      'https://host' + prefix
    );
  }
});
