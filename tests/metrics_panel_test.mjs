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
const src = readFileSync(join(here, '..', 'src', 'html', 'metrics.js'), 'utf8');

// The file guards its IIFE on `typeof document`, so in node only the top-level
// helpers evaluate. It hands them back through `globalThis.__mlxPanel`.
new Function(src)();
const { computeRates, prefillPhaseLabel } = globalThis.__mlxPanel ?? {};
assert.ok(computeRates, 'metrics.js must expose computeRates for tests');
assert.ok(prefillPhaseLabel, 'metrics.js must expose prefillPhaseLabel for tests');

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
  prefill_target_tokens: 0,
  prefill_prompt_tokens: 0,
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


// ── Prefill progress: a numerator needs a denominator on the same base ───────
//
// Live numbers from tests/test_metrics.sh Phase 6 (Qwen3.5-4B-4bit, M4 Max):
// cold prompt=target=16512; warm prompt=22518 target=6037 reused=16481.

test('prefill progress renders live/target with the reuse, on real Phase 6 numbers', () => {
  const label = prefillPhaseLabel(gauges({
    prefill_tokens_live: 2048, prefill_target_tokens: 6037, prefill_prompt_tokens: 22518,
  }));
  assert.equal(label, 'prefilling · 2.0K / 6.0K (34%) · 16.5K reused');
});

test('a cold prefill shows no reuse clause (target == prompt)', () => {
  const label = prefillPhaseLabel(gauges({
    prefill_tokens_live: 8192, prefill_target_tokens: 16512, prefill_prompt_tokens: 16512,
  }));
  assert.equal(label, 'prefilling · 8.2K / 16.5K (50%)');
});

// This is the panel-side statement of what Phase 6 pins on the server side: a
// target published ABOVE the hot-cache trim equals the prompt, so the bar jumps
// to the reuse fraction on the first chunk and the reuse vanishes. Same live
// count, same prompt — only the target moves.
test('a pre-trim target would hide the reuse and inflate the bar', () => {
  const g = { prefill_tokens_live: 2048, prefill_prompt_tokens: 22518 };
  const correct = prefillPhaseLabel(gauges({ ...g, prefill_target_tokens: 6037 }));
  const broken  = prefillPhaseLabel(gauges({ ...g, prefill_target_tokens: 22518 }));
  assert.match(correct, /16\.5K reused/);
  assert.doesNotMatch(broken, /reused/);
  assert.match(correct, /\(34%\)/);
  assert.match(broken, /\(9%\)/);
});

// target == 0 is "the size is not known yet", NOT "0% done". It happens for the
// whole of the hot-cache restore (seconds on a long prompt) and for every
// ds4/llama/diffusion prefill, which never move any of the three gauges.
test('no denominator yet renders the phase word, never 0%', () => {
  assert.equal(prefillPhaseLabel(gauges({ prefill_prompt_tokens: 22518 })), 'prefilling');
  assert.equal(prefillPhaseLabel(gauges()), 'prefilling');
});

// An older server (or a ds4 path that moved only the live counter) must keep the
// pre-pair behaviour rather than dividing by zero.
test('a feed without the pair falls back to the bare token count', () => {
  const g = gauges({ prefill_tokens_live: 4096 });
  delete g.prefill_target_tokens;
  delete g.prefill_prompt_tokens;
  assert.equal(prefillPhaseLabel(g), 'prefilling · 4.1K tok');
});

// The chunk loop stops one token (plus the SSM snapshot backoff) short of the
// target, and a sampler tick can pair a new request's live count with the
// previous target. Neither may render as more than 100%.
test('the percentage is clamped at 100', () => {
  const label = prefillPhaseLabel(gauges({
    prefill_tokens_live: 30000, prefill_target_tokens: 6037, prefill_prompt_tokens: 22518,
  }));
  assert.match(label, /\(100%\)/);
});

// Same discipline as every other tile: derived from the current feed, nothing
// carried between ticks. Once the prefill ends all three read 0 together.
test('the pair returns to rest together, so the label carries nothing forward', () => {
  assert.equal(prefillPhaseLabel(gauges()), 'prefilling');
});

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
