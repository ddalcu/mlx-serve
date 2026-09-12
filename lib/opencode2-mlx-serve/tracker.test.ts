import assert from "node:assert/strict"
import { test } from "node:test"
import {
  SpeedTracker,
  parseMetricsJson,
  resolveMetricsUrl,
} from "./tracker.ts"
import { footerLabel } from "./rows.ts"
import { progressBar } from "./stats.ts"

test("resolveMetricsUrl accepts base, /metrics, and /metrics.json", () => {
  assert.equal(resolveMetricsUrl("http://127.0.0.1:11234"), "http://127.0.0.1:11234/metrics.json")
  assert.equal(resolveMetricsUrl("http://127.0.0.1:11234/"), "http://127.0.0.1:11234/metrics.json")
  assert.equal(resolveMetricsUrl("http://127.0.0.1:11234/metrics"), "http://127.0.0.1:11234/metrics.json")
  assert.equal(
    resolveMetricsUrl("http://127.0.0.1:11234/metrics.json"),
    "http://127.0.0.1:11234/metrics.json",
  )
})

test("parseMetricsJson reads live gauges", () => {
  const sample = parseMetricsJson(
    {
      gauges: {
        prefill_tokens_live: 8192,
        prefill_tokens_expected: 48000,
        generation_tokens_live: 100,
        requests_prefilling: 1,
        requests_running: 1,
      },
    },
    1_000,
  )
  assert.equal(sample.prefillLive, 8192)
  assert.equal(sample.prefillExpected, 48000, "the bar's real target rides the same feed")
  assert.equal(sample.prefilling, true)
  assert.equal(sample.running, true)
  assert.equal(parseMetricsJson({ gauges: {} }, 1_000).prefillExpected, 0, "old servers omit it: zero, never NaN")
})

test("a fully-cached step never draws a 0/xx bar", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m0", 0)
  t.finishStep("s1", "m0", { input: 20_000, output: 100 }, 3_000)
  t.beginStep("s1", "m1", 4_000)
  // The whole prompt restored from cache, no tokens streamed, no samples.
  t.finishStep("s1", "m1", { input: 5_000, output: 0, cacheRead: 5_000 }, 4_200)
  const v = t.value("s1", 4_200)
  assert.equal(v?.prefillTokens, null, "forwarded=0 is not a live reading")
  assert.equal(v?.phase, "generate", "a settled step retires the prefill phase")
  assert.equal(footerLabel(v, { barCells: 18 }), "100 tok · decode 0.0 t/s · prefill 0.0 t/s", "quiet zeros, not a 0/xx bar")
})

test("a settled step retires a live prefill phase", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  t.applyMetrics("s1", { t: 500, prefillLive: 20_000, genLive: 0, prefilling: true, running: true })
  assert.equal(t.value("s1", 500)?.phase, "prefill")
  t.finishStep("s1", "m1", { input: 48_000, output: 50 }, 800)
  const v = t.value("s1", 800)
  assert.equal(v?.phase, "generate", "step.ended means the call — and its prefill — is over")
  assert.equal(v?.prefillTokens, 48_000, "the settled count still reads as the last prefill")
})

test("another client's completions do not end our prefill", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  t.applyMetrics("s1", { t: 1_000, prefillLive: 8_000, genLive: 5_000, prefilling: true, running: true, runningCount: 2 })
  t.applyMetrics("s1", { t: 2_000, prefillLive: 16_000, genLive: 5_200, prefilling: true, running: true, runningCount: 2 })
  assert.equal(t.value("s1", 2_000)?.phase, "prefill", "their 200 tokens are not our first token")
  t.pushDelta("s1", "hello", 2_100, "m1")
  assert.equal(t.value("s1", 2_100)?.phase, "generate", "our own bytes still flip it")
})

test("step.streamed is the first-token mark, before any delta", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  t.markStreamed("s1", "m1", 1_500)
  const v = t.value("s1", 1_500)
  assert.equal(v?.phase, "generate")
  assert.equal(v?.ttftMs, 1_500, "the provider's wait is over at the first streamed content")
  assert.equal(footerLabel(v, { barCells: 10 }), "~0 tok · decode ~0.0 t/s · ttft 1.50s")
  // A late marker for another message must not touch the running step.
  t.markStreamed("s1", "m0", 2_000)
  assert.equal(t.value("s1", 2_000)?.ttftMs, 1_500)
})

test("a stream-only step measures decode from its own bytes", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  // A metered step at ~300 t/s. Its last rate must not leak into the next step.
  t.applyMetrics("s1", { t: 0, prefillLive: 0, prefillExpected: 0, genLive: 1_000, prefilling: false, running: true, runningCount: 1 })
  t.applyMetrics("s1", { t: 1_000, prefillLive: 0, prefillExpected: 0, genLive: 1_300, prefilling: false, running: true, runningCount: 1 })
  t.applyMetrics("s1", { t: 2_000, prefillLive: 0, prefillExpected: 0, genLive: 1_600, prefilling: false, running: true, runningCount: 1 })
  assert.equal(t.value("s1", 2_000)?.genTps, 300)
  // The next step is not mlx-serve: no metrics sample applies to it.
  t.beginStep("s1", "m2", 3_000)
  for (let i = 1; i <= 10; i++) t.pushDelta("s1", "abcdefghij", 3_000 + i * 100, "m2")
  const v = t.value("s1", 4_000)
  assert.equal(v?.phase, "generate", "our own bytes flip the phase without a feed")
  assert.ok(v?.genTps !== null && v.genTps < 100, `byte rate, not the stale 300 t/s: ${v?.genTps}`)
  assert.equal(v?.tokensEstimated, true, "byte-derived counts stay marked as estimates")
})

test("a stream-only step settles prefill from TTFT and usage", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  t.markStreamed("s1", "m1", 2_000)
  t.finishStep("s1", "m1", { input: 10_000, output: 50 }, 8_000)
  const v = t.value("s1", 8_000)
  assert.equal(v?.prefillTokens, 10_000)
  assert.equal(v?.prefillTps, 5_000, "the API's forwarded count over its own wait")
  assert.equal(footerLabel(v, { barCells: 10 }), "50 tok · decode 8.3 t/s · prefill 5000 t/s", "the frozen line carries the step's own average decode")
})

test("prefill phase reports live tokens and rate from metrics chunks", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  t.applyMetrics("s1", {
    t: 0,
    prefillLive: 0,
    genLive: 10_000,
    prefilling: true,
    running: true,
  })
  t.applyMetrics("s1", {
    t: 2_000,
    prefillLive: 8_192,
    genLive: 10_000,
    prefilling: true,
    running: true,
  })
  const v = t.value("s1", 2_000)
  assert.equal(v?.phase, "prefill")
  assert.equal(v?.prefillTokens, 8_192)
  assert.ok(v?.prefillTps !== null && Math.abs(v.prefillTps - 8_192 / 2) < 1)
  assert.match(footerLabel(v!) ?? "", /prefill/)
  assert.match(footerLabel(v!) ?? "", /t\/s/)
})

test("metrics gen bursts become a live generate rate", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  const idle = { prefillLive: 0, prefilling: false, running: true }
  t.applyMetrics("s1", { t: 1_000, genLive: 30_000, ...idle })
  t.applyMetrics("s1", { t: 2_500, genLive: 30_106, ...idle })
  t.applyMetrics("s1", { t: 4_000, genLive: 30_212, ...idle })
  const v = t.value("s1", 4_000)
  assert.equal(v?.phase, "generate")
  assert.equal(v?.genTokens, 212)
  assert.ok(v?.genTps !== null && Math.abs(v.genTps - 106 / 1.5) < 1)
})

test("stream deltas estimate gen speed without metrics", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  t.pushDelta("s1", "abcd", 200, "m1") // 4 bytes → ~1 tok
  t.pushDelta("s1", "abcdefghijkl", 700, "m1") // 12 bytes
  const v = t.value("s1", 700)
  assert.equal(v?.phase, "generate")
  assert.ok((v?.genTokens ?? 0) > 0)
  assert.ok(v?.genTps !== null)
  assert.equal(v?.tokensEstimated, true)
})

test("finishStep keeps live prefill tps and applies exact gen tokens", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  t.applyMetrics("s1", { t: 0, prefillLive: 0, genLive: 0, prefilling: true, running: true })
  t.applyMetrics("s1", { t: 2_000, prefillLive: 8_000, genLive: 0, prefilling: true, running: true })
  t.pushDelta("s1", "hello world", 2_100, "m1")
  t.finishStep("s1", "m1", { input: 12_000, cacheRead: 4_000, output: 50, reasoning: 10 }, 3_000)
  t.finish("s1", 3_000)
  const v = t.value("s1", 3_000)
  assert.equal(v?.phase, "frozen")
  assert.equal(v?.genTokens, 60)
  assert.equal(v?.tokensEstimated, false)
  assert.ok(v?.prefillTps !== null && Math.abs(v.prefillTps - 4_000) < 1)
  assert.equal(v?.prefillTokens, 8_000)
})

test("text deltas do not inflate gen t/s above mlx-serve gen_live", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  const idle = { prefillLive: 0, prefilling: false, running: true }
  t.applyMetrics("s1", { t: 0, genLive: 10_000, ...idle })
  for (let i = 1; i <= 20; i++) t.pushDelta("s1", "abcdefghij", i * 10, "m1")
  t.applyMetrics("s1", { t: 1_500, genLive: 10_106, ...idle })
  t.applyMetrics("s1", { t: 3_000, genLive: 10_212, ...idle })
  const v = t.value("s1", 3_000)
  assert.ok(v?.genTps !== null)
  assert.ok(v!.genTps! < 90, `got ${v?.genTps}`)
  assert.ok(Math.abs(v!.genTps! - 106 / 1.5) < 5)
})

test("a 250ms gen_live jump does not report 300 t/s", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  const idle = { prefillLive: 0, prefilling: false, running: true }
  t.applyMetrics("s1", { t: 0, genLive: 10_000, ...idle })
  t.applyMetrics("s1", { t: 250, genLive: 10_106, ...idle })
  const early = t.value("s1", 250)
  assert.equal(early?.genTps ?? null, null)
  t.applyMetrics("s1", { t: 1_750, genLive: 10_212, ...idle })
  const v = t.value("s1", 1_750)
  assert.ok(v?.genTps !== null && v.genTps < 90, `got ${v?.genTps}`)
})

test("a single late burst does not report 30 t/s", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  const idle = { prefillLive: 0, prefilling: false, running: true }
  t.applyMetrics("s1", { t: 0, genLive: 10_000, ...idle })
  t.applyMetrics("s1", { t: 3_500, genLive: 10_106, ...idle })
  const early = t.value("s1", 3_500)
  assert.equal(early?.genTps ?? null, null)
  t.applyMetrics("s1", { t: 5_000, genLive: 10_212, ...idle })
  const v = t.value("s1", 5_000)
  assert.ok(v?.genTps !== null && Math.abs(v.genTps - 106 / 1.5) < 5, `got ${v?.genTps}`)
})

test("tool idle between gen bursts is excluded from average gen t/s", () => {
  const t = new SpeedTracker()
  t.beginRun("s1", 0)
  t.beginStep("s1", "m1", 0)
  const idle = {
    prefillLive: 0,
    prefilling: false,
    running: true,
  }
  t.applyMetrics("s1", { t: 0, genLive: 1_000, ...idle })
  t.applyMetrics("s1", { t: 1_000, genLive: 1_100, ...idle })
  t.applyMetrics("s1", { t: 2_000, genLive: 1_200, ...idle })
  const during = t.value("s1", 2_000)
  assert.ok(during?.genTps !== null && Math.abs(during.genTps - 100) < 1)

  t.finishStep("s1", "m1", { output: 100, reasoning: 0 }, 1_100)
  t.applyMetrics("s1", { t: 12_000, genLive: 1_200, ...idle })
  const waiting = t.value("s1", 12_000)
  assert.ok(waiting?.genTps !== null && Math.abs(waiting.genTps - 100) < 1, "tool wait must not dilute the rate")

  t.beginStep("s1", "m2", 12_000)
  t.applyMetrics("s1", { t: 12_000, genLive: 1_200, ...idle })
  t.applyMetrics("s1", { t: 13_000, genLive: 1_300, ...idle })
  t.applyMetrics("s1", { t: 14_000, genLive: 1_400, ...idle })
  const after = t.value("s1", 14_000)
  assert.ok(after?.genTps !== null && Math.abs(after.genTps - 100) < 1)
})

test("the footer line is turn stats, and holds no memory figure", () => {
  assert.equal(
    footerLabel({
      phase: "prefill",
      prefillTokens: null,
      prefillExpected: null,
      prefillTps: null,
      genTokens: 0,
      genTps: null,
      ttftMs: null,
      elapsedMs: 4_200,
      tokensEstimated: true,
    }),
    "prefill waiting · 4.2s",
  )

  assert.equal(
    footerLabel({
      phase: "generate",
      prefillTokens: 8_192,
      prefillExpected: null,
      prefillTps: 410.4,
      genTokens: 312,
      genTps: 51.4,
      ttftMs: 1_200,
      elapsedMs: 8_000,
      tokensEstimated: true,
    }),
    "~312 tok · decode ~51.4 t/s · prefill 410 t/s",
  )

})

test("a second client decoding alongside us does not inflate our rate", () => {
  const tracker = new SpeedTracker()
  // The server-wide counter says 100 tok/s across two clients; our own stream
  // is ~10 tok/s of bytes. The meter must report our turn, not the machine.
  const shared = (t: number, runningCount: number) => ({
    t,
    prefillLive: 0,
    genLive: (t / 1000) * 100,
    prefilling: false,
    running: true,
    runningCount,
  })
  for (let i = 1; i <= 10; i++) {
    tracker.pushDelta("a", "x".repeat(48), i * 1000, "m1")
    tracker.applyMetrics("a", shared(i * 1000, 2))
  }
  const v = tracker.value("a", 10_000)
  assert.ok(v)
  assert.ok(v.genTps !== null && v.genTps < 30, `expected our own ~10 t/s, got ${v.genTps}`)
  assert.equal(v.tokensEstimated, true, "a byte-estimated rate says so")
})

test("alone, the turn still uses the server's own token counter", () => {
  const tracker = new SpeedTracker()
  tracker.beginRun("a", 0)
  tracker.beginStep("a", "m1", 0)
  for (let i = 1; i <= 6; i++) {
    tracker.pushDelta("a", "x".repeat(48), i * 1000, "m1")
    tracker.applyMetrics("a", {
      t: i * 1000,
      prefillLive: 0,
      genLive: i * 100,
      prefilling: false,
      running: true,
      runningCount: 1,
    })
  }
  const v = tracker.value("a", 6_000)
  assert.ok(v)
  assert.equal(v.genTps, 100, "one client: the counter is ours, and it beats a byte estimate")
})

test("the prefill target is the server gauge, never settled usage", () => {
  const tracker = new SpeedTracker()
  tracker.beginRun("a", 0)
  tracker.beginStep("a", "m1", 0)
  tracker.pushDelta("a", "hello", 200, "m1")
  assert.equal(tracker.value("a", 200)?.prefillExpected, null, "no gauge reading yet, no target")
  // Settled usage does NOT become a target: only the gauge does.
  tracker.finishStep("a", "m1", { input: 50_000, output: 100, cacheRead: 48_000 }, 1_000)
  tracker.beginStep("a", "m2", 2_000)
  assert.equal(tracker.value("a", 2_100)?.prefillExpected, null, "a settled 2k is not a denominator")
  tracker.applyMetrics("a", { t: 2_200, prefillLive: 500, prefillExpected: 48_000, genLive: 100, prefilling: true, running: true, runningCount: 1 })
  assert.equal(tracker.value("a", 2_200)?.prefillExpected, 48_000, "the gauge reading is the target")
})

test("a new step starts with no target until the gauge speaks", () => {
  const tracker = new SpeedTracker()
  tracker.beginRun("a", 0)
  tracker.beginStep("a", "m1", 0)
  tracker.applyMetrics("a", { t: 500, prefillLive: 1_000, prefillExpected: 9_000, genLive: 0, prefilling: true, running: true, runningCount: 1 })
  assert.equal(tracker.value("a", 500)?.prefillExpected, 9_000)
  tracker.finishStep("a", "m1", { input: 9_000, output: 10, cacheRead: 4_000 }, 1_000)
  tracker.beginStep("a", "m2", 2_000)
  assert.equal(tracker.value("a", 2_100)?.prefillExpected, null, "the old target does not leak into the new prefill")
  tracker.evict("a")
  tracker.beginRun("a", 3_000)
  tracker.beginStep("a", "m3", 3_000)
  assert.equal(tracker.value("a", 3_100)?.prefillExpected, null, "a deleted session starts over")
})

test("the footer prefill bar fills against the server's expected total", () => {
  const t = new SpeedTracker()
  t.beginRun("a", 0)
  t.beginStep("a", "m1", 0)
  for (let chunk = 1024; chunk <= 12_288; chunk += 1024) {
    t.applyMetrics("a", {
      t: chunk / 8,
      prefillLive: chunk,
      prefillExpected: 48_000,
      genLive: 0,
      prefilling: true,
      running: true,
      runningCount: 1,
    })
  }
  const line = footerLabel(t.value("a", 1_600), { barCells: 18 })
  assert.ok(line !== null)
  assert.match(line, /^prefill [█░]{18} 12,288\/48,000 · /, "bar, then live over the real total")
  assert.equal(line.includes("Memory"), false, "no memory figure in a turn readout")
  assert.equal([...line].filter((c) => c === "█").length, 5, "12,288 of 48,000 fills five cells")
})

test("no expected gauge, no bar: the prefill stays a rate line", () => {
  const t = new SpeedTracker()
  t.beginRun("a", 0)
  t.beginStep("a", "m1", 0)
  t.applyMetrics("a", { t: 1_200, prefillLive: 8_192, genLive: 0, prefilling: true, running: true, runningCount: 1 })
  const line = footerLabel(t.value("a", 1_200), { barCells: 18 })
  assert.equal(line, "prefill 8,192 tok · 6827 t/s")
  assert.equal(progressBar(0.5, 6), "███░░░", "the bar primitive the footer draws with")
  assert.equal(footerLabel(t.value("a", 1_200), { barCells: 0 }), line, "barCells 0 keeps the rate shape")
})

test("with two clients the token count comes from our bytes too, not the machine's", () => {
  const tracker = new SpeedTracker()
  tracker.beginRun("a", 0)
  tracker.beginStep("a", "m1", 0)
  // The server counts 100 tok/s across both clients; we streamed 480 bytes.
  for (let i = 1; i <= 10; i++) {
    tracker.pushDelta("a", "x".repeat(48), i * 1000, "m1")
    tracker.applyMetrics("a", {
      t: i * 1000,
      prefillLive: 0,
      genLive: i * 100,
      prefilling: false,
      running: true,
      runningCount: 2,
    })
  }
  const v = tracker.value("a", 10_000)
  assert.ok(v)
  assert.ok(v.genTps !== null && v.genTps < 30, `expected our own rate, got ${v.genTps}`)
  assert.ok(v.genTokens < 200, `the count must come from our bytes too, got ${v.genTokens}`)
  assert.ok(v.genTokens > 50, `and it must still be our own ~101 tokens, got ${v.genTokens}`)
})

test("a byte-estimated count stays marked as an estimate after the turn ends", () => {
  const t = new SpeedTracker()
  t.beginRun("a", 0)
  t.beginStep("a", "m1", 0)
  for (let i = 1; i <= 10; i++) t.pushDelta("a", "x".repeat(48), i * 100, "m1")
  assert.equal(t.value("a", 1_000)?.tokensEstimated, true)
  t.finish("a", 1_000) // no usage ever arrived
  const frozen = t.value("a", 1_100)
  assert.ok(frozen)
  assert.ok(frozen.genTokens > 0)
  assert.equal(frozen.tokensEstimated, true, "no usage payload ever settled this count")
})

test("a settled step plus a live byte estimate is still an estimate", () => {
  const t = new SpeedTracker()
  t.beginRun("a", 0)
  t.beginStep("a", "m1", 0)
  t.finishStep("a", "m1", { input: 100, output: 500 }, 1_000)
  assert.equal(t.value("a", 1_000)?.tokensEstimated, false, "settled from usage")
  t.beginStep("a", "m2", 2_000)
  for (let i = 1; i <= 10; i++) t.pushDelta("a", "x".repeat(48), 2_000 + i * 100, "m2")
  const v = t.value("a", 3_000)
  assert.ok(v)
  assert.ok(v.genTokens > 500, "the settled step is banked under the live one")
  assert.equal(v.tokensEstimated, true, "500 exact + ~100 guessed is a guess")
})

test("a run the host never ended stops counting as active", () => {
  const t = new SpeedTracker()
  t.beginRun("a", 0)
  t.beginStep("a", "m1", 0)
  t.pushDelta("a", "hello", 1_000, "m1")
  assert.equal(t.hasActive(2_000), true)
  assert.equal(t.hasActive(50_000), true, "a quiet minute is still the same turn")
  assert.equal(t.hasActive(70_000), false, "no bytes and no metric samples for 60s: the turn is over")
  t.applyMetrics("a", { t: 70_000, prefillLive: 0, genLive: 10, prefilling: false, running: true })
  t.applyMetrics("a", { t: 71_000, prefillLive: 0, genLive: 110, prefilling: false, running: true })
  assert.equal(t.hasActive(71_500), true, "a fresh gen sample revives it")
})
