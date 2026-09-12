import assert from "node:assert/strict"
import { test } from "node:test"
import {
  EMPTY_PROPS,
  ServiceTracker,
  downsample,
  fmtBytes,
  fmtCount,
  fmtDur,
  fmtExact,
  fmtGib,
  fmtMs,
  fmtRate,
  mbToGb,
  parseFeed,
  parseModels,
  parseProps,
  parseCacheTier,
  parseSampling,
  parseSpecStats,
  progressBar,
  shortModelName,
  levelBar,
  sparkline,
  wiredCeilingGb,
} from "./stats.ts"
import {
  CHAT_LINE,
  DFLASH_LINE,
  feed,
  GATED_LINE,
  LIVE_FEED,
  LIVE_MODELS,
  LIVE_PROPS,
  MTP_LINE,
  PLD_LINE,
  raw,
  RESPONSES_LINE,
  SSD_PARTIAL_LINE,
  SSD_TIER_LINE,
} from "./fixtures.ts"


test("fmtExact groups digits and never abbreviates", () => {
  assert.equal(fmtExact(1833), "1,833")
  assert.equal(fmtExact(17453), "17,453")
  assert.equal(fmtExact(999), "999")
  assert.equal(fmtExact(1_234_567), "1,234,567")
  assert.equal(fmtExact(8192.4), "8,192", "rounded, not truncated")
  assert.equal(fmtExact(Number.NaN), "0", "a broken counter cannot print NaN into the footer")
  assert.notEqual(fmtExact(1833), fmtCount(1833), "the narrow panel columns still abbreviate")
})

test("only a declared wired limit earns a gauge", () => {
  assert.equal(wiredCeilingGb(117.1875), 117.1875)
  assert.equal(wiredCeilingGb(null), null, "no declaration, no denominator \u2014 the row shows bytes alone")
  assert.equal(wiredCeilingGb(0), null)
  assert.equal(wiredCeilingGb(Number.NaN), null)
  assert.equal(mbToGb(120_000), 117.1875, "the sysctl speaks megabytes")
  assert.equal(mbToGb(0), null)
  assert.equal(mbToGb(null), null, "no OID at all is normal")
  assert.equal(mbToGb("120000" as unknown as number), null, "a string is not a size")
})


test("levelBar is a reading against a limit, in its own glyph", () => {
  assert.equal(levelBar(0.5, 10), "\u25ae\u25ae\u25ae\u25ae\u25ae\u2591\u2591\u2591\u2591\u2591")
  assert.equal(levelBar(null, 4), "\u2591\u2591\u2591\u2591", "unknown level is an empty gauge")
  assert.equal(levelBar(1.4, 4), "\u25ae\u25ae\u25ae\u25ae", "over the ceiling saturates, never overflows")
  assert.equal(levelBar(0.5, 0), "")
  assert.notEqual(levelBar(0.5, 2), progressBar(0.5, 2), "a level and a progress bar are drawn differently")
})

// --- /metrics.json ---------------------------------------------------------

test("parseFeed reads the live payload", () => {
  const f = parseFeed(raw(LIVE_FEED))
  assert.equal(f.counters.promptTokens, 347251)
  assert.equal(f.counters.prefillTokens, 132503)
  assert.equal(f.gauges.running, 1)
  assert.equal(f.gauges.genLive, 3094)
  assert.equal(f.gauges.prefillExpected, 0, "an older capture has no expected gauge: zero, never NaN")
  assert.equal(parseFeed(raw({ gauges: { prefill_tokens_live: 24_000, prefill_tokens_expected: 48_000 } })).gauges.prefillExpected, 48_000)
  assert.equal(f.gauges.mlxActiveBytes, 77386308302)
  assert.equal(f.histograms.decode_time_seconds?.count, 15)
})

test("parseFeed survives missing and hostile fields", () => {
  const f = parseFeed(raw({ gauges: { requests_running: -3, gpu_utilization_pct: "63", memory_mb: null } }))
  assert.equal(f.gauges.running, 0, "negatives are not stats")
  assert.equal(f.gauges.gpuPct, 0, "strings are not numbers")
  assert.equal(f.gauges.memMb, 0)
  assert.equal(f.counters.genTokens, 0, "missing counters read as zero")
  assert.deepEqual(f.histograms, {})
  assert.equal(parseFeed(null).counters.genTokens, 0)
  assert.equal(parseFeed(undefined).gauges.running, 0)
})

test("parseFeed drops histograms with no observations", () => {
  const f = parseFeed({
    histograms: { decode_time_seconds: { count: 0, sum: 0, bounds: [1], bucket_counts: [0, 0] } },
  })
  assert.equal(f.histograms.decode_time_seconds, undefined, "a count=0 histogram would divide by zero downstream")
})

// --- /props ----------------------------------------------------------------

test("parseProps reads memory headroom and ngram progress", () => {
  const p = parseProps(LIVE_PROPS)
  assert.ok(p.memory)
  assert.equal(Math.round(p.memory.activeGb * 100) / 100, 71.38)
  assert.equal(Math.round(p.memory.freeGb * 10), 445, "44.5 GiB free RAM")
  assert.equal(p.ngramProgress, 1)
  assert.deepEqual(parseProps(null), EMPTY_PROPS)
})

test("parseProps leaves ngram progress unknown without a total", () => {
  const p = parseProps({ ngram_warm: { bytes: 1000 } })
  assert.equal(p.ngramBytes, 1000)
  assert.equal(p.ngramProgress, null, "no total means no percentage, not 0%")
})

// --- /v1/models ------------------------------------------------------------

test("parseModels names the resident model and its spec decoder", () => {
  const m = parseModels(LIVE_MODELS)
  assert.ok(m)
  assert.equal(m.architecture, "qwen4_exp")
  assert.equal(m.quantization, "4-bit")
  assert.equal(m.layers, 48)
  assert.equal(m.contextLength, 1048576)
  assert.equal(m.kvQuant, "8")
  assert.equal(m.mtpLoaded, true)
  assert.equal(m.drafterLoaded, false)
})

test("parseModels ignores the model-card facts the panel does not draw", () => {
  const m = parseModels(LIVE_MODELS)!
  assert.equal("isMoe" in m, false, "architectural facts are parsed away with the rows that used them")
  assert.equal("genTemperature" in m, false, "the model's recommended sampling is not drawn; the log says what ran")
})

test("parseModels is null when nothing is loaded and tolerant of a bare id", () => {
  assert.equal(parseModels({ data: [] }), null)
  assert.equal(parseModels(null), null)
  assert.equal(parseModels({ data: "nope" }), null)
  const thin = parseModels({ data: [{ id: "m", loaded: false }] })
  assert.equal(thin?.id, "m", "falls back to the first entry so the panel can still name it")
  assert.equal(thin?.state, "unknown")
})

test("shortModelName strips the server-built id scaffolding", () => {
  assert.equal(shortModelName("Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit"), "Qwen3.8-Flash-Next")
  assert.equal(shortModelName("deepseek-v4-flash"), "deepseek-v4-flash")
  assert.equal(shortModelName("a-very-long-model-name-that-will-never-fit").length, 26)
  assert.equal(shortModelName("  spaced  "), "spaced")
  assert.equal(shortModelName("glm-4.6air-4bit"), "glm-4.6air", "a plain quantizer suffix goes too")
})

// --- [spec-stats] and request lines ---------------------------------------

test("parseSpecStats reads an mtp round", () => {
  const s = parseSpecStats(MTP_LINE)
  assert.ok(s)
  assert.deepEqual([s.mode, s.attempts, s.accepts], ["mtp", 168, 191])
  assert.equal(s.avgPerRound, 1.14)
  assert.equal(s.perDraftPct, 67.7)
  assert.equal(s.drafted, 282)
  assert.equal(s.runtimeDisabled, false)
  assert.equal(s.reason, "none")
  assert.equal(s.adaptive, "mtp")
  assert.deepEqual([s.syncMs, s.roundMs], [2.93, 47.31])
})

test("parseSpecStats reports the runtime gate going off, with its reason", () => {
  const s = parseSpecStats(GATED_LINE)
  assert.ok(s)
  assert.equal(s.runtimeDisabled, true)
  assert.equal(s.reason, "adaptive")
  assert.equal(s.adaptive, "serial", "it fell back to serial decoding")
  assert.equal(s.perDraftPct, 38.4)
})

test("parseSpecStats handles modes without a per-draft percentage", () => {
  const pld = parseSpecStats(PLD_LINE)
  assert.ok(pld)
  assert.equal(pld.perDraftPct, null, "PLD depth varies, so there is no fixed denominator")
  assert.equal(pld.avgPerRound, 1.49)

  const dflash = parseSpecStats(DFLASH_LINE)
  assert.ok(dflash)
  assert.equal(dflash.perDraftPct, 74.4)
  assert.equal(dflash.drafted, null)
})

test("parseSpecStats ignores anything that is not a spec line", () => {
  assert.equal(parseSpecStats("Decode (mtp): 1200ms (40 tokens)"), null)
  assert.equal(parseSpecStats("[spec-stats] attempts=1"), null, "no mode is not a mode")
  assert.equal(parseSpecStats(""), null)
})

test("parseSampling reads the sampling params mlx-serve actually served", () => {
  const s = parseSampling(CHAT_LINE)
  assert.ok(s)
  assert.equal(s.endpoint, "chat/completions")
  assert.deepEqual([s.temperature, s.topP, s.topK], [1, 0.95, 20])
  assert.equal(s.maxTokens, 64000)
  assert.equal(s.maxTokensOrigin, "launch default")
  assert.equal(s.stream, true)
  assert.equal(s.messages, 127)
})

test("parseSampling handles an endpoint that reports less", () => {
  const s = parseSampling(RESPONSES_LINE)
  assert.ok(s)
  assert.equal(s.temperature, 0.7)
  assert.equal(s.topP, null, "/v1/responses logs no top_p")
  assert.equal(s.topK, null)
  assert.equal(s.maxTokens, 4096)
  assert.equal(s.maxTokensOrigin, null)
})

test("parseSampling ignores non-request lines", () => {
  assert.equal(parseSampling("GET /v1/models"), null)
  assert.equal(parseSampling("POST /v1/chat/completions (1 msgs, stream=true)"), null, "no temp= is no sampling")
  const garbage = parseSampling(CHAT_LINE.replace("temp=1.00", "temp=nan"))
  assert.ok(garbage, "an unparsable temp is still a request we want named")
  assert.equal(garbage.temperature, null, "it reads as unknown, never as 0")
})

test("field lookup cannot match a suffix of a longer key", () => {
  const s = parseSampling(CHAT_LINE)
  assert.ok(s)
  assert.equal(s.topK, 20, "top_k must not be read from k= or tool_k=")
})

test("histogram sums still drive the cumulative rates", () => {
  const f = parseFeed(raw(LIVE_FEED))
  assert.equal(Math.round((f.counters.prefillTokens / f.histograms.prefill_time_seconds!.sum) * 10) / 10, 1310.8)
  assert.equal(f.histograms.time_to_first_token_seconds?.count, 15, "the feed still carries the latency histograms")
})

// --- ServiceTracker rates --------------------------------------------------

test("decode rate comes from gen_live over the trailing window", () => {
  const t = new ServiceTracker()
  t.sample(feed({ gauges: { generation_tokens_live: 1000, requests_running: 1 } }), 1_000)
  t.sample(feed({ gauges: { generation_tokens_live: 1030, requests_running: 1 } }), 2_000)
  const s = t.statsAt(2_000)
  assert.ok(s)
  assert.equal(s.genTps, 30, "30 tokens in 1s")
  assert.equal(s.phase, "decode")
  assert.equal(s.link, "live")
})

test("prefill rate needs the live prefill gauge to be moving", () => {
  const t = new ServiceTracker()
  t.sample(feed({ gauges: { prefill_tokens_live: 8192, requests_prefilling: 1, requests_running: 1 } }), 1_000)
  t.sample(feed({ gauges: { prefill_tokens_live: 24576, requests_prefilling: 1, requests_running: 1 } }), 3_000)
  const s = t.statsAt(3_000)
  assert.ok(s)
  assert.equal(s.prefillTps, 8192, "16384 tokens over 2s")
  assert.equal(s.phase, "prefill")
})

test("live rates go null once the feed stops, cumulative stats stay", () => {
  const t = new ServiceTracker()
  t.sample(feed({ gauges: { generation_tokens_live: 1000, requests_running: 1 } }), 1_000)
  t.sample(feed({ gauges: { generation_tokens_live: 1030, requests_running: 1 } }), 2_000)
  const stale = t.statsAt(30_000)
  assert.ok(stale)
  assert.equal(stale.genTps, null, "a 28s-old sample says nothing about now")
  assert.equal(stale.reqPerSec, null)
  assert.equal(stale.phase, "idle")
  assert.equal(stale.requestsOk, 15)
  assert.notEqual(stale.avgPrefillTps, null, "cumulative prefill speed survives")
})

test("two samples closer than the window floor do not report a rate", () => {
  const t = new ServiceTracker()
  t.sample(feed({ gauges: { generation_tokens_live: 1000, requests_running: 1 } }), 1_000)
  t.sample(feed({ gauges: { generation_tokens_live: 1030, requests_running: 1 } }), 1_050)
  assert.equal(t.statsAt(1_050)?.genTps, null, "a 50ms double-read is a poll artifact, not 600 t/s")
})

test("idle gauges with a live feed read as idle or queued, not a stuck decode", () => {
  const t = new ServiceTracker()
  t.sample(feed({ gauges: { requests_running: 0, requests_waiting: 0, requests_prefilling: 0 } }), 1_000)
  assert.equal(t.statsAt(1_500)?.phase, "idle")
  t.sample(feed({ gauges: { requests_running: 0, requests_waiting: 2, requests_prefilling: 0 } }), 2_000)
  const queued = t.statsAt(2_000)
  assert.equal(queued?.phase, "queued")
  assert.equal(queued?.waiting, 2)
  assert.equal(queued?.genTps, null)
})

test("a server restart clears history instead of printing negative rates", () => {
  const t = new ServiceTracker()
  t.sample(feed({ counters: { generation_tokens_total: 900_000 }, gauges: { generation_tokens_live: 900_000, requests_running: 1 } }), 1_000)
  t.sample(feed({ counters: { generation_tokens_total: 10 }, gauges: { generation_tokens_live: 10, requests_running: 1 } }), 2_000)
  const s = t.statsAt(2_000)
  assert.ok(s)
  assert.equal(s.genTps, null, "the counter went backwards: no rate, ever")
  assert.deepEqual(s.genSeries, [0], "the pre-restart sparkline is gone; only the new point stands")
  assert.equal(s.link, "live")
})

test("avg prefill divides forwarded tokens, never billed prompt tokens", () => {
  const t = new ServiceTracker()
  const warmed = parseFeed({
    counters: {
      prompt_tokens_total: 1_000_000,
      prefill_tokens_total: 100,
      prefix_cache_tokens_total: 999_900,
      generation_tokens_total: 50,
    },
    gauges: { requests_running: 0 },
    histograms: { prefill_time_seconds: { count: 1, sum: 1, bounds: [5], bucket_counts: [1, 1] } },
  })
  t.sample(warmed, 1_000)
  const s = t.statsAt(1_000)
  assert.ok(s)
  assert.equal(s.avgPrefillTps, 100, "100 forwarded tokens ÷ 1s, not 1M")
  assert.equal(s.cacheTokenPct, 100, "all but 100 tokens were restored, not computed")
})

test("noteLink marks a dark feed and the next sample revives it", () => {
  const t = new ServiceTracker()
  t.sample(feed(), 1_000)
  t.noteLink("down")
  assert.equal(t.statsAt(1_000)?.link, "down")
  t.noteLink("disabled")
  assert.equal(t.statsAt(1_000)?.link, "disabled")
  t.sample(feed(), 2_000)
  assert.equal(t.statsAt(2_000)?.link, "live")
  assert.equal(new ServiceTracker().statsAt(1_000), null, "no feed at all")
})


test("props fill the memory gaps the metrics feed leaves", () => {
  const t = new ServiceTracker()
  t.sample(parseFeed({ gauges: { requests_running: 0 } }), 1_000)
  assert.equal(t.statsAt(1_000)?.freeRamGb, null, "nothing to show before /props answers")
  t.noteProps(parseProps(LIVE_PROPS))
  assert.equal(Math.round((t.statsAt(1_000)?.freeRamGb ?? 0) * 10), 445)
})

// --- histogram math --------------------------------------------------------





// --- sparkline -------------------------------------------------------------

test("series buckets one point per second while the phase runs", () => {
  const t = new ServiceTracker()
  for (let i = 0; i < 6; i++) {
    t.sample(feed({ counters: { generation_tokens_total: 1000 + i * 10 }, gauges: { generation_tokens_live: 1000 + i * 10, requests_running: 1 } }), 1_000 + i * 1_000)
  }
  const s = t.statsAt(6_000)
  assert.ok(s)
  assert.equal(s.genSeries.length, 6, "six seconds of samples, no phantom leading gap")
  assert.ok(s.genSeries.slice(1).every((v) => v > 0), "the first second has no window yet, so it is 0")
  assert.equal("prefillSeries" in s, false, "only the decode series is plotted, so only it is kept")
})

test("sparkline scales to the peak and downsamples", () => {
  assert.equal(sparkline([], 8), "")
  assert.equal(sparkline([5, 5, 5, 5], 4), "████", "flat series is a solid bar")
  assert.equal(sparkline([0, 4], 2), "▁█")
  assert.equal(sparkline([1, 2, 3, 4], 2), "▅█", "keeps the peak of each slice")
  assert.equal(downsample([1, 2, 3], 8).length, 3, "a short series is not stretched")
  assert.equal(sparkline([0, 0], 2), "▁▁")
  assert.equal(sparkline([1, 2], 0), "", "cells=0 turns the row off")
})

// --- formatting ------------------------------------------------------------

test("formatters keep narrow columns", () => {
  assert.equal(fmtRate(24.6), "24.6")
  assert.equal(fmtRate(1234), "1234")
  assert.equal(fmtRate(12345), "12.3k")
  assert.equal(fmtRate(null), "—")
  assert.equal(fmtMs(430), "430ms")
  assert.equal(fmtMs(1234), "1.23s")
  assert.equal(fmtMs(23456), "23s")
  assert.equal(fmtMs(null), "—")
  assert.equal(fmtGib(71.37), "71.4G")
  assert.equal(fmtGib(120), "120G")
  assert.equal(fmtGib(0), "—")
  assert.equal(fmtGib(null), "—")
  assert.equal(fmtCount(999), "999")
  assert.equal(fmtCount(1500), "1.5k")
  assert.equal(fmtCount(2_400_000), "2.4M")
  assert.equal(fmtDur(1_500), "1.5s")
  assert.equal(fmtDur(65_400), "1m05s")
  assert.equal(fmtDur(125_000), "2m05s")
  assert.equal(fmtBytes(700), "700B")
  assert.equal(fmtBytes(40 * 1024), "40K")
  assert.equal(fmtBytes(13_500_000), "12.9M")
  assert.equal(fmtBytes(2 * 1024 ** 3), "2.00G")
})

// --- regressions -----------------------------------------------------------

test("a prefill after a long idle is rated over the prefill, not the window", () => {
  const t = new ServiceTracker()
  // A minute of idle samples: prefill_tokens_live reads 0 per request.
  for (let s = 0; s <= 60; s++) {
    t.sample(feed({ gauges: { prefill_tokens_live: 0, requests_prefilling: 0, requests_running: 0 } }), s * 1_000)
  }
  t.sample(feed({ gauges: { prefill_tokens_live: 16_384, requests_prefilling: 1, requests_running: 1 } }), 62_000)
  const s = t.statsAt(62_000)
  assert.ok(s)
  assert.equal(s.prefillTps, 8192, "16,384 tokens in the 2s the prefill took, not over a 30s window")
})

test("a gap in decode reads as zero, and the series ends at now", () => {
  const t = new ServiceTracker()
  const busy = (i: number, live: number) =>
    t.sample(feed({ counters: { generation_tokens_total: live }, gauges: { generation_tokens_live: live, requests_running: 1 } }), i * 1_000)
  for (let i = 1; i <= 5; i++) busy(i, 1_000 + i * 100) // 100 t/s
  // A five-second tool pause: nothing is running, so nothing is binned.
  for (let i = 6; i <= 10; i++) {
    t.sample(feed({ counters: { generation_tokens_total: 1_500 }, gauges: { generation_tokens_live: 1_500, requests_running: 0 } }), i * 1_000)
  }
  for (let i = 11; i <= 15; i++) busy(i, 1_500 + (i - 10) * 20) // 20 t/s
  const s = t.statsAt(15_000)
  assert.ok(s)
  assert.equal(s.genSeries.length, 15, "one point per second from the first bin to now")
  assert.deepEqual(s.genSeries.slice(5, 10), [0, 0, 0, 0, 0], "the tool pause is a gap, not sustained decoding")
  assert.ok((s.genSeries[4] ?? 0) > 90, "the busy seconds before it are still there")

  const after = t.statsAt(20_000)
  assert.ok(after)
  assert.deepEqual(after.genSeries.slice(-5), [0, 0, 0, 0, 0], "five seconds after decode stopped the trace is flat")
})

test("a dead feed stops claiming work in flight", () => {
  const t = new ServiceTracker()
  t.sample(feed({ gauges: { requests_running: 1, requests_waiting: 2, requests_prefilling: 1, gpu_utilization_pct: 63 } }), 1_000)
  const live = t.statsAt(1_000)
  assert.equal(live?.running, 1)
  const dead = t.statsAt(30_000)
  assert.ok(dead)
  assert.deepEqual(
    [dead.running, dead.waiting, dead.prefilling, dead.gpuPct],
    [0, 0, 0, 0],
    "an unreachable feed must not draw `running 1` and `gpu 63%` under it",
  )
})

test("the disk tier parses whatever the write parenthetical says", () => {
  assert.deepEqual(parseCacheTier(SSD_TIER_LINE), { kind: "ssd", residentMb: 10933.6, capMb: null })
  assert.deepEqual(
    parseCacheTier(SSD_PARTIAL_LINE),
    { kind: "ssd", residentMb: 38009.6, capMb: null },
    "a partial persist with no ssm-cp copies is the same row",
  )
  assert.equal(parseCacheTier("  [disk-cache] e17 complete on disk: 865171 tokens, 845 chunks, 16 ssm-cp"), null, "not an occupancy line")
})

test("the request line parses with the trailing space the server writes", () => {
  const s = parseSampling(CHAT_LINE)
  assert.ok(CHAT_LINE.endsWith(") "), "captured verbatim, trailing space and all")
  assert.equal(s?.toolMsgs, 68)
  assert.equal(s?.stream, true)
})
