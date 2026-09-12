import assert from "node:assert/strict"
import { test } from "node:test"
import { parseCacheTier, parseFeed, parseModels, parseProps, parseSampling, parseSpecStats, progressBar, ServiceTracker } from "./stats.ts"
import {
  ALL_SECTIONS,
  SECTION_TITLES,
  buildSections,
  clip,
  footerLabel,
  feedStatus,
  resolveSections,
  turnRows,
  type PanelInput,
  type SidebarRow,
} from "./rows.ts"
import { HOT_TIER_LINE, SSD_TIER_LINE, CHAT_LINE, GATED_LINE, LIVE_MODELS, LIVE_PROPS, MTP_LINE, RESPONSES_LINE, feed, raw } from "./fixtures.ts"
import type { LogStatus, Observed } from "./logtail.ts"
import type { Link, SamplingStats, ServiceStats } from "./stats.ts"
import type { SpeedValue } from "./tracker.ts"

const NOW = 1_800_000_000_000

function observed<T>(value: T, at = NOW): Observed<T> {
  return { value, at }
}

function logStatus(part: Partial<LogStatus> = {}): LogStatus {
  return {
    path: "/Users/beam/.mlx-serve/logs/mlx-serve-11234.log",
    name: "mlx-serve-11234.log",
    bytes: 302 * 1024,
    mtimeMs: NOW - 1_500,
    error: null,
    lines: 4,
    dropped: 0,
    ...part,
  }
}

/** A tracker fed the live payload twice, so both rates and averages exist. */
function service(overrides: { counters?: never; gauges?: never } = {}): ServiceStats {
  const t = new ServiceTracker()
  t.sample(feed(), NOW - 4_000)
  t.sample(feed(), NOW)
  t.noteProps(parseProps(LIVE_PROPS))
  const stats = t.statsAt(NOW)
  assert.ok(stats)
  return stats
}

function speed(overrides: Partial<SpeedValue> = {}): SpeedValue {
  return {
    phase: "generate",
    prefillTokens: 128_000,
    prefillExpected: null,
    prefillTps: 4200,
    genTokens: 1400,
    genTps: 24.6,
    ttftMs: 430,
    elapsedMs: 62_000,
    tokensEstimated: false,
    ...overrides,
  }
}

function input(part: Partial<PanelInput> = {}): PanelInput {
  return {
    speed: null,
    service: service(),
    model: parseModels(LIVE_MODELS),
    spec: observed(parseSpecStats(MTP_LINE)!),
    sampling: observed(parseSampling(CHAT_LINE)!),
    log: logStatus(),
    sparkCells: 0,
    barCells: 0,
    now: NOW,
    link: "live",
    ...part,
  }
}

/** As the TUI draws it: a row with an empty label is a continuation line. */
const text = (rows: readonly SidebarRow[]) =>
  rows.map((r) => `${r.label === "" ? "" : `${r.label} `}${r.value}${r.note ? ` ${r.note}` : ""}`)

const render = (sec: { rows: readonly SidebarRow[] }) => text(sec.rows)

const obs = <T,>(value: T) => ({ value, at: NOW })

/** The Prefix cache section as the panel draws it, with the tiers it was handed. */
const cacheRowsOf = (part: Partial<PanelInput>) =>
  text(buildSections(input(part), ["cache"])[0]?.rows ?? [])

const section = (name: string, i: PanelInput = input()) =>
  buildSections(i, [name as never]).flatMap((s) => text(s.rows))

// --- header ----------------------------------------------------------------

test("the feed status is a heading aside, not a row", () => {
  assert.deepEqual(feedStatus(service().link), { value: "live", tone: "live" })
  const link = (which: "disabled" | "down" | "unauthorized" | "unknown") => {
    const t = new ServiceTracker()
    if (which !== "unknown") t.sample(feed(), NOW)
    t.noteLink(which)
    return feedStatus(which === "unknown" ? undefined : t.statsAt(NOW)?.link ?? t.linkState())
  }
  assert.deepEqual(link("disabled"), { value: "--metrics off", tone: "warn" }, "a server without --metrics is a config fact, not an outage")
  assert.deepEqual(link("down"), { value: "unreachable", tone: "error" })
  assert.deepEqual(link("unauthorized"), { value: "401 unauthorized", tone: "error" })
  assert.deepEqual(link("unknown"), { value: "connecting", tone: "muted" })
  assert.deepEqual(feedStatus(undefined), { value: "connecting", tone: "muted" })
})

test("turn rows are the footer meter, line by line", () => {
  assert.deepEqual(section("turn", input({ speed: speed() })), [
    "decode 24.6 t/s · 1.4k tok",
    "prefill 4200 t/s 128.0k tok · last",
    "ttft 430ms · turn 1m02s",
  ])
})

test("turn rows say what they are while the prompt is still being read", () => {
  assert.deepEqual(text(turnRows(speed({ phase: "prefill", genTps: null, genTokens: 0, ttftMs: null }))), [
    "prefill 4200 t/s · 128.0k tok",
  ])
  assert.deepEqual(
    text(turnRows(speed({ phase: "prefill", prefillTps: null, prefillTokens: null, genTps: null, ttftMs: null, elapsedMs: 400 }))),
    ["prefill waiting · 0.4s"],
  )
  assert.deepEqual(turnRows(null), [])
})

test("turn rows keep the estimate marker the footer HUD had", () => {
  const rows = turnRows(speed({ tokensEstimated: true }))
  assert.equal(rows[0]?.value, "~24.6 t/s", "byte-estimated tokens are marked as estimates")
  assert.equal(rows[0]?.note, "· ~1.4k tok")
})

// --- server sections -------------------------------------------------------

test("throughput shows live rates when the server is busy and zeros with averages when it is not", () => {
  const busy = new ServiceTracker()
  busy.sample(feed({ gauges: { generation_tokens_live: 1000, requests_running: 1 } }), NOW - 1_000)
  busy.sample(feed({ gauges: { generation_tokens_live: 1030, requests_running: 1 } }), NOW)
  assert.deepEqual(section("throughput", input({ service: busy.statsAt(NOW)! })), [
    "decode 30.0 t/s · avg 68.9",
    "prefill 0.0 t/s · avg 1311",
    "admitted 0.00 req/s",
  ])

  const idle = new ServiceTracker()
  idle.sample(feed({ gauges: { requests_running: 0, requests_prefilling: 0 } }), NOW)
  assert.deepEqual(section("throughput", input({ service: idle.statsAt(NOW)! })), [
    "decode 0.0 t/s · avg 68.9",
    "prefill 0.0 t/s · avg 1311",
    "admitted 0.00 req/s",
  ])
})

test("with nothing measured since boot, both lines still show 0", () => {
  const t = new ServiceTracker()
  t.sample(
    parseFeed(
      raw({
        counters: {
          prompt_tokens_total: 0,
          prefill_tokens_total: 0,
          generation_tokens_total: 0,
          requests_success_total: 0,
          prefix_cache_queries_total: 0,
          prefix_cache_hits_total: 0,
        },
        gauges: { requests_running: 0, requests_waiting: 0, requests_prefilling: 0 },
        histograms: {},
      }),
    ),
    NOW,
  )
  assert.deepEqual(section("throughput", input({ service: t.statsAt(NOW)! })), [
    "decode 0.0 t/s · avg —",
    "prefill 0.0 t/s · avg —",
    "admitted 0.00 req/s",
  ])
})

test("a phase that ends keeps its Throughput line", () => {
  // The blink this fixes: prefill measured, decode idle (and the other way
  // round) used to draw one line where the idle server draws two, so every line
  // under Throughput jumped twice per turn.
  const lines = (s: ServiceStats) => section("throughput", input({ service: s }))

  const prefilling = new ServiceTracker()
  prefilling.sample(
    feed({ gauges: { prefill_tokens_live: 8192, requests_prefilling: 1, requests_running: 1 } }),
    NOW - 2_000,
  )
  prefilling.sample(
    feed({ gauges: { prefill_tokens_live: 24576, requests_prefilling: 1, requests_running: 1 } }),
    NOW,
  )
  const prefillStats = prefilling.statsAt(NOW)!
  assert.equal(prefillStats.phase, "prefill")
  assert.equal(prefillStats.genTps, null, "no token decoded in this window")
  assert.deepEqual(lines(prefillStats), [
    "decode 0.0 t/s · avg 68.9",
    "prefill 8192 t/s · avg 1311",
    "admitted 0.00 req/s",
  ])

  const decoding = new ServiceTracker()
  decoding.sample(feed({ gauges: { generation_tokens_live: 1000, requests_running: 1 } }), NOW - 1_000)
  decoding.sample(feed({ gauges: { generation_tokens_live: 1030, requests_running: 1 } }), NOW)
  assert.deepEqual(
    lines(decoding.statsAt(NOW)!).map((l) => l.split(" ")[0]),
    ["decode", "prefill", "admitted"],
    "decoding draws the same lines prefilling draws",
  )

  const idle = new ServiceTracker()
  idle.sample(feed({ gauges: { requests_running: 0, requests_prefilling: 0 } }), NOW)
  assert.deepEqual(
    lines(idle.statsAt(NOW)!).map((l) => l.split(" ")[0]),
    ["decode", "prefill", "admitted"],
    "and the same lines idle",
  )
})

test("the prefill bar fills against the target, saturates past it", () => {
  assert.equal(progressBar(0.5, 12), "██████░░░░░░")
  assert.equal(progressBar(0, 4), "░░░░")
  assert.equal(progressBar(null, 4), "░░░░", "unknown progress is an empty bar, never a half one")
  assert.equal(progressBar(2, 4), "████", "a turn bigger than the target saturates instead of throwing")
  assert.equal(progressBar(0.3, 0), "", "zero cells means no bar at all")
})

test("prefill becomes a progress bar with speed and x/y tokens", () => {
  const rows = turnRows(
    speed({
      phase: "prefill",
      prefillTokens: 12_400,
      prefillExpected: 48_000,
      prefillTps: 1_600,
      genTps: null,
      ttftMs: null,
    }),
    1,
    12,
  )
  assert.deepEqual(text(rows), [
    "prefill ███░░░░░░░░░ 12.4k/48.0k",
    "1600 t/s · ~22s left",
  ])
})

test("the bar's denominator is marked as an estimate", () => {
  const rows = turnRows(
    speed({ phase: "prefill", prefillTokens: 1_000, prefillExpected: 4_000, prefillTps: null, genTps: null, ttftMs: null }),
    1,
    8,
  )
  assert.deepEqual(text(rows), ["prefill ██░░░░░░ 1.0k/4.0k", "measuring"])
})

test("no target yet: the prefill line stays a plain rate", () => {
  const firstTurn = speed({ phase: "prefill", prefillTokens: 8_192, prefillExpected: null, genTps: null, ttftMs: null })
  assert.deepEqual(text(turnRows(firstTurn, 1, 12)), ["prefill 4200 t/s · 8.2k tok"])
  const cold = speed({ phase: "prefill", prefillTokens: null, prefillExpected: null, prefillTps: null, genTps: null, ttftMs: null })
  assert.deepEqual(text(turnRows(cold, 1, 12)), ["prefill waiting · 1m02s"], "nothing to measure, so nothing is invented")
})

test("barCells 0 keeps the old two-line meter exactly", () => {
  const rows = turnRows(speed({ phase: "prefill", genTps: null, ttftMs: null }), 1, 0)
  assert.deepEqual(text(rows), ["prefill 4200 t/s · 128.0k tok"])
})

test("the footer decode line always shows all three slots", () => {
  assert.equal(footerLabel(speed({ prefillTps: null })), "1,400 tok · decode 24.6 t/s · ttft 430ms")
  assert.equal(
    footerLabel(speed({ genTps: null, prefillTps: 4200 })),
    "1,400 tok · decode 0.0 t/s · prefill 4200 t/s",
    "an unsettled rate reads as zero, not as a missing clause",
  )
  assert.equal(
    footerLabel(speed({ genTokens: 0, genTps: null, prefillTps: null, ttftMs: null, tokensEstimated: true })),
    "~0 tok · decode ~0.0 t/s · prefill 0.0 t/s",
    "estimates stay marked, zeros stay visible",
  )
  assert.equal(
    footerLabel(speed({ genTps: null, prefillTps: null, ttftMs: 1_500 })),
    "1,400 tok · decode 0.0 t/s · ttft 1.50s",
    "a stream-measured prefill reports its TTFT",
  )
})

test("the footer prefill shapes zero-fill every slot", () => {
  assert.equal(
    footerLabel(
      speed({ phase: "prefill", prefillTokens: 12_400, prefillExpected: 48_000, prefillTps: 1_600, genTps: null, ttftMs: null }),
      { barCells: 10 },
    ),
    "prefill ███░░░░░░░ 12,400/48,000 · 1600 t/s · ~22s left",
  )
  assert.equal(
    footerLabel(
      speed({ phase: "prefill", prefillTokens: null, prefillExpected: 48_000, prefillTps: null, genTps: null, ttftMs: null }),
      { barCells: 10 },
    ),
    "prefill ░░░░░░░░░░ 0/48,000 · 0.0 t/s · measuring",
  )
  assert.equal(
    footerLabel(speed({ phase: "prefill", prefillTokens: null, prefillExpected: null, prefillTps: null, genTps: null, ttftMs: null })),
    "prefill waiting · 1m02s",
    "no server metrics: the wait is the only prefill measure, and it counts up",
  )
})

test("the Throughput prefill line becomes the real progress bar", () => {
  const busy = new ServiceTracker()
  busy.sample(
    feed({ gauges: { prefill_tokens_live: 12_000, prefill_tokens_expected: 48_000, requests_prefilling: 1, requests_running: 1 } }),
    NOW - 1_000,
  )
  busy.sample(
    feed({ gauges: { prefill_tokens_live: 24_000, prefill_tokens_expected: 48_000, requests_prefilling: 1, requests_running: 1 } }),
    NOW,
  )
  const rows = section("throughput", input({ service: busy.statsAt(NOW), barCells: 12 }))
  assert.match(
    rows.find((l) => l.startsWith("prefill ")) ?? "",
    /^prefill [█░]{12} 24\.0k\/48\.0k · 12\.0k t\/s$/,
    "half the target fills half the 12-block bar",
  )
  // No target (idle, or an older build): the rate line stays.
  assert.match(
    section("throughput", input({ barCells: 12 })).find((l) => l.startsWith("prefill ")) ?? "",
    /^prefill 0\.0 t\/s · avg/,
  )
})

test("one client in flight: throughput does not repeat the turn's rate", () => {
  const busy = new ServiceTracker()
  busy.sample(feed({ gauges: { generation_tokens_live: 1000, requests_running: 1 } }), NOW - 1_000)
  busy.sample(feed({ gauges: { generation_tokens_live: 1030, requests_running: 1 } }), NOW)
  const stats = busy.statsAt(NOW)
  assert.ok(stats)
  const rows = buildSections(
    input({ speed: speed({ genTps: 30, prefillTps: null, prefillTokens: null, ttftMs: null }), service: stats }),
    ["turn", "throughput"],
  ).flatMap(render)
  assert.deepEqual(rows, [
    "decode 30.0 t/s · 1.4k tok",
    "decode 68.9 t/s · since boot",
    "prefill 0.0 t/s · avg 1311",
    "admitted 0.00 req/s",
  ])
})

test("Throughput carries the admission rate the Queue section used to own", () => {
  const busy = new ServiceTracker()
  // Admissions are a counter over a 60s window, so the fixture has to move it.
  for (let i = 0; i < 5; i++) {
    busy.sample(
      feed({ counters: { requests_success_total: 15 + i }, gauges: { generation_tokens_live: 1000 + i * 10, requests_running: 1 } }),
      NOW - (4 - i) * 1_000,
    )
  }
  const rows = section("throughput", input({ service: busy.statsAt(NOW) }))
  assert.ok(rows.some((l) => /^admitted \d\.\d\d req\/s$/.test(l)), `expected an admitted line in ${rows.join(" / ")}`)
})

test("two clients in flight: both rates stay, and the turn's names the crowd", () => {
  const busy = new ServiceTracker()
  busy.sample(feed({ gauges: { generation_tokens_live: 1000, requests_running: 2 } }), NOW - 1_000)
  busy.sample(feed({ gauges: { generation_tokens_live: 1130, requests_running: 2 } }), NOW)
  const stats = busy.statsAt(NOW)
  assert.ok(stats)
  assert.equal(stats.running, 2)
  const rows = buildSections(
    input({ speed: speed({ genTps: 12.5, prefillTps: null, prefillTokens: null, ttftMs: null }), service: stats }),
    ["turn", "throughput"],
  ).flatMap(render)
  assert.deepEqual(rows, [
    "decode 12.5 t/s · 1.4k tok · 2 clients",
    "decode 130 t/s · avg 68.9",
    "prefill 0.0 t/s · avg 1311",
    "admitted 0.00 req/s",
  ])
  assert.equal(rows.length, 4, "two clients: the server-wide rate is a different measurement, so it stays")
})

test("the sparkline row is opt-in by cells and never stretches past them", () => {
  const busy = new ServiceTracker()
  for (let i = 0; i < 6; i++) {
    busy.sample(feed({ gauges: { generation_tokens_live: 1000 + i * 10, requests_running: 1 } }), NOW - (5 - i) * 1_000)
  }
  const rows = buildSections(input({ service: busy.statsAt(NOW)!, sparkCells: 8 }), ["throughput"])[0]?.rows ?? []
  const spark = rows.find((r) => r.label === "60s")
  assert.ok(spark)
  assert.equal(spark.value.length, 6, "six seconds of history draw six cells, not eight")
  assert.equal(sparklineWidth(spark.value), 6)
})

/** Sparkline blocks are one cell each regardless of byte length. */
function sparklineWidth(spark: string): number {
  return [...spark].length
}

test("cache rows read from the feed alone", () => {
  assert.deepEqual(section("cache"), ["tokens 62% · from cache", "requests 60% · had a hit"])
})

test("the queue has no section of its own any more", () => {
  assert.deepEqual(buildSections(input(), ["queue" as never]), [], "unknown names draw nothing, they do not fall back")
  assert.equal(section("server").find((l) => l.startsWith("running ")), "running 1 · 0 waiting")
})

test("memory rows carry the allocator split and the accelerator warm", () => {
  assert.deepEqual(section("memory"), [
    "footprint 73.3G",
    "mlx-serve 72.1G · pool 0.7G",
    "free 44.5G · peak 75.7G",
    "ngram 29.8G",
  ])

  const partial = new ServiceTracker()
  partial.sample(parseFeed({ gauges: { memory_mb: 1000 } }), NOW)
  assert.deepEqual(section("memory", input({ service: partial.statsAt(NOW)! })), ["footprint 1.0G"], "no props, no claims")

  const warming = { ...service(), ngramBytes: 8 * 1024 ** 3, ngramProgress: 0.25 }
  assert.deepEqual(text(memoryRowsOf(warming)).at(-1), "ngram 25% · 8.0G read", "a partial warm is a percentage")
})

function memoryRowsOf(s: ServiceStats): readonly SidebarRow[] {
  return buildSections(input({ service: s }), ["memory"])[0]?.rows ?? []
}

test("Server shows load, then the serving totals", () => {
  assert.deepEqual(section("server"), [
    "gpu 63%",
    "running 1 · 0 waiting",
    "tokens 347.3k in · 3.3k out",
    "requests 15 ok · 0 cancelled",
    "messages 127",
    "tool calls 68",
  ])
})

test("Server stops reciting the model card", () => {
  const lines = section("server")
  assert.equal(lines.some((l) => l.includes("4-bit")), false, "weight quantization is not a serving statistic")
  assert.equal(lines.some((l) => /layers/.test(l)), false, "layer count never changes while it runs")
  assert.equal(lines.some((l) => /moe/.test(l)), false, "is_moe explains nothing about this turn")
  assert.ok(lines.every((l) => [...l].length <= 34), `every Server line must fit the sidebar: ${lines.filter((l) => [...l].length > 34).join(" / ")}`)
  assert.equal(lines.some((l) => l.startsWith("model ")), false, "the model card moved to Model & sampling")
  const card = section("sampling")
  assert.ok(card.includes("model Qwen3.8-Flash-Next"), "the model row moved, it did not leave the panel")
  assert.ok(card.includes("context 1,048,576"), "exact count, the way the host writes its context meter")
})

test("a third client waiting shows on the merged queue line", () => {
  const loaded = new ServiceTracker()
  loaded.sample(feed({ gauges: { requests_running: 2, requests_waiting: 3, requests_prefilling: 1 } }), NOW)
  assert.equal(section("server", input({ service: loaded.statsAt(NOW) })).find((l) => l.startsWith("running ")), "running 2 · 3 waiting · 1 prefilling")
})

test("Server still names the model when nothing is running", () => {
  const idle = new ServiceTracker()
  idle.sample(feed({ gauges: { requests_running: 0, requests_waiting: 0, gpu_utilization_pct: 0 } }), NOW)
  assert.deepEqual(section("server", input({ service: idle.statsAt(NOW) })), [
    "gpu 0%",
    "running 0 · 0 waiting",
    "tokens 347.3k in · 3.3k out",
    "requests 15 ok · 0 cancelled",
    "messages 127",
    "tool calls 68",
  ], "an idle GPU reads 0%, it does not take its line")
})

test("no model loaded: serving statistics draw without the card", () => {
  assert.deepEqual(section("server", input({ model: null })), [
    "gpu 63%",
    "running 1 · 0 waiting",
    "tokens 347.3k in · 3.3k out",
    "requests 15 ok · 0 cancelled",
    "messages 127",
    "tool calls 68",
  ])
})

test("a declared wired limit puts the bar on its own line under Memory", () => {
  const withCeiling = buildSections(input({ wiredLimitGb: 117.1875, ratioCells: 10 }), ["memory"])[0]
  assert.ok(withCeiling)
  assert.equal(withCeiling.note, undefined, "the heading stays still while the numbers move")
  assert.deepEqual(text(withCeiling.rows), [
    "▮▮▮▮▮▮░░░░ 63% · of 117G wired",
    "mlx-serve 72.1G · pool 0.7G",
    "free 44.5G · peak 75.7G",
    "ngram 29.8G",
  ], "no footprint row: it would repeat the bar")

  const firstRow = (wiredLimitGb: number) => text(buildSections(input({ wiredLimitGb }), ["memory"])[0]?.rows ?? [])[0] ?? ""
  assert.match(firstRow(77), /95% · of 77.0G wired/)
  assert.match(firstRow(80), /92% · of 80.0G wired/)

  const without = buildSections(input(), ["memory"])[0]
  assert.equal(without.note, undefined, "no declaration, no bar")
  assert.equal(text(without.rows)[0], "footprint 73.3G", "bytes alone, exactly as before")
})

test("the wired bar needs a ceiling and a feed, and reads on without one", () => {
  const firstRow = (part: Partial<PanelInput>) => text(buildSections(input(part), ["memory"])[0]?.rows ?? [])[0] ?? ""
  assert.equal(firstRow({}), "footprint 73.3G", "nothing declared: plain bytes")
  assert.equal(firstRow({ wiredLimitGb: 0 }), "footprint 73.3G", "zero is not a ceiling")
  assert.equal(firstRow({ wiredLimitGb: 117.1875, ratioCells: 0 }), "63% · of 117G wired", "cells 0 drops the bar but keeps the reading, with no stray space")
  assert.equal(firstRow({ service: null, wiredLimitGb: 117 }), "", "no feed, no section to hold the bar")
})
test("acceptance keeps the gauge, the percent, and only room for one note", () => {
  const withBar = text(buildSections(input({ ratioCells: 8 }), ["spec"])[0]?.rows ?? [])
  assert.equal(withBar[0], "accept ▮▮▮▮▮░░░ 67.7% · 191/282", "bar and percent lead; the counts shrink to fit")
  assert.equal([...withBar[0]].length <= 34, true, "the percent must survive a 34-column sidebar")
  const noBar = text(buildSections(input({ ratioCells: 0 }), ["spec"])[0]?.rows ?? [])
  assert.equal(noBar[0], "accept 67.7% · 191/282 drafts", "with no bar the note spells out what it counts")
})

test("spec rows report acceptance, per-round yield and round cost", () => {
  assert.deepEqual(section("spec"), [
    "accept 67.7% · 191/282 drafts",
    "per round 1.14 tok · 168 rounds",
    "round 47ms · sync 2.93ms",
  ])
  assert.equal(section("spec").some((r) => /^spec /.test(r)), false, "no mode-and-age line: Server already names the decoder")
})

test("spec rows name the runtime gate when it goes off", () => {
  const rows = section("spec", input({ spec: observed(parseSpecStats(GATED_LINE)!) }))
  assert.deepEqual(rows, [
    "accept 38.4% · 89/232 drafts",
    "per round 1.62 tok · 55 rounds",
    "round 35ms · sync 3.60ms",
    "gate off · adaptive → serial",
  ], "speculation stopped mid-request: the headline fact")
})

test("spec modes without a fixed depth report tokens per round", () => {
  const pld = parseSpecStats("[spec-stats] mode=pld attempts=410 accepts=612 avg_per_round=1.49 runtime_disabled=false")!
  assert.deepEqual(section("spec", input({ spec: observed(pld) })), ["accept 1.49 tok/round · 612/410 rounds"])
})

test("a stale spec line just shows its numbers", () => {
  const old = observed(parseSpecStats(MTP_LINE)!, NOW - 20 * 60_000)
  assert.deepEqual(section("spec", input({ spec: old })), [
    "accept 67.7% · 191/282 drafts",
    "per round 1.14 tok · 168 rounds",
    "round 47ms · sync 2.93ms",
  ], "a quiet server is not stale data, and the tail already gates dead runs")
})

// --- sampling --------------------------------------------------------------

test("the model card and the sampling rows share a section", () => {
  assert.deepEqual(section("sampling"), [
    "model Qwen3.8-Flash-Next",
    "kv-quant 8-bit",
    "context 1,048,576",
    "spec mtp head · qwen4_exp",
    "temp 1.00 · p 0.95 k 20",
    "max out 64000 · launch default",
  ])
})

test("sampling rows note the route only when it is not chat completions", () => {
  const responses = observed(parseSampling(RESPONSES_LINE)!)
  assert.deepEqual(section("sampling", input({ sampling: responses })), [
    "model Qwen3.8-Flash-Next",
    "kv-quant 8-bit",
    "context 1,048,576",
    "spec mtp head · qwen4_exp",
    "temp 0.70",
    "max out 4096",
    "route responses",
  ])
})

test("the model card holds the section when the log has no request in it", () => {
  assert.deepEqual(section("sampling", input({ sampling: null })), [
    "model Qwen3.8-Flash-Next",
    "kv-quant 8-bit",
    "context 1,048,576",
    "spec mtp head · qwen4_exp",
  ])
  assert.deepEqual(section("sampling", input({ sampling: null, model: null })), [], "neither card nor request: nothing to draw")
  const nan = observed(parseSampling(CHAT_LINE.replace("temp=1.00", "temp=nan"))!)
  assert.equal(text(samplingRowsOf(nan)).find((l) => l.startsWith("temp ")), "temp unknown · p 0.95 k 20")
})

function samplingRowsOf(s: Observed<SamplingStats> | null): readonly SidebarRow[] {
  return buildSections(input({ sampling: s }), ["sampling"])[0]?.rows ?? []
}

// --- log section -----------------------------------------------------------

test("log rows point at the file and its freshness", () => {
  assert.deepEqual(logRowsOf(logStatus()), [
    "log mlx-serve-11234.log · 302K",
    "last write just now",
  ])
  assert.deepEqual(logRowsOf(logStatus({ error: "no log file", bytes: null, mtimeMs: null })), [
    "log mlx-serve-11234.log · no log file",
  ], "the name still shows, so the user knows which file is missing")
  assert.deepEqual(logRowsOf(logStatus({ mtimeMs: NOW - 3_600_000 })), [
    "log mlx-serve-11234.log · 302K",
    "last write 60m00s ago",
  ])
  assert.deepEqual(logRowsOf(logStatus({ dropped: 40 * 1024 })), [
    "log mlx-serve-11234.log · 302K",
    "last write just now",
    "tail +40K · unread",
  ])
  assert.deepEqual(logRowsOf(null), [
    "log not tailed · logPath off",
  ], "the feed row answers for the whole section when there is no file to point at")
  const down = buildSections(input({ log: null, link: "down", logDisabled: false }), ["log"])[0]
  assert.equal(down?.note, "· unreachable", "the feed state moved to the heading")
  assert.equal(down?.noteTone, "error", "a dark feed keeps its alert colour")
  assert.deepEqual(down?.rows[0], { label: "log", value: "not tailed", note: "· waiting for a server" })
})

test("the Server log heading names the feed state, dim when live", () => {
  const heading = (link: Link) => buildSections(input({ link }), ["log"])[0]
  assert.equal(heading("live")?.note, "· live")
  assert.equal(heading("live")?.noteTone, undefined, "all is well: gray, not green")
  assert.equal(heading("live")?.noteBright, undefined)
  assert.deepEqual([heading("down")?.note, heading("down")?.noteTone], ["· unreachable", "error"])
  assert.deepEqual([heading("disabled")?.note, heading("disabled")?.noteTone], ["· --metrics off", "warn"])
  assert.deepEqual([heading("unauthorized")?.note, heading("unauthorized")?.noteTone], ["· 401 unauthorized", "error"])
  assert.equal(heading("unknown")?.note, "· connecting")
})

function logRowsOf(log: LogStatus | null): string[] {
  const rows = buildSections(input({ log }), ["log"])[0]?.rows ?? []
  return rows.map((r) => `${r.label} ${r.value}${r.note ? ` ${r.note}` : ""}`)
}

// --- assembly --------------------------------------------------------------

test("buildSections keeps the requested order and drops empty sections", () => {
  const input0 = input({ speed: speed(), sparkCells: 0 })
  const names = (i: PanelInput) => buildSections(i, resolveSections(["log", "turn", "cache", "nosuch"])).map((s) => s.name)
  assert.deepEqual(names(input0), ["log", "turn", "cache"], "order follows the user's list, not ours")
  assert.deepEqual(names({ ...input0, log: null, service: null }), ["log", "turn"], "the log section still names the feed when the server is dead")
})

test("every section has a title and every default section draws", () => {
  assert.deepEqual([...ALL_SECTIONS, "attach"], Object.keys(SECTION_TITLES))
  const full = buildSections(input({ speed: speed() }), ALL_SECTIONS)
  assert.deepEqual(full.map((s) => s.title), [...ALL_SECTIONS].map((n) => SECTION_TITLES[n]))
  assert.equal(full.length, ALL_SECTIONS.length, `missing sections: ${ALL_SECTIONS.filter((n) => !full.some((s) => s.name === n)).join(",")}`)
})

test("the KV cache tiers gauge themselves, without the entry count", () => {
  const hot = parseCacheTier(HOT_TIER_LINE)!
  const ssd = parseCacheTier(SSD_TIER_LINE)!
  assert.deepEqual(hot, { kind: "hot", residentMb: 9481.06, capMb: 28672 }, "only the two numbers the row draws are captured")
  assert.equal(ssd.capMb, null, "the SSD cap is a launch flag; no log line carries it")
  const rows = cacheRowsOf({ hot: obs(hot), ssd: obs(ssd), diskCacheGb: 100, ratioCells: 8 })
  assert.deepEqual(rows, [
    "tokens 62% · from cache",
    "requests 60% · had a hit",
    "hot \u25ae\u25ae\u25ae\u2591\u2591\u2591\u2591\u2591 33% · 9.3G/28.0G",
    "ssd \u25ae\u2591\u2591\u2591\u2591\u2591\u2591\u2591 11% · 10.7G/100G",
  ])
  assert.equal(rows.some((l) => l.includes("entries")), false, "the entry count is the first thing cut, so it is not drawn")
  for (const l of rows) assert.ok([...l].length <= 34, `${l} is ${[...l].length} cells`)
  for (const l of rows) assert.ok([...l].length <= 32, `${l} must fit the sidebar without clipping its bytes`)
  assert.equal(rows[2].includes("budget"), false, "naming the denominator cost the bytes their place; the README says what they are")
})

test("an undeclared SSD cap draws bytes and no gauge", () => {
  const ssd = parseCacheTier(SSD_TIER_LINE)!
  assert.deepEqual(cacheRowsOf({ ssd: obs(ssd), diskCacheGb: null, ratioCells: 10 }), [
    "tokens 62% · from cache",
    "requests 60% · had a hit",
    "ssd 10.7G",
  ], "no invented denominator, and no entry count to fill the space")
  assert.deepEqual(cacheRowsOf({ hot: obs(parseCacheTier("[hot-cache] resident=9481.06 MB (1/2 entries)")!), ratioCells: 10 }), [
    "tokens 62% · from cache",
    "requests 60% · had a hit",
    "hot 9.3G",
  ], "the cap-less hot-cache line variant reports bytes only, like an undeclared SSD")
})

test("no cache tiers in the log means no tier rows", () => {
  assert.deepEqual(cacheRowsOf({ ratioCells: 8 }), ["tokens 62% · from cache", "requests 60% · had a hit"])
  assert.deepEqual(cacheRowsOf({ hot: obs(parseCacheTier("[hot-cache] resident=0.00 / 28672.00 MB (0/1 entries)")!), ratioCells: 8 }), [
    "tokens 62% · from cache",
    "requests 60% · had a hit",
    "hot ░░░░░░░░ 0% · —/28.0G",
  ], "an empty tier holds its line at 0%")
})

test("zero cache hits read as 0%, not as missing rows", () => {
  const cold = new ServiceTracker()
  cold.sample(feed({ counters: { prefix_cache_hits_total: 0, prefix_cache_tokens_total: 0 } }), NOW)
  assert.deepEqual(section("cache", input({ service: cold.statsAt(NOW)! })), [
    "tokens 0% · from cache",
    "requests 0% · had a hit",
  ])
  const fresh = new ServiceTracker()
  fresh.sample(
    feed({
      counters: {
        prefix_cache_queries_total: 0,
        prefix_cache_hits_total: 0,
        prefix_cache_tokens_total: 0,
        prompt_tokens_total: 0,
      },
    }),
    NOW,
  )
  assert.deepEqual(section("cache", input({ service: fresh.statsAt(NOW)! })), [], "no queries at all is no data, not a 0%")
})


test("the attach section stays out of the panel while the plugin is healthy", () => {
  assert.deepEqual(buildSections(input({ speed: speed() }), ["attach"]), [])
  assert.deepEqual(buildSections(input({ speed: speed(), attachErrors: [] }), ["attach"]), [])
})

test("an integration that threw is named on screen instead of hidden", () => {
  const rows = buildSections(
    input({ attachErrors: [{ where: "slot", detail: "TypeError: ctx.ui.slot is not a function" }, { where: "store", detail: "denied" }] }),
    ["attach"],
  )
  assert.deepEqual(render(rows[0] ?? { rows: [] }), [
    "slot TypeError: ctx.ui.slot is not a f… · degraded",
    "store denied · degraded",
  ])
  assert.equal(clip("abcdefghij", 6), "abcde…")
  assert.equal(clip("short", 12), "short", "nothing to truncate")
})

test("two failures in the same place get two lines", () => {
  // A row label is the sidebar's render key: a repeat would merge two rows.
  const rows = buildSections(
    input({ attachErrors: [{ where: "ui.slot", detail: "first" }, { where: "ui.slot", detail: "second" }] }),
    ["attach"],
  )
  assert.deepEqual(render(rows[0] ?? { rows: [] }), ["ui.slot first · degraded", "ui.slot 2 second · degraded"])
})

test("rows fit a narrow sidebar", () => {
  const wide = input({ speed: speed(), sparkCells: 24 })
  for (const s of buildSections(wide, ALL_SECTIONS)) {
    for (const r of s.rows) {
      const line = `${r.label} ${r.value}${r.note ? ` ${r.note}` : ""}`
      assert.ok([...line].length <= 40, `${s.name}/${line} is ${[...line].length} cells wide`)
      assert.ok(r.label.length <= 11, `label ${r.label} crowds the value column`)
    }
  }
})

test("resolveSections is permissive about typos and strict about order", () => {
  assert.deepEqual(resolveSections(undefined), [...ALL_SECTIONS])
  assert.deepEqual(resolveSections(["log", "turn"]), ["log", "turn"])
  assert.deepEqual(resolveSections(["turn", "bogus"]), ["turn"])
  assert.deepEqual(resolveSections([]), [], "an explicit empty list draws only the header")
})

// --- regressions -----------------------------------------------------------

test("the Turn section draws the prefill bar buildSections was given cells for", () => {
  const prefilling = speed({
    phase: "prefill",
    prefillTokens: 12_400,
    prefillExpected: 48_000,
    prefillTps: 1_600,
    genTps: null,
    ttftMs: null,
  })
  assert.deepEqual(section("turn", input({ speed: prefilling, barCells: 18 })), [
    "prefill █████░░░░░░░░░░░░░ 12.4k/48.0k",
    "1600 t/s · ~22s left",
  ])
})

test("attach can be asked for, and shows up uninvited when something broke", () => {
  assert.deepEqual(resolveSections(["attach", "log"]), ["attach", "log"], "attach is a name a user may write")
  assert.equal(resolveSections(undefined).includes("attach"), false, "but it is not a default")
  const broken = input({ attachErrors: [{ where: "slot", detail: "denied" }] })
  const names = buildSections(broken, ["log"]).map((s) => s.name)
  assert.deepEqual(names.at(-1), "attach", "a failed integration names itself even when nobody listed it")
  assert.equal(buildSections(broken, ["attach", "log"]).filter((s) => s.name === "attach").length, 1, "and only once")
})

test("no tail yet is not the same as a tail turned off", () => {
  assert.equal(
    logRowsOf(null).at(-1),
    "log not tailed · logPath off",
    "logPath off, by configuration",
  )
  const waiting = buildSections(input({ log: null, logDisabled: false }), ["log"])[0]?.rows ?? []
  assert.equal(
    text(waiting).at(-1),
    "log not tailed · waiting for a server",
    "no server has answered, so no log file has been claimed",
  )
})
