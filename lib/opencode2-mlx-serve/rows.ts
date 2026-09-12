/**
 * What the sidebar draws: one `SidebarRow` per statistic, grouped into sections.
 * Pure functions over the values `stats.ts` and `logtail.ts` produce, so the
 * wording is testable without a terminal.
 *
 * A row is `label value · note` (label and note dim, value bright). The sidebar
 * is narrow and truncates the end of a line, so rows aim for ~30 cells and drop
 * a note before wrapping.
 */

import {
  fmtBytes,
  fmtCount,
  fmtDur,
  fmtGib,
  fmtMs,
  fmtMsFine,
  fmtExact,
  fmtRate,
  levelBar,
  mbToGb,
  progressBar,
  sparkline,
  wiredCeilingGb,
  type ModelStats,
  type SamplingStats,
  type ServiceStats,
  type SpecStats,
  type CacheTier,
  type Link,
} from "./stats.ts"
import type { LogStatus, Observed } from "./logtail.ts"
import type { SpeedValue } from "./tracker.ts"

export interface SidebarRow {
  /**
   * Dimmed key. Also the sidebar's render key: a section's labels must be unique
   * and stay the same between repaints, or the line is torn down and built again
   * and everything under it jumps.
   */
  readonly label: string
  /** Colour for the value: a status worth colouring, a number that is not. */
  readonly tone?: Tone
  /** The statistic, in the default (bright) colour. */
  readonly value: string
  /** Dimmed trailing context, carrying its own `· ` separator. */
  readonly note?: string
  /** Draw the note in the value colour: a second statistic, not an aside. */
  readonly noteBright?: boolean
}

export interface SidebarSection {
  readonly name: SectionName
  readonly title: string
  /**
   * Drawn on the heading line behind the title, carrying its own `· `
   * separator like a row note. A dim aside unless `noteBright` says it is a
   * second statistic, or `noteTone` colours it as an alert.
   */
  readonly note?: string
  /** Draw the heading note in the value colour, e.g. the Memory gauge. */
  readonly noteBright?: boolean
  /** Alert colour for the heading note, e.g. a dark feed. Wins over `noteBright`. */
  readonly noteTone?: Tone
  readonly rows: readonly SidebarRow[]
}

export type SectionName =
  | "turn"
  | "throughput"
  | "server"
  | "cache"
  | "memory"
  | "spec"
  | "sampling"
  | "log"
  | "attach"

export const ALL_SECTIONS: readonly SectionName[] = [
  "turn",
  "throughput",
  "server",
  "cache",
  "memory",
  "spec",
  "sampling",
  "log",
]

/**
 * Names a user may write in `sections`. `attach` is outside `ALL_SECTIONS` — it
 * is about this plugin rather than the server, and `buildSections` splices it in
 * by itself when something broke — but asking for it explicitly is allowed.
 */
const KNOWN_SECTIONS: readonly SectionName[] = [...ALL_SECTIONS, "attach"]

export const SECTION_TITLES: Readonly<Record<SectionName, string>> = {
  turn: "Turn",
  throughput: "Throughput",
  server: "Server",
  cache: "Prefix cache",
  memory: "Memory",
  spec: "Speculative Decoding",
  sampling: "Model & sampling",
  log: "Server log",
  attach: "Attach",
}

/**
 * Sections a user asked for, in their order; unknown names are dropped. Anything
 * that is not a list at all yields `fallback` — the caller's default, which is
 * not the same thing as "everything".
 */
export function resolveSections(raw: unknown, fallback: readonly SectionName[] = ALL_SECTIONS): SectionName[] {
  if (!Array.isArray(raw)) return [...fallback]
  const known = new Set<string>(KNOWN_SECTIONS)
  return raw.filter((name): name is SectionName => typeof name === "string" && known.has(name))
}

function row(label: string, value: string, note?: string, tone?: Tone): SidebarRow {
  const base: SidebarRow = note ? { label, value, note } : { label, value }
  return tone ? { ...base, tone } : base
}

/** ` 12m ago` when the newest log line of this kind predates our attention span. */
function ageNote(observed: Observed<unknown>, now: number): string | undefined {
  const age = Math.max(0, now - observed.at)
  if (age < 90_000) return undefined
  return `· ${fmtDur(age)} ago`
}

// ---------------------------------------------------------------------------
// Link status
// ---------------------------------------------------------------------------

export type Tone = "live" | "warn" | "error" | "muted"

/** How the last /metrics.json read went; drawn as the `feed` row in Server log. */
export function feedStatus(link: Link | undefined): { value: string; tone: Tone } {
  switch (link ?? "unknown") {
    case "live":
      return { value: "live", tone: "live" }
    case "disabled":
      return { value: "--metrics off", tone: "warn" }
    case "down":
      return { value: "unreachable", tone: "error" }
    case "unauthorized":
      return { value: "401 unauthorized", tone: "error" }
    default:
      return { value: "connecting", tone: "muted" }
  }
}

/**
 * The feed state as the `Server log` heading's aside — `Server log · live` —
 * instead of a row of its own. A heading is chrome, so the state reads dim
 * when all is well; a dark feed keeps the alert colour the row had.
 */
function logNote(link: Link | undefined): { note: string; noteTone?: Tone } {
  const status = feedStatus(link)
  switch (status.tone) {
    case "live":
    case "muted":
      return { note: `· ${status.value}` }
    default:
      return { note: `· ${status.value}`, noteTone: status.tone }
  }
}

/**
 * Live serving statistics: how busy the GPU is, what is in flight, and the
 * counters since boot. The model card (what is loaded) lives in Server log;
 * everything here moves while the server works, and no line here disappears.
 */
function serverRows(s: ServiceStats, sampling: SamplingStats | null, cells: number): SidebarRow[] {
  const pct = Math.max(0, Math.min(100, s.gpuPct))
  const waiting = `${s.waiting} waiting`
  const busy = s.prefilling > 0 ? ` · ${s.prefilling} prefilling` : ""
  return [
    row("gpu", `${gauge(pct / 100, cells)}${pct}%`),
    row("running", `${s.running} · ${waiting}${busy}`),
    ...totalsRows(s, sampling),
  ]
}

/**
 * The panel's gauge, `▮▮▮▮░░░░░░ ` with a trailing space, drawn as part of the
 * row's value so it takes the bright colour. Every percentage in the panel (GPU,
 * acceptance, cache tiers, wired ceiling) uses it. `█` is reserved for the
 * prefill progress bar and `▁▂▅` for sparklines.
 */
function gauge(fraction: number | null, cells: number): string {
  return cells > 0 ? `${levelBar(fraction, cells)} ` : ""
}

/** Safe divide: a missing or zero denominator yields an empty bar, never NaN. */
function ratio(part: number | null, whole: number | null): number | null {
  if (part === null || whole === null || whole <= 0) return null
  return Math.max(0, Math.min(1, part / whole))
}

// ---------------------------------------------------------------------------
// Turn: the per-session meter (drawn in the sidebar only if `turn` is in `sections`)
// ---------------------------------------------------------------------------

/**
 * Rows for the per-session speed meter, the same readout as the footer line,
 * drawn in the panel when `turn` is in `sections`. `inflight` is the server's
 * request count, used to label a rate measured from this session's own bytes.
 * `barCells` turns the prefill line into a progress bar; 0 keeps the plain rate.
 */
export function turnRows(speed: SpeedValue | null, inflight = 1, barCells = 0): SidebarRow[] {
  if (!speed) return []
  const approx = speed.tokensEstimated ? "~" : ""
  if (speed.phase === "prefill") {
    const bar = prefillBar(speed, barCells)
    if (bar !== null) return bar
    const rate = speed.prefillTps === null ? null : `${fmtRate(speed.prefillTps)} t/s`
    const tokens = speed.prefillTokens === null ? null : `${fmtCount(speed.prefillTokens)} tok`
    if (rate !== null && tokens !== null) return [row("prefill", rate, `· ${tokens}`)]
    if (rate !== null) return [row("prefill", rate)]
    if (tokens !== null) return [row("prefill", tokens, `· ${fmtDur(speed.elapsedMs)}`)]
    return [row("prefill", "waiting", `· ${fmtDur(speed.elapsedMs)}`)]
  }

  const rows: SidebarRow[] = []
  // With several clients decoding, this session's rate came from its own streamed
  // bytes rather than the server-wide counter; say so.
  const crowd = inflight > 1 ? ` · ${inflight} clients` : ""
  if (speed.genTps !== null) {
    rows.push(row("decode", `${approx}${fmtRate(speed.genTps)} t/s`, `· ${approx}${fmtCount(speed.genTokens)} tok${crowd}`))
  } else if (speed.genTokens > 0) {
    rows.push(row("decode", `${approx}${fmtCount(speed.genTokens)} tok`, "· settling"))
  }
  if (speed.prefillTps !== null) {
    const tokens = speed.prefillTokens === null ? "" : ` ${fmtCount(speed.prefillTokens)} tok`
    rows.push(row("prefill", `${fmtRate(speed.prefillTps)} t/s${tokens}`, "· last"))
  }
  if (speed.ttftMs !== null) {
    rows.push(row("ttft", fmtMs(speed.ttftMs), `· turn ${fmtDur(speed.elapsedMs)}`))
  }
  return rows
}

/**
 * `prefill ████████░░░░░░░░ 12.4k/48.0k` plus `1.6k t/s · ~22s left`.
 *
 * The denominator is the server's real target for the running prefill
 * (`prefill_tokens_expected`, the post-cache tail) — no estimate, no `~`.
 */
function prefillBar(speed: SpeedValue, cells: number): SidebarRow[] | null {
  if (cells <= 0) return null
  const live = speed.prefillTokens
  const base = speed.prefillExpected
  // No denominator, no bar: an empty bar would read as "0% of a known total".
  if (base === null || base <= 0 || live === null) return null
  const tps = speed.prefillTps ?? 0
  const remaining = base - live
  const left = tps > 0 && remaining > 0 ? Math.round(remaining / tps) : null
  const note = tps <= 0 ? "measuring" : left === null ? "· done" : `· ~${left}s left`
  return [
    row("prefill", `${progressBar(live / base, cells)} ${fmtCount(live)}/${fmtCount(base)}`),
    row("", note.startsWith("measuring") ? note : `${fmtRate(tps)} t/s ${note}`.trim()),
  ]
}

// ---------------------------------------------------------------------------
// Server stats sections
// ---------------------------------------------------------------------------

/** `~49.3 t/s`, `1311 t/s`, `12.3k t/s` → the number behind the string. */
function parseTps(value: string): number | null {
  const m = /^~?([\d.]+)(k)? t\/s$/.exec(value.trim())
  if (!m?.[1]) return null
  const n = Number(m[1])
  return Number.isFinite(n) ? n * (m[2] ? 1000 : 1) : null
}

/** Rates the Turn meter already put on screen, by row label. */
function turnRates(rows: readonly SidebarRow[]): Map<string, number> {
  const out = new Map<string, number>()
  for (const r of rows) {
    const n = parseTps(r.value)
    if (n !== null) out.set(r.label, n)
  }
  return out
}

/**
 * The turn meter and the server row window the same counter differently, so
 * "same number" is a 5% tolerance, not equality.
 */
function sameRate(rates: ReadonlyMap<string, number>, label: string, value: number | null): boolean {
  const other = rates.get(label)
  if (other === undefined || value === null) return false
  const span = Math.max(other, value)
  return span === 0 || Math.abs(other - value) / span < 0.05
}

/**
 * Server-wide throughput: `decode` and `prefill` on fixed lines, in that order,
 * plus the 60s trace and the admission rate. A rate that nothing is measuring
 * reads `0.0 t/s` with the since-boot average behind it — it never drops its
 * line. Every line below a removed one moves up, and the sidebar redraws the
 * whole panel when it does, so a finished prefill used to blink the panel twice:
 * once when the prefill line went away, once when it came back.
 *
 * While the server reports a running prefill with a known target, the `prefill`
 * line becomes the real progress bar (`prefill ███ 12.4k/48.0k · 4.2k t/s`);
 * the row keeps its label and its slot either way. With one request in flight
 * the server's live rate is this turn's rate, so when the Turn section already
 * drew that number the plain-rate form yields to the since-boot average.
 */
function throughputRows(
  s: ServiceStats,
  cells: number,
  barCells: number,
  already: ReadonlyMap<string, number>,
  inflight: number,
): SidebarRow[] {
  const rate = (label: string, live: number | null, avg: number | null): SidebarRow => {
    if (live === null) return row(label, `${fmtRate(0)} t/s`, `· avg ${fmtRate(avg)}`)
    // One request in flight: the Turn section already drew this rate, so the line
    // carries the since-boot average instead of a repeat. With two or more the
    // two are different measurements, and the server-wide one stays.
    if (inflight <= 1 && avg !== null && sameRate(already, label, live)) {
      return row(label, `${fmtRate(avg)} t/s`, "· since boot")
    }
    return row(label, `${fmtRate(live)} t/s`, `· avg ${fmtRate(avg)}`)
  }
  // The server publishes the running prefill's progress and its real target; the
  // pair is a real bar, so the prefill line draws it while one is running.
  const prefilling = barCells > 0 && s.prefillExpected > 0
  const prefill = prefilling
    ? row(
        "prefill",
        `${progressBar(s.prefillLive / s.prefillExpected, barCells)} ${fmtCount(s.prefillLive)}/${fmtCount(s.prefillExpected)}`,
        s.prefillTps === null ? "· measuring" : `· ${fmtRate(s.prefillTps)} t/s`,
      )
    : rate("prefill", s.prefillTps, s.avgPrefillTps)
  const rows = [rate("decode", s.genTps, s.avgGenTps), prefill]
  const spark = sparkline(s.genSeries, cells)
  if (spark !== "") rows.push(row("60s", spark))
  // Admissions per second over 60s. ~0.07 is one agent working; a jump is the
  // earliest sign of a second client. Idle is 0.00, not an absent line.
  rows.push(row("admitted", `${(s.reqPerSec ?? 0).toFixed(2)} req/s`))
  return rows
}

/**
 * Prefix cache: the two percentages come from `/metrics.json`; the two tiers come
 * from the log, because the feed counts hits and says nothing about bytes.
 *
 * `hot` prints its own cap in its log line, so its gauge is measured. `ssd`'s cap
 * is the `--prefix-cache-disk` launch flag, which appears nowhere, so it is
 * gauged only when `diskCacheGb` is declared; otherwise the row is bytes alone.
 */
function cacheRows(
  s: ServiceStats,
  hot: CacheTier | null,
  ssd: CacheTier | null,
  cells: number,
  diskCapGb: number | null,
): SidebarRow[] {
  const rows: SidebarRow[] = []
  if (s.cacheTokenPct !== null) rows.push(row("tokens", `${s.cacheTokenPct}%`, "· from cache"))
  if (s.cacheHitPct !== null) rows.push(row("requests", `${s.cacheHitPct}%`, "· had a hit"))
  // One tier against its own denominator. No entry count and no trailing word:
  // with a gauge the row is already ~31 cells and the sidebar cuts the end.
  const tier = (name: string, t: CacheTier | null, capGb: number | null): SidebarRow | null => {
    if (t === null) return null
    // A tier the log named holds its line even at zero: an empty cache is a
    // reading, and a row that comes and goes moves everything under it. Only a
    // corrupt number hides the row. `fmtGib` reads 0 GiB as "—".
    const rawMb = t.residentMb
    if (!Number.isFinite(rawMb) || rawMb < 0) return null
    const used = rawMb / 1024
    if (capGb === null || cells <= 0) return row(name, fmtGib(used))
    const fraction = ratio(used, capGb) ?? 0
    return { label: name, value: `${gauge(fraction, cells)}${Math.round(fraction * 100)}%`, note: `· ${fmtGib(used)}/${fmtGib(capGb)}` }
  }
  const capOf = (t: CacheTier | null): number | null => (t?.capMb == null ? null : mbToGb(t.capMb))
  // The hot cap is the server's computed budget (`ctx_kv + idle`), not the
  // `--prefix-cache-mem` flag, which is only the idle part. See README.
  const hotRow = tier("hot", hot, capOf(hot))
  const ssdRow = tier("ssd", ssd, diskCapGb)
  if (hotRow) rows.push(hotRow)
  if (ssdRow) rows.push(ssdRow)
  return rows
}

/**
 * Allocator and system memory. With a declared wired ceiling the footprint is
 * a bar row of its own, first in the section; without one it is a plain bytes
 * row. The bar sits on a row rather than the heading so the heading never
 * changes width while the numbers move.
 */
function memoryRows(s: ServiceStats, input: PanelInput): SidebarRow[] {
  const rows: SidebarRow[] = []
  const ceiling = wiredCeilingGb(input.wiredLimitGb ?? null)
  if (s.memGb !== null && ceiling !== null) {
    const fraction = ratio(s.memGb, ceiling) ?? 0
    const cells = input.ratioCells ?? 0
    const pct = Math.round(fraction * 100)
    // No label: the bar is the section's headline number. The ceiling it is
    // measured against rides dim behind it, so only the reading is bright.
    rows.push(row("", `${gauge(fraction, cells)}${pct}%`, `· of ${fmtGib(ceiling)} wired`))
  } else if (s.memGb !== null) {
    rows.push(row("footprint", `${s.memGb.toFixed(1)}G`))
  }
  // MLX's allocator holds bytes in use plus a reclaimable pool not yet returned.
  if (s.mlxActiveGb !== null) {
    rows.push(row("mlx-serve", fmtGib(s.mlxActiveGb), `· pool ${fmtGib(s.mlxPoolGb)}`))
  }
  if (s.freeRamGb !== null) {
    rows.push(row("free", fmtGib(s.freeRamGb), `· peak ${fmtGib(s.peakRamGb)}`))
  }
  if (s.aneBytes > 0) rows.push(row("ane", fmtGib(s.aneBytes / 1024 ** 3), `· ${s.aneLayers} layers`))
  if (s.ngramBytes > 0) {
    const size = fmtGib(s.ngramBytes / 1024 ** 3)
    if (s.ngramProgress === null) rows.push(row("ngram warm", size))
    else if (s.ngramProgress >= 1) rows.push(row("ngram", size))
    else rows.push(row("ngram", `${Math.round(s.ngramProgress * 100)}%`, `· ${size} read`))
  }
  return rows
}

/** The counts after the acceptance percent: terse beside a gauge, spelled out without one. */
function acceptNote(s: SpecStats, drafted: number, cells: number): string {
  if (drafted <= 0) return cells > 0 ? `· ${fmtCount(s.accepts)} tok` : `· ${fmtCount(s.accepts)} accepted tok`
  return cells > 0 ? `· ${fmtCount(s.accepts)}/${fmtCount(drafted)}` : `· ${fmtCount(s.accepts)}/${fmtCount(drafted)} drafts`
}

function specRows(observed: Observed<SpecStats> | null, now: number, cells: number): SidebarRow[] {
  const s = observed?.value
  if (!s || !observed) return []
  const rows: SidebarRow[] = []
  if (s.perDraftPct !== null) {
    const drafted = s.drafted ?? 0
    rows.push(
      row(
        "accept",
        `${gauge(s.perDraftPct / 100, cells)}${fmtRate(s.perDraftPct)}%`,
        acceptNote(s, drafted, cells),
      ),
    )
  } else if (s.avgPerRound !== null) {
    // PLD and dspark vary their draft width per round, so there is no fixed
    // denominator for a percentage; tokens per round is the unit.
    rows.push(
      row("accept", `${s.avgPerRound.toFixed(2)} tok/round`, `· ${fmtCount(s.accepts)}/${fmtCount(s.attempts)} rounds`),
    )
  }
  if (s.avgPerRound !== null && s.perDraftPct !== null) {
    rows.push(row("per round", `${s.avgPerRound.toFixed(2)} tok`, `· ${fmtCount(s.attempts)} rounds`))
  }
  if (s.roundMs !== null || s.syncMs !== null) {
    // Sub-10ms precision matters here (a 2.9ms sync vs a 9ms one).
    const round = s.roundMs === null ? "—" : fmtMsFine(s.roundMs)
    const sync = s.syncMs === null ? "—" : fmtMsFine(s.syncMs)
    rows.push(row("round", round, `· sync ${sync}`))
  }
  if (s.runtimeDisabled) {
    rows.push(row("gate", "off", `· ${s.reason ?? "adaptive"} → ${s.adaptive ?? "serial"}`, "warn"))
  }
  return rows
}

/**
 * What model this is, how it runs, and how the last request sampled: the model
 * card first, then what the last request actually ran with. mlx-serve logs
 * sampling per request, so this is what the server did, not what config says
 * it should do.
 */
function modelSamplingRows(model: ModelStats | null, observed: Observed<SamplingStats> | null, now: number): SidebarRow[] {
  return [...identityRows(model), ...samplingRows(observed, now)]
}

/**
 * The sampling the last request actually ran with. mlx-serve logs it per
 * request, so this is what the server did, not what config says it should do.
 */
function samplingRows(observed: Observed<SamplingStats> | null, now: number): SidebarRow[] {
  const v = observed?.value
  if (!v) return []
  const rows: SidebarRow[] = []
  const temp = v.temperature === null ? null : v.temperature.toFixed(2)
  const topP = v.topP === null ? null : v.topP.toFixed(2)
  if (temp === null && topP === null && v.topK === null) return []

  const tail: string[] = []
  if (topP !== null) tail.push(`p ${topP}`)
  if (v.topK !== null) tail.push(`k ${v.topK}`)
  rows.push(row("temp", temp ?? "unknown", tail.length > 0 ? `· ${tail.join(" ")}` : undefined))

  if (v.maxTokens !== null) {
    rows.push(row("max out", String(v.maxTokens), v.maxTokensOrigin === null ? undefined : `· ${v.maxTokensOrigin}`))
  }
  // A request that did not stream is worth naming; the message count rides in
  // Server beside the token totals.
  if (v.stream === false) rows.push(row("stream", "off"))
  if (v.endpoint !== "chat/completions") {
    rows.push(row("route", v.endpoint, ageNote(observed, now)))
  }
  return rows
}

/** Since-boot totals and the last request's shape, drawn inside `Server`. */
function totalsRows(s: ServiceStats, sampling: SamplingStats | null): SidebarRow[] {
  const rows: SidebarRow[] = [
    // "in" and "out" are both statistics, so both take the value colour.
    { label: "tokens", value: `${fmtCount(s.promptTokens)} in`, note: `· ${fmtCount(s.genTokens)} out`, noteBright: true },
    {
      label: "requests",
      value: `${fmtCount(s.requestsOk)} ok`,
      note: `· ${s.requestsCancelled} cancelled`,
      noteBright: s.requestsCancelled > 0,
    },
  ]
  const msgs = sampling?.messages
  if (msgs !== null && msgs !== undefined) {
    // Message count of the last prompt, in exact digits.
    rows.splice(2, 0, { label: "messages", value: fmtExact(msgs) })
  }
  const toolMsgs = sampling?.toolMsgs
  if (toolMsgs !== null && toolMsgs !== undefined) {
    // Per conversation, not since boot: `tool_msgs` counts the role=="tool"
    // messages the last prompt carried; the feed has no tool-call counter.
    rows.push({ label: "tool calls", value: fmtCount(toolMsgs) })
  }
  return rows
}

/**
 * What is loaded: the model card that used to lead Server. It describes the
 * run rather than the work, so it leads the sampling section — what model this
 * is, then how it sampled. Weight quantization is
 * left out (it describes the checkpoint, not the server); KV quantization is
 * kept because it sets the memory cost per token.
 */
function identityRows(model: ModelStats | null): SidebarRow[] {
  if (model === null) return []
  const rows = [row("model", model.shortId)]
  if (model.kvQuant !== null) rows.push(row("kv-quant", `${model.kvQuant}-bit`))
  if (model.contextLength !== null) rows.push(row("context", fmtExact(model.contextLength)))
  if (model.mtpLoaded || model.drafterLoaded) {
    const which = model.mtpLoaded ? "mtp head" : "drafter"
    // The architecture rides on this row, not `model`, so it cannot truncate
    // the model name. Dropped when the row would overflow.
    const notes = [model.mtpLoaded && model.drafterLoaded ? "+ drafter" : null, model.architecture].filter(
      (n): n is string => n !== null,
    )
    let note = notes.length > 0 ? `· ${notes.join(" · ")}` : undefined
    if (note !== undefined && 4 + which.length + note.length > 34) note = undefined
    rows.push(row("spec", which, note))
  }
  return rows
}

/**
 * The panel's last section: whether the server answers, and which log file is
 * tailed.
 */
function logRows(input: PanelInput): SidebarRow[] {
  const log = input.log
  // No file: either the operator turned the tail off, or no server has answered
  // yet and there is nothing to say which of the log directory's files is this run's.
  if (!log) {
    return [row("log", "not tailed", input.logDisabled === false ? "· waiting for a server" : "· logPath off")]
  }

  if (log.error !== null || log.bytes === null) {
    return [row("log", log.name, `· ${log.error ?? "unreadable"}`, "warn")]
  }
  const age = log.mtimeMs === null ? null : Math.max(0, input.now - log.mtimeMs)
  const rows: SidebarRow[] = [row("log", log.name, `· ${fmtBytes(log.bytes)}`)]
  if (age !== null) rows.push(row("last write", age < 5_000 ? "just now" : `${fmtDur(age)} ago`))
  if (log.dropped > 0) rows.push(row("tail", `+${fmtBytes(log.dropped)}`, "· unread"))
  return rows
}

/** The plugin's own health: which host integration threw. Drawn whenever one did. */
function attachRows(input: PanelInput): SidebarRow[] {
  const errors = input.attachErrors ?? []
  const seen = new Map<string, number>()
  // Labels are render keys, so a second failure in the same place is numbered
  // rather than sharing a line with the first.
  return errors.slice(0, 4).map((failure) => {
    const n = (seen.get(failure.where) ?? 0) + 1
    seen.set(failure.where, n)
    const label = n === 1 ? failure.where : `${failure.where} ${n}`
    return row(label, clip(failure.detail, 34), "· degraded")
  })
}

/** Truncated to fit the column, with the cut made visible. */
export function clip(text: string, cells: number): string {
  const chars = [...text]
  return chars.length <= cells ? text : `${chars.slice(0, cells - 1).join("")}…`
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

export interface PanelInput {
  /** This session's turn meter, from `SpeedTracker`. */
  readonly speed: SpeedValue | null
  readonly service: ServiceStats | null
  readonly model: ModelStats | null
  readonly spec: Observed<SpecStats> | null
  readonly sampling: Observed<SamplingStats> | null
  readonly log: LogStatus | null
  /** Sparkline width in cells; 0 turns the sparkline off. */
  readonly sparkCells: number
  readonly now: number
  /** Filled in by `buildSections`: rates the Turn section already drew. */
  readonly turnRates?: ReadonlyMap<string, number>
  /** Width of the prefill progress bar in cells; 0 draws the plain rate instead. */
  readonly barCells?: number
  /** The tail is off by configuration (`logPath: "off"`), rather than not attached yet. */
  readonly logDisabled?: boolean
  /** Host integrations that threw (a refused slot, a missing API). */
  readonly attachErrors?: readonly { where: string; detail: string }[]
  /** The declared wired ceiling (cli.json, or `iogpu.wired_limit_mb`), in GiB. */
  readonly wiredLimitGb?: number | null
  /** Width of the ratio bars in cells; 0 draws percentages only. */
  readonly ratioCells?: number
  /** KV-cache tiers (newest line of each) and the declared SSD cap in GiB. */
  readonly hot?: Observed<CacheTier> | null
  readonly ssd?: Observed<CacheTier> | null
  readonly diskCacheGb?: number | null
  /** Outcome of the last /metrics.json read. Defaults to what the snapshot
   * carries, so a caller cannot accidentally report "connecting" over live data. */
  readonly link?: Link
  /** Requests the server says are running; 1 or fewer means "this turn is alone".
   * Defaults to the feed's `requests_running` — tests override it. */
  readonly inflight?: number
}

/** How many clients were behind the shared counters when the feed was read. */
function inflightOf(input: PanelInput): number {
  return input.inflight ?? input.service?.running ?? 0
}

function rowsFor(name: SectionName, input: PanelInput): SidebarRow[] {
  const s = input.service
  const inflight = inflightOf(input)
  switch (name) {
    case "throughput":
      return s === null ? [] : throughputRows(s, input.sparkCells, input.barCells ?? 0, input.turnRates ?? new Map(), inflight)
    case "server":
      return s === null ? [] : serverRows(s, input.sampling?.value ?? null, input.ratioCells ?? 0)
    case "cache":
      return s === null ? [] : cacheRows(s, input.hot?.value ?? null, input.ssd?.value ?? null, input.ratioCells ?? 0, input.diskCacheGb ?? null)
    case "memory":
      return s === null ? [] : memoryRows(s, input)
    case "spec":
      return specRows(input.spec, input.now, input.ratioCells ?? 0)
    case "sampling":
      return modelSamplingRows(input.model, input.sampling, input.now)
    case "log":
      return logRows(input)
    case "attach":
      return attachRows(input)
    default:
      return []
  }
}

/**
 * Sections in the order asked for, with the empty ones dropped. The Turn
 * section is drawn first regardless of order so the sections below it can tell
 * which numbers are already on screen.
 */
export function buildSections(input: PanelInput, enabled: readonly SectionName[]): SidebarSection[] {
  const turn = enabled.includes("turn") ? turnRows(input.speed, inflightOf(input), input.barCells ?? 0) : []
  // Derived once for every section: the rates the turn meter already drew (so
  // Throughput does not repeat one) and the link state (so the panel is never
  // silently empty).
  const derived: PanelInput = {
    ...input,
    link: input.link ?? input.service?.link ?? "unknown",
    turnRates: turnRates(turn),
  }
  const sections: SidebarSection[] = []
  for (const name of enabled) {
    const rows = name === "turn" ? turn : rowsFor(name, derived)
    if (rows.length === 0) continue
    const note = name === "log" ? logNote(derived.link) : null
    sections.push({
      name,
      title: SECTION_TITLES[name],
      ...(note === null ? {} : note),
      rows,
    })
  }
  // The panel is where a broken host integration is named, so the section shows
  // up whether or not anybody listed it.
  if (!enabled.includes("attach") && (input.attachErrors?.length ?? 0) > 0) {
    const rows = attachRows(derived)
    if (rows.length > 0) sections.push({ name: "attach", title: SECTION_TITLES.attach, rows })
  }
  return sections
}

// ---------------------------------------------------------------------------
// Footer meter (the always-on line under the prompt)
// ---------------------------------------------------------------------------

/**
 * The footer's one line of turn stats, in a fixed handful of shapes — a
 * prefill bar, a prefill rate, a prefill wait, or a decode line — so the line
 * never breathes while the numbers move. Every slot draws in every shape,
 * zero-filled: nothing pops in or out mid-turn. ttft, age and history live in
 * the panel, not here.
 *
 * Without mlx-serve metrics (any other provider), the stream is the meter:
 * the prefill shape counts the wait up, and the decode shape shows the TTFT
 * the provider took instead of a prefill rate it never reported.
 */
export function footerLabel(speed: SpeedValue | null, options: FooterOptions = {}): string | null {
  if (!speed) return null
  if (speed.phase === "prefill") {
    const base = speed.prefillExpected
    if ((options.barCells ?? 0) > 0 && base !== null && base > 0) return prefillBarLine(speed, base, options.barCells ?? 0)
    if (speed.prefillTokens !== null || speed.prefillTps !== null) return prefillRateLine(speed)
    // Nothing measured yet — and for a non-mlx-serve server, nothing ever will
    // be: count the wait up. The API answers with TTFT when it starts.
    return `prefill waiting · ${fmtDur(speed.elapsedMs)}`
  }

  const approx = speed.tokensEstimated ? "~" : ""
  // The third slot is the prefill this decode came out of: its rate when mlx-
  // serve measured one, the TTFT the provider took when it did not.
  const prefill =
    speed.prefillTps !== null
      ? `prefill ${fmtRate(speed.prefillTps)} t/s`
      : speed.ttftMs !== null
        ? `ttft ${fmtMs(speed.ttftMs)}`
        : `prefill ${fmtRate(0)} t/s`
  return `${approx}${fmtExact(speed.genTokens)} tok · decode ${approx}${fmtRate(speed.genTps ?? 0)} t/s · ${prefill}`
}

export interface FooterOptions {
  /** Prefill progress-bar width in cells; 0 keeps the plain rate line. */
  readonly barCells?: number
}

/** Prefill with a real target: bar, live count, rate and eta — every slot always. */
function prefillBarLine(speed: SpeedValue, base: number, cells: number): string {
  const live = speed.prefillTokens ?? 0
  const tps = speed.prefillTps ?? 0
  // A measured rate of 0 divides nothing: it reads as still measuring, not as
  // an infinite eta (remaining / 0).
  const eta =
    tps > 0
      ? base - live > 0
        ? `~${Math.max(1, Math.round((base - live) / tps))}s left`
        : "done"
      : "measuring"
  return `prefill ${progressBar(live / base, cells)} ${fmtExact(live)}/${fmtExact(base)} · ${fmtRate(tps)} t/s · ${eta}`
}

/** Prefill without a reported target: tokens and rate, zero-filled like the bar. */
function prefillRateLine(speed: SpeedValue): string {
  const live = speed.prefillTokens ?? 0
  return `prefill ${fmtExact(live)} tok · ${fmtRate(speed.prefillTps ?? 0)} t/s`
}
