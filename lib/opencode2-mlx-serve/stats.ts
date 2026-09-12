/**
 * Parsing and math for what mlx-serve publishes:
 *
 *   GET /metrics.json   counters + gauges + histograms (needs --metrics)
 *   GET /props          memory headroom, n-gram warm progress
 *   GET /v1/models      resident model
 *   the server log      `[spec-stats]`, per-request sampling, cache tier lines
 *                       (generate.zig documents `[spec-stats]` as a stable format)
 *
 * Pure: no fetch, no OpenTUI, so `node --test` covers it. `ServiceTracker`
 * computes rates over a trailing window and averages from histogram sums, the
 * same way mlx-serve's own web panel does. A stale sample yields `null`, not the
 * last number seen.
 */

import type { SpeedValue } from "./tracker.ts"

export type { SpeedValue }

// ---------------------------------------------------------------------------
// /metrics.json shapes
// ---------------------------------------------------------------------------

/** A Prometheus histogram reduced to what is used: observation count and sum. */
export interface HistogramJson {
  readonly count: number
  readonly sum: number
}

export interface WireCounters {
  prompt_tokens_total?: number
  /** Prompt tokens actually forwarded (excludes prefix-cache restores). */
  prefill_tokens_total?: number
  prefix_cache_tokens_total?: number
  generation_tokens_total?: number
  requests_success_total?: number
  requests_cancelled_total?: number
  prefix_cache_queries_total?: number
  prefix_cache_hits_total?: number
}

export interface WireGauges {
  requests_running?: number
  requests_waiting?: number
  gpu_utilization_pct?: number
  memory_mb?: number
  generation_tokens_live?: number
  prefill_tokens_live?: number
  /** Post-cache tail the running prefill will forward (0 when idle). */
  prefill_tokens_expected?: number
  requests_prefilling?: number
  mlx_active_bytes?: number
  mlx_cache_bytes?: number
  ane_int8_bytes?: number
  ane_layers?: number
  ngram_warm_bytes?: number
}

export interface FeedCounters {
  /** Billed prompt tokens (includes prefix-cache restores). */
  readonly promptTokens: number
  /** Prompt tokens actually forwarded through the model. */
  readonly prefillTokens: number
  readonly cacheTokens: number
  readonly genTokens: number
  readonly requestsOk: number
  readonly requestsCancelled: number
  readonly cacheQueries: number
  readonly cacheHits: number
}

export interface FeedGauges {
  readonly running: number
  readonly waiting: number
  readonly gpuPct: number
  readonly memMb: number
  readonly genLive: number
  readonly prefillLive: number
  readonly prefillExpected: number
  readonly prefilling: number
  readonly mlxActiveBytes: number
  readonly mlxCacheBytes: number
  readonly aneBytes: number
  readonly aneLayers: number
  readonly ngramBytes: number
}

export interface MetricsFeed {
  readonly counters: FeedCounters
  readonly gauges: FeedGauges
  readonly histograms: Readonly<Record<string, HistogramJson>>
}

/** What the server actually sends. Every field optional so an older build parses. */
export interface RawMetricsJson {
  counters?: WireCounters | null
  gauges?: WireGauges | null
  histograms?: Record<string, unknown> | null
}

function num(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : 0
}

function parseHistogram(value: unknown): HistogramJson | null {
  if (typeof value !== "object" || value === null) return null
  const raw = value as { count?: unknown; sum?: unknown }
  if (typeof raw.count !== "number" || !Number.isFinite(raw.count) || raw.count <= 0) return null
  return { count: raw.count, sum: num(raw.sum) }
}

export function parseFeed(json: RawMetricsJson | null | undefined): MetricsFeed {
  const c = json?.counters ?? {}
  const g = json?.gauges ?? {}
  const histograms: Record<string, HistogramJson> = {}
  for (const [key, value] of Object.entries(json?.histograms ?? {})) {
    const hist = parseHistogram(value)
    if (hist) histograms[key] = hist
  }
  return {
    counters: {
      promptTokens: num(c.prompt_tokens_total),
      prefillTokens: num(c.prefill_tokens_total),
      cacheTokens: num(c.prefix_cache_tokens_total),
      genTokens: num(c.generation_tokens_total),
      requestsOk: num(c.requests_success_total),
      requestsCancelled: num(c.requests_cancelled_total),
      cacheQueries: num(c.prefix_cache_queries_total),
      cacheHits: num(c.prefix_cache_hits_total),
    },
    gauges: {
      running: num(g.requests_running),
      waiting: num(g.requests_waiting),
      gpuPct: num(g.gpu_utilization_pct),
      memMb: num(g.memory_mb),
      genLive: num(g.generation_tokens_live),
      prefillLive: num(g.prefill_tokens_live),
      prefillExpected: num(g.prefill_tokens_expected),
      prefilling: num(g.requests_prefilling),
      mlxActiveBytes: num(g.mlx_active_bytes),
      mlxCacheBytes: num(g.mlx_cache_bytes),
      aneBytes: num(g.ane_int8_bytes),
      aneLayers: num(g.ane_layers),
      ngramBytes: num(g.ngram_warm_bytes),
    },
    histograms,
  }
}

// ---------------------------------------------------------------------------
// /props
// ---------------------------------------------------------------------------

export interface PropsMemory {
  readonly activeGb: number
  readonly peakGb: number
  /** Free SYSTEM RAM — the same number that gates a model load. */
  readonly freeGb: number
  readonly poolGb: number
}

export interface PropsSnapshot {
  readonly memory: PropsMemory | null
  /** 0..1 once the n-gram table's total is known; null when nothing is warming. */
  readonly ngramProgress: number | null
  readonly ngramBytes: number
}

export const EMPTY_PROPS: PropsSnapshot = { memory: null, ngramProgress: null, ngramBytes: 0 }

export interface RawPropsJson {
  memory?: Record<string, unknown> | null
  ngram_warm?: Record<string, unknown> | null
}

export function parseProps(json: RawPropsJson | null | undefined): PropsSnapshot {
  const raw = json?.memory
  let memory: PropsMemory | null = null
  if (typeof raw === "object" && raw !== null) {
    memory = {
      activeGb: toGb(num(raw.active_bytes)),
      peakGb: toGb(num(raw.peak_bytes)),
      freeGb: toGb(num(raw.available_bytes)),
      poolGb: toGb(num(raw.cache_bytes)),
    }
  }
  const warm = json?.ngram_warm
  let ngramProgress: number | null = null
  let ngramBytes = 0
  if (typeof warm === "object" && warm !== null) {
    const bytes = num((warm as { bytes?: unknown }).bytes)
    const total = num((warm as { total?: unknown }).total)
    ngramBytes = bytes
    if (total > 0) ngramProgress = Math.min(1, bytes / total)
  }
  return { memory, ngramProgress, ngramBytes }
}

// ---------------------------------------------------------------------------
// The server log: [spec-stats] and the per-request sampling line
// ---------------------------------------------------------------------------

/** One `[spec-stats]` line: the tally of the request that just finished. */
export interface SpecStats {
  readonly mode: string
  /** Speculative rounds (one verify forward each). */
  readonly attempts: number
  /** Drafted tokens accepted (excludes the always-committed first token). */
  readonly accepts: number
  /** accepts ÷ attempts. */
  readonly avgPerRound: number | null
  /** accepts ÷ drafts proposed, the metric vLLM calls "acceptance rate". */
  readonly perDraftPct: number | null
  readonly drafted: number | null
  /** The runtime gate turned speculation off mid-request. */
  readonly runtimeDisabled: boolean
  readonly reason: string | null
  readonly adaptive: string | null
  readonly syncMs: number | null
  readonly roundMs: number | null
}

/** The `POST /v1/...` line mlx-serve logs per request at info level. */
export interface SamplingStats {
  readonly endpoint: string
  readonly temperature: number | null
  readonly topP: number | null
  readonly topK: number | null
  /** Requested output cap, e.g. 64000. */
  readonly maxTokens: number | null
  /** Where that cap came from, e.g. `launch default`; null when the log did not say. */
  readonly maxTokensOrigin: string | null
  readonly stream: boolean | null
  readonly messages: number | null
  /**
   * Messages with role == "tool" in that request, i.e. how many tool results the
   * prompt carried. It grows by one per tool round trip, so it is cumulative
   * within a conversation and resets on a new one — there is no server-wide
   * tool-call counter in the feed, which is why the row cannot claim "since boot".
   */
  readonly toolMsgs: number | null
}

const SPEC_MARKER = "[spec-stats]"

function fieldValue(line: string, key: string): string | null {
  const at = line.indexOf(`${key}=`)
  if (at < 0) return null
  if (at > 0 && !/\s/.test(line[at - 1] as string)) return null // `top_k` must not match `k=`
  const rest = line.slice(at + key.length + 1)
  const end = rest.search(/[\s,]/)
  const value = end < 0 ? rest : rest.slice(0, end)
  return value.length > 0 ? value : null
}

function asFloat(value: string | null): number | null {
  if (value === null) return null
  const n = Number(value.replace(/%$/, ""))
  return Number.isFinite(n) ? n : null
}

function asInt(value: string | null): number | null {
  const n = asFloat(value)
  return n === null ? null : Math.round(n)
}

function asBool(value: string | null): boolean | null {
  if (value === "true") return true
  if (value === "false") return false
  return null
}

/** Parses `  [spec-stats] mode=mtp attempts=55 ...`; null for any other line. */
export function parseSpecStats(line: string): SpecStats | null {
  if (!line.includes(SPEC_MARKER)) return null
  const mode = fieldValue(line, "mode")
  if (!mode) return null
  const attempts = asInt(fieldValue(line, "attempts")) ?? 0
  return {
    mode,
    attempts,
    accepts: asInt(fieldValue(line, "accepts")) ?? 0,
    avgPerRound: asFloat(fieldValue(line, "avg_per_round")),
    perDraftPct: asFloat(fieldValue(line, "per_draft_pct")),
    drafted: asInt(fieldValue(line, "drafted")),
    runtimeDisabled: asBool(fieldValue(line, "runtime_disabled")) === true,
    reason: fieldValue(line, "reason"),
    adaptive: fieldValue(line, "adaptive"),
    syncMs: asFloat(fieldValue(line, "sync_ms")),
    roundMs: asFloat(fieldValue(line, "round_ms")),

  }
}

const REQUEST_LINE = /^POST (\/v1\/[a-z0-9/_-]+) \((.*)\)/

/** Parses the per-request `POST /v1/... (…, temp=…, top_p=…)` line. */
export function parseSampling(line: string): SamplingStats | null {
  const m = REQUEST_LINE.exec(line.trim())
  if (!m) return null
  const body = m[2] ?? ""
  if (!body.includes("temp=")) return null
  const max = /(?:max_tokens|max_out)=(\d+)(?:\s*\(([^)]*)\))?/.exec(body)
  const msgs = /^(\d+) msgs/.exec(body)
  const toolMsgs = /tool_msgs=(\d+)/.exec(body)
  return {
    endpoint: (m[1] as string).replace("/v1/", ""),
    temperature: asFloat(fieldValue(body, "temp")),
    topP: asFloat(fieldValue(body, "top_p")),
    topK: asInt(fieldValue(body, "top_k")),
    maxTokens: max?.[1] === undefined ? null : Number(max[1]),
    maxTokensOrigin: max?.[2] ? max[2].trim() : null,
    stream: asBool(fieldValue(body, "stream")),
    messages: msgs ? Number(msgs[1]) : null,
    toolMsgs: toolMsgs ? Number(toolMsgs[1]) : null,
  }
}

// ---------------------------------------------------------------------------
// /v1/models — which model is behind these numbers
// ---------------------------------------------------------------------------

export interface ModelStats {
  readonly id: string
  readonly shortId: string
  readonly architecture: string | null
  /** e.g. `4-bit`; mlx-serve reports the mixed build's dominant width. */
  readonly quantization: string | null
  readonly layers: number | null
  readonly contextLength: number | null
  readonly kvQuant: string | null
  /** The model's own MTP head (the speculative decoder `[spec-stats]` measures). */
  readonly mtpLoaded: boolean
  readonly drafterLoaded: boolean
  readonly state: string
}

interface RawModelMeta {
  architecture?: unknown
  quantization?: unknown
  num_layers?: unknown
  context_length?: unknown
  kv_quant?: unknown
  mtp_loaded?: unknown
  drafter_loaded?: unknown
}

interface RawModelEntry {
  id?: unknown
  loaded?: unknown
  state?: unknown
  context_length?: unknown
  meta?: RawModelMeta
}

/** The resident chat model of `GET /v1/models`; null when nothing is loaded. */
export function parseModels(json: unknown): ModelStats | null {
  const data = (json as { data?: RawModelEntry[] } | null)?.data
  if (!Array.isArray(data)) return null
  const entry = data.find((item) => item?.loaded === true) ?? data[0]
  if (!entry || typeof entry.id !== "string") return null
  const meta = entry.meta ?? {}
  const ctx = typeof meta.context_length === "number" ? meta.context_length : num(entry.context_length)
  return {
    id: entry.id,
    shortId: shortModelName(entry.id),
    architecture: typeof meta.architecture === "string" ? meta.architecture : null,
    quantization: typeof meta.quantization === "string" ? meta.quantization : null,
    layers: typeof meta.num_layers === "number" ? meta.num_layers : null,
    contextLength: ctx > 0 ? ctx : null,
    kvQuant: typeof meta.kv_quant === "string" && meta.kv_quant !== "" ? meta.kv_quant : null,
    mtpLoaded: meta.mtp_loaded === true,
    drafterLoaded: meta.drafter_loaded === true,
    state: typeof entry.state === "string" ? entry.state : "unknown",
  }
}

/** Drops the scaffolding server-built ids carry (`…-MLX-Serve-mixed-4-8bit`). */
export function shortModelName(id: string): string {
  let clean = id.trim()
  clean = clean.replace(/-?mlx[-_]?serve[-_]?/gi, "-")
  clean = clean.replace(/[-_](?:mixed|pure)(?=[-_]|$)/gi, "")
  clean = clean.replace(/[-_](?:\d+-)*\d+bit/gi, "")
  clean = clean.replace(/[-_]{2,}/g, "-").replace(/^[-_\s]+|[-_\s]+$/g, "")
  return clean.length > 26 ? `${clean.slice(0, 25).replace(/[-_\s]$/, "")}…` : clean
}

// ---------------------------------------------------------------------------
// The KV cache tiers, from the server log
// ---------------------------------------------------------------------------

/**
 * One KV-cache tier, read from the log lines the tiers print whenever they
 * insert or evict. Neither tier is published over HTTP — `/metrics.json` counts
 * prefix-cache *hits* (the RAM tier's business) and says nothing about bytes on
 * the SSD — so the log is the only place this is visible:
 *
 *   [hot-cache]  resident=9481.06 / 28672.00 MB (1/1 entries)
 *   [disk-cache] persisted 643681/643681 tokens (+1 chunks, 3 ssm-cp, 7.6 MB, 12ms); resident=10933.6 MB (12 entries)
 *
 * The hot tier's line carries its own cap, so its gauge is measured. The disk
 * tier's cap is a launch flag (`--prefix-cache-disk 100GB`) that appears in no
 * line, so `capGb` stays null unless the operator declares it in cli.json.
 */
export interface CacheTier {
  readonly kind: "hot" | "ssd"
  /** Bytes held by the tier, in the megabytes the server prints (÷1024 → GiB). */
  readonly residentMb: number
  readonly capMb: number | null
}

// The entry counts and per-write totals are matched past, not captured: nothing draws
// them, and a capture group nobody reads is code that looks like data.
const HOT_LINE = /\[hot-cache\] resident=([\d.]+)(?: \/ ([\d.]+))? MB \(\d+\/\d+ entries\)/
const SSD_LINE = /\[disk-cache\] persisted \d+\/\d+ tokens \([^)]*?\); resident=([\d.]+) MB \(\d+ entries\)/

function grp(m: RegExpExecArray | null, i: number): number | null {
  const v = m?.[i]
  if (v === undefined) return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

/** Parses one cache-tier line; null for anything else. */
export function parseCacheTier(line: string): CacheTier | null {
  if (line.includes("[hot-cache] resident=")) {
    const m = HOT_LINE.exec(line)
    if (!m) return null
    return { kind: "hot", residentMb: grp(m, 1) ?? 0, capMb: grp(m, 2) }
  }
  if (line.includes("[disk-cache] persisted ")) {
    const m = SSD_LINE.exec(line)
    if (!m) return null
    return { kind: "ssd", residentMb: grp(m, 1) ?? 0, capMb: null }
  }
  return null
}

// ---------------------------------------------------------------------------
// Derived service stats
// ---------------------------------------------------------------------------

/** Outcome of the last /metrics.json read. */
export type Link = "unknown" | "live" | "disabled" | "down" | "unauthorized"

export type ServerPhase = "idle" | "queued" | "prefill" | "decode"

export interface ServiceStats {
  readonly t: number
  readonly link: Link
  readonly phase: ServerPhase
  /** Live decode tok/s over a short window; null when nothing is decoding. */
  readonly genTps: number | null
  /** Live prefill tok/s over a wide window; null when nothing is prefilling. */
  readonly prefillTps: number | null
  /** Prompt tokens forwarded so far by the running prefill; 0 when idle. */
  readonly prefillLive: number
  /** Post-cache tail the running prefill will forward; 0 when idle or unreported. */
  readonly prefillExpected: number
  /** Cumulative prefill tok/s: forwarded tokens ÷ prefill time. */
  readonly avgPrefillTps: number | null
  /** Cumulative decode tok/s: generated tokens ÷ decode time. */
  readonly avgGenTps: number | null
  readonly reqPerSec: number | null
  readonly running: number
  readonly waiting: number
  readonly prefilling: number
  readonly gpuPct: number
  /** Process physical footprint (phys_footprint), GiB. */
  readonly memGb: number | null
  readonly mlxActiveGb: number | null
  readonly mlxPoolGb: number | null
  readonly freeRamGb: number | null
  readonly peakRamGb: number | null
  readonly cacheHitPct: number | null
  /** Share of billed prompt tokens served from the prefix cache. */
  readonly cacheTokenPct: number | null
  readonly promptTokens: number
  readonly genTokens: number
  readonly requestsOk: number
  readonly requestsCancelled: number
  readonly aneBytes: number
  readonly aneLayers: number
  /** n-gram table bytes read into the page cache (qwen4 background warm). */
  readonly ngramBytes: number
  readonly ngramProgress: number | null
  /** Decode tok/s, one point per second for the last 60 seconds. */
  readonly genSeries: readonly number[]
}

interface Sample {
  t: number
  genLive: number
  prefillLive: number
  genTotal: number
  requests: number
  running: number
  waiting: number
  prefilling: number
  gpuPct: number
  memMb: number
  activeBytes: number
  poolBytes: number
}

const RETAIN_MS = 120_000
const MAX_SAMPLES = 1_024
/** A sample older than this says nothing about "now" — live rates go null. */
const FRESH_MS = 6_000
const GEN_WINDOW_MS = 4_000
const REQ_WINDOW_MS = 60_000
/** Shortest span a rate may be divided by; below this a poll artifact reads as throughput. */
const MIN_WINDOW_MS = 250
const SERIES_SECONDS = 60

const GIB = 1024 ** 3

function toGb(bytes: number): number {
  return bytes > 0 ? bytes / GIB : 0
}

/** Newest sample at least `windowMs` older than `base`. */
function windowStart(samples: readonly Sample[], base: Sample, windowMs: number): Sample {
  let out = samples[0] as Sample
  for (const sample of samples) {
    if (base.t - sample.t >= windowMs) out = sample
    else break
  }
  return out
}

/** Rate over the trailing window, or null when the window says nothing moved. */
function windowRate(
  samples: readonly Sample[],
  base: Sample,
  windowMs: number,
  pick: (sample: Sample) => number,
): number | null {
  if (samples.length < 2) return null
  const from = windowStart(samples, base, windowMs)
  const dtMs = base.t - from.t
  if (dtMs < MIN_WINDOW_MS) return null
  const delta = pick(base) - pick(from)
  if (!(delta > 0)) return null // also rejects NaN from a feed missing the gauge
  const rate = delta / (dtMs / 1000)
  return Number.isFinite(rate) && rate > 0 ? rate : null
}

/**
 * One point per whole second, ending at `nowSecond`, oldest first. A second with
 * no bin is 0: the bins are only written while something runs, so a missing
 * second is idle, and repeating the last rate would draw a tool pause as decode.
 * Leading seconds before the oldest bin are left off rather than padded.
 */
function bucketSeries(bins: ReadonlyMap<number, number>, points: number, nowSecond: number): number[] {
  if (bins.size === 0) return []
  let first = Infinity
  for (const key of bins.keys()) first = Math.min(first, key)
  const start = Math.max(nowSecond - points + 1, first)
  const out: number[] = []
  for (let second = start; second <= nowSecond; second++) out.push(bins.get(second) ?? 0)
  return out
}

/** Slices a per-second series down to `cells` values, keeping each slice's peak. */
export function downsample(series: readonly number[], cells: number): number[] {
  if (series.length === 0 || cells <= 0) return []
  if (series.length <= cells) return series.slice()
  const out: number[] = []
  for (let i = 0; i < cells; i++) {
    const from = Math.floor((i * series.length) / cells)
    const to = Math.max(from + 1, Math.floor(((i + 1) * series.length) / cells))
    let max = 0
    for (let j = from; j < to && j < series.length; j++) max = Math.max(max, series[j] ?? 0)
    out.push(max)
  }
  return out
}

function ratio(numerator: number, denominator: number | null): number | null {
  if (denominator === null || denominator <= 0 || numerator <= 0) return null
  return numerator / denominator
}

function histSeconds(hist: HistogramJson | undefined): number | null {
  if (!hist || hist.count <= 0 || hist.sum <= 0) return null
  return hist.sum
}

// A zero numerator is 0%, not missing data: only a zero denominator, a
// negative part, or a non-number hides the row.
function percent(part: number, whole: number): number | null {
  if (!Number.isFinite(part) || !Number.isFinite(whole) || whole <= 0 || part < 0) return null
  return Math.min(100, Math.round((part / whole) * 100))
}

/**
 * Live prefill tok/s: forwarded tokens since the gauge last read 0, over the
 * time since then. `prefill_tokens_live` resets per request, so a trailing
 * window wider than the prefill divides by idle time and understates the rate.
 */
function prefillRate(base: Sample | null, last: Sample): number | null {
  if (base === null) return null
  const dtMs = last.t - base.t
  if (dtMs < MIN_WINDOW_MS) return null
  // A base that already carries tokens of THIS prefill is a real starting point;
  // one left over from a previous request counts as zero.
  const from = base.prefillLive > 0 && base.prefillLive <= last.prefillLive ? base.prefillLive : 0
  const delta = last.prefillLive - from
  if (!(delta > 0)) return null
  const rate = delta / (dtMs / 1000)
  return Number.isFinite(rate) && rate > 0 ? rate : null
}

export class ServiceTracker {
  private samples: Sample[] = []
  private latest: MetricsFeed | null = null
  private props: PropsSnapshot = EMPTY_PROPS
  private readonly genBins = new Map<number, number>()
  private link: Link = "unknown"
  /** The sample the current prefill started from; see `prefillRate`. */
  private prefillBase: Sample | null = null

  /** Called on every successful /metrics.json read. */
  sample(feed: MetricsFeed, now: number): void {
    this.latest = feed
    this.link = "live"

    const current: Sample = {
      t: now,
      // Older servers have no live gauge; fall back the way the web panel does.
      genLive: feed.gauges.genLive || feed.counters.genTokens,
      prefillLive: feed.gauges.prefillLive,
      genTotal: feed.counters.genTokens,
      requests: feed.counters.requestsOk,
      running: feed.gauges.running,
      waiting: feed.gauges.waiting,
      prefilling: feed.gauges.prefilling,
      gpuPct: feed.gauges.gpuPct,
      memMb: feed.gauges.memMb,
      activeBytes: feed.gauges.mlxActiveBytes,
      poolBytes: feed.gauges.mlxCacheBytes,
    }
    this.commit(current)
  }

  private commit(current: Sample): void {
    const previous = this.samples[this.samples.length - 1]
    // A server restart resets counters. Drop the history instead of letting a
    // negative delta divide by a long window and print itself as throughput.
    if (previous && (current.genTotal < previous.genTotal || current.requests < previous.requests)) {
      this.samples = []
      this.genBins.clear()
      this.prefillBase = null
    }
    // Where the current prefill started: the newest sample with the gauge at 0,
    // or the sample before it went backwards (a new request took the slot).
    if (current.prefillLive <= 0) this.prefillBase = current
    else if (this.prefillBase === null || (previous !== undefined && current.prefillLive < previous.prefillLive)) {
      this.prefillBase = previous ?? current
    }
    this.samples.push(current)
    while (this.samples.length > 2 && current.t - (this.samples[0] as Sample).t > RETAIN_MS) this.samples.shift()
    if (this.samples.length > MAX_SAMPLES) this.samples.shift()

    // Publish sparkline points while a phase is running, so an idle second reads
    // as 0 (a gap) rather than a repeat of the last rate.
    const second = Math.floor(current.t / 1000)
    const genTps = windowRate(this.samples, current, GEN_WINDOW_MS, (s) => s.genLive)
    if (current.running > 0) this.genBins.set(second, genTps ?? 0)
    this.trimBins(this.genBins, second)
  }

  /**
   * The outcome of the last /metrics.json read, readable even when no read has
   * ever succeeded — that is exactly when the panel needs to say why it is empty.
   */
  linkState(): Link {
    return this.link
  }

  noteProps(props: PropsSnapshot): void {
    this.props = props
  }

  /** Records a failed read. The next successful `sample` puts the link back to live. */
  noteLink(link: Link): void {
    if (link === "live" || link === "unknown") return
    this.link = link
  }

  private trimBins(bins: Map<number, number>, second: number): void {
    for (const key of bins.keys()) {
      if (key < second - SERIES_SECONDS) bins.delete(key)
    }
  }

  statsAt(now: number): ServiceStats | null {
    const feed = this.latest
    if (!feed) return null
    const samples = this.samples
    const last = samples[samples.length - 1]
    const fresh = last !== undefined && now - last.t <= FRESH_MS
    const h = feed.histograms

    const genTps = fresh && last && last.running > 0 ? windowRate(samples, last, GEN_WINDOW_MS, (s) => s.genLive) : null
    const prefillTps =
      fresh && last && last.prefilling > 0 && last.prefillLive > 0 ? prefillRate(this.prefillBase, last) : null
    const reqPerSec = fresh && last ? windowRate(samples, last, REQ_WINDOW_MS, (s) => s.requests) : null

    const phase: ServerPhase = !fresh || !last
      ? "idle"
      : last.prefilling > 0
        ? "prefill"
        : last.running > 0
          ? "decode"
          : last.waiting > 0
            ? "queued"
            : "idle"

    return {
      t: now,
      link: this.link,
      phase,
      genTps,
      prefillTps,
      // The server publishes how far the running prefill has got and how far it
      // will go; the pair is the Throughput section's real progress bar.
      prefillLive: fresh ? feed.gauges.prefillLive : 0,
      prefillExpected: fresh ? feed.gauges.prefillExpected : 0,
      // Forwarded tokens only: dividing BILLED prompt tokens by prefill time
      // overstates warm-cache prefill speed by prompt ÷ (prompt - cached).
      avgPrefillTps: ratio(feed.counters.prefillTokens, histSeconds(h.prefill_time_seconds)),
      avgGenTps: ratio(feed.counters.genTokens, histSeconds(h.decode_time_seconds)),
      reqPerSec,
      // Momentary gauges, gated on freshness like the rates: a sample nobody has
      // refreshed says nothing about what is in flight now.
      running: fresh ? (last?.running ?? 0) : 0,
      waiting: fresh ? (last?.waiting ?? 0) : 0,
      prefilling: fresh ? (last?.prefilling ?? 0) : 0,
      gpuPct: fresh ? (last?.gpuPct ?? feed.gauges.gpuPct) : 0,
      memGb: last && last.memMb > 0 ? last.memMb / 1024 : null,
      mlxActiveGb: toGb(last?.activeBytes ?? 0) || this.props.memory?.activeGb || null,
      mlxPoolGb: toGb(last?.poolBytes ?? 0) || this.props.memory?.poolGb || null,
      freeRamGb: this.props.memory?.freeGb ?? null,
      peakRamGb: this.props.memory?.peakGb ?? null,
      cacheHitPct: percent(feed.counters.cacheHits, feed.counters.cacheQueries),
      cacheTokenPct: percent(feed.counters.cacheTokens, feed.counters.promptTokens),
      promptTokens: feed.counters.promptTokens,
      genTokens: feed.counters.genTokens,
      requestsOk: feed.counters.requestsOk,
      requestsCancelled: feed.counters.requestsCancelled,
      aneBytes: feed.gauges.aneBytes,
      aneLayers: feed.gauges.aneLayers,
      ngramBytes: feed.gauges.ngramBytes || this.props.ngramBytes,
      ngramProgress: this.props.ngramProgress,
      genSeries: bucketSeries(this.genBins, SERIES_SECONDS, Math.floor(now / 1000)),
    }
  }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

export function fmtRate(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—"
  if (value >= 10_000) return `${(value / 1000).toFixed(1)}k`
  if (value >= 100) return String(Math.round(value))
  return value.toFixed(1)
}

export function fmtMs(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—"
  if (value < 1000) return `${Math.round(value)}ms`
  if (value < 10_000) return `${(value / 1000).toFixed(2)}s`
  return `${Math.round(value / 1000)}s`
}

/**
 * Milliseconds with the precision the small end deserves: a 2.93ms sync is
 * worth distinguishing from 3.4ms, while 24.6s does not need four digits.
 */
export function fmtMsFine(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—"
  if (value < 10) return `${value.toFixed(2)}ms`
  if (value < 1000) return `${Math.round(value)}ms`
  return fmtMs(value)
}

export function fmtGib(value: number | null): string {
  if (value === null || !Number.isFinite(value) || value <= 0) return "—"
  if (value >= 100) return `${Math.round(value)}G`
  return `${value.toFixed(1)}G`
}

/**
 * Exact grouped digits — `1,833`, `17,453` — the way the host's own context meter
 * writes token counts (`.toLocaleString()`). Used where a number is read for its
 * value rather than its magnitude; `fmtCount` (`1.8k`) stays for the narrow panel
 * columns.
 */
export function fmtExact(value: number): string {
  if (!Number.isFinite(value)) return "0"
  return Math.round(value).toLocaleString("en-US")
}

export function fmtCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`
  return String(Math.round(value))
}

export function fmtDur(ms: number): string {
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)}s`
  if (ms < 60_000) return `${Math.round(ms / 1000)}s`
  return `${Math.floor(ms / 60_000)}m${String(Math.round((ms % 60_000) / 1000)).padStart(2, "0")}s`
}

/**
 * The wired ceiling the memory gauge is drawn against. Only a declared limit
 * counts: mlx-serve compares against Metal's `max_recommended_working_set_size`,
 * which it never publishes, so with no declared limit there is no gauge.
 */
export function wiredCeilingGb(declaredGb: number | null): number | null {
  return declaredGb !== null && Number.isFinite(declaredGb) && declaredGb > 0 ? declaredGb : null
}

/** `iogpu.wired_limit_mb` is megabytes; the panel speaks GiB. */
export function mbToGb(mb: number | null | undefined): number | null {
  if (mb === null || mb === undefined || !Number.isFinite(mb) || mb <= 0) return null
  return mb / 1024
}

export function fmtBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)}G`
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)}M`
  if (n >= 1024) return `${Math.round(n / 1024)}K`
  return `${n}B`
}

const FILLED = "█"
const LEVEL = "▮"
const EMPTY = "░"

/**
 * Block progress bar, exactly `cells` wide so it can never wrap the column. A
 * null fraction draws an empty bar: "we do not know how far along we are" must
 * not look like a bar that moved and then stopped.
 */
export function progressBar(fraction: number | null, cells: number): string {
  if (cells <= 0) return ""
  const f = fraction === null || !Number.isFinite(fraction) ? 0 : Math.max(0, Math.min(1, fraction))
  const filled = Math.round(f * cells)
  return FILLED.repeat(filled) + EMPTY.repeat(cells - filled)
}

/**
 * Level gauge, `▮▮▮▮▮▮▮░░░`: a quantity against its own limit. Distinct from the
 * `█` progress bar, which fills toward a total over time.
 */
export function levelBar(fraction: number | null, cells: number): string {
  if (cells <= 0) return ""
  const f = fraction === null || !Number.isFinite(fraction) ? 0 : Math.max(0, Math.min(1, fraction))
  const filled = Math.round(f * cells)
  return LEVEL.repeat(filled) + EMPTY.repeat(cells - filled)
}

const SPARK_BLOCKS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"] as const

/** Compact terminal sparkline; a flat series renders as a solid bar. */
export function sparkline(values: readonly number[], cells: number): string {
  const data = downsample(values, cells)
  if (data.length === 0) return ""
  const max = Math.max(...data)
  if (max <= 0) return (SPARK_BLOCKS[0] as string).repeat(data.length)
  return data
    .map((value) => {
      const level = Math.round((value / max) * (SPARK_BLOCKS.length - 1))
      return SPARK_BLOCKS[Math.max(0, Math.min(SPARK_BLOCKS.length - 1, level))] as string
    })
    .join("")
}
