/** @jsxImportSource @opentui/solid */
/**
 * mlx-serve stats in the OpenCode sidebar, plus the turn speed meter in the
 * prompt footer.
 *
 *   footer   — turn stats, always on: `~456 tok · decode ~24.0 t/s`, and a
 *              progress bar while a prompt is being prefilled. Sessions on
 *              another provider are metered from their own stream (wait, then
 *              TTFT, then streamed bytes) instead of this server's feed.
 *   sidebar  — server throughput and serving totals, model card and sampling,
 *              prefix cache, memory, MTP acceptance, server log.
 *
 * No slash commands: this OpenCode build does not dispatch CLI-plugin commands
 * into the prompt, so everything is always-on or a `cli.json` option.
 *
 * Read-only: three GETs (/metrics.json, /props, /v1/models) and a tail of the
 * log file mlx-serve already writes.
 */
import { createEffect, createMemo, createSignal, For, onCleanup, Show } from "solid-js"
import { parseMetricsJson, SpeedTracker, type MetricsJson, type SpeedValue } from "./tracker.ts"
import {
  mbToGb,
  parseFeed,
  parseModels,
  parseProps,
  ServiceTracker,
  type Link,
  type ModelStats,
  type RawMetricsJson,
  type RawPropsJson,
} from "./stats.ts"
import { LogTail, defaultLogPath, portFromUrl, type LogRead } from "./logtail.ts"
import {
  buildSections,
  footerLabel,
  type PanelInput,
  type SectionName,
  type SidebarSection,
} from "./rows.ts"
import { originOf, resolveOptions, type ServeOptions } from "./options.ts"
import {
  PROBE_TIMEOUT_MS,
  REPROBE_AFTER_MS,
  answered,
  candidatePorts,
  logIsStale,
  logsDir,
  metricsUrlForPort,
} from "./discover.ts"
import { holdsServerBusy, isBusy, isDue, isWanted, metricsEveryMs } from "./schedule.ts"
import { execFileSync } from "node:child_process"
import { readdirSync, statSync } from "node:fs"

// Do not import @opencode-ai/plugin at runtime: a v1 copy in
// ~/.config/opencode/node_modules shadows the package OpenCode 2 bundles. The
// host shapes below are written structurally and every call into them is
// wrapped, so an API change costs a section of the panel, not a dead TUI.
type AnyEvent = {
  id: string
  type: string
  created?: number
  data: Record<string, unknown>
}

type Ctx = {
  options?: Record<string, unknown>
  storage: {
    memory: (
      key: string,
      init: { initial: Record<string, unknown> },
    ) => [Record<string, unknown>, (fn: (d: Record<string, unknown>) => void) => void]
  }
  data: { on: (type: string, handler: (e: AnyEvent) => void) => () => void }
  ui: {
    slot: (spec: Record<string, unknown>) => () => void
    toast: { show: (spec: { title?: string; message: string; variant?: string; duration?: number }) => void }
  }
  theme: {
    text?: { subdued?: string; muted?: string; default?: string }
    feedback?: Record<string, { default?: string }>
  }
}

export type { ServeOptions, SectionName }

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

const definition = {
  id: "opencode2.mlx-serve",
  setup(ctx: Ctx) {
    const options = resolveOptions(ctx.options)

    /**
     * The wired ceiling the memory gauge is drawn against: `wiredLimitGb` if set,
     * otherwise `iogpu.wired_limit_mb`, read once at load (mlx-serve reads its own
     * limit once too). Metal's real working-set size is what the server compares
     * against and it is never published, so nothing is guessed: a missing sysctl
     * (stock Mac, Linux, Windows) yields null and the panel draws no gauge.
     */
    function declaredWiredGb(): number | null {
      if (options.wiredLimitGb !== null) return options.wiredLimitGb
      try {
        const out = execFileSync("sysctl", ["-n", "iogpu.wired_limit_mb"], {
          encoding: "utf8",
          timeout: 1_500,
          stdio: ["ignore", "pipe", "ignore"],
        })
        return mbToGb(Number(out.trim()))
      } catch {
        return null
      }
    }
    const wiredGb = declaredWiredGb()

    /** Host integrations that threw, named in the panel so a partial load is visible. */
    const attachErrors: { where: string; detail: string }[] = []
    const clipWhere = (name: string) => name.replace(/\(.*\)$/, "").slice(0, 12)

    function attempt<T>(what: string, fn: () => T): T | null {
      try {
        return fn()
      } catch (err) {
        const detail = err instanceof Error ? `${err.name}: ${err.message}` : String(err)
        attachErrors.push({ where: clipWhere(what), detail })
        try {
          console.error(`[mlx-serve] ${what} failed:`, err)
        } catch {
          // no stderr to echo to
        }
        return null
      }
    }

    // Only the newest plugin instance drives the UI: OpenCode reloads CLI
    // plugins on file change, and the old instance must not keep rendering.
    const guard = attempt("storage.memory(generation)", () =>
      ctx.storage.memory("generation", { initial: { active: 0 } }),
    ) as [{ active: number }, (fn: (d: { active: number }) => void) => void] | null
    const gen = guard?.[0] ?? { active: 0 }
    // With no store the local object is mutated instead, so the guard degrades to
    // "always ours" rather than to "never ours", which would render nothing at all.
    const setGen = guard?.[1] ?? ((fn: (d: { active: number }) => void) => fn(gen))
    const mine = Number(gen.active) + 1
    setGen((d) => {
      d.active = mine
    })
    const instanceIsOurs = () => gen.active === mine

    const tracker = new SpeedTracker({ bytesPerToken: options.bytesPerToken })
    const service = new ServiceTracker()
    // The feed URL in force: the configured one until a probe adopts another port.
    let feedUrl = options.metricsUrl
    let tail: LogTail | null = null
    let tailPath: string | null = null
    let logStale = false

    /**
     * Tail the log of the port that answered, not of the port that was asked
     * for: ~/.mlx-serve/logs keeps a file per run, and a dead one reads as
     * current. An explicit `logPath` wins and never moves.
     */
    function ensureTail(url: string): void {
      const path = options.logPathExplicit ? options.logPath : defaultLogPath(portFromUrl(url))
      if (path === null) {
        tail = null
        tailPath = null
        return
      }
      if (tailPath === path) return
      tail = new LogTail(path)
      tailPath = path
      logRead = null
      logStale = false
      // Read it at once: the back-read is what fills the spec, sampling and
      // cache-tier rows, and the next slow tick may be seconds away.
      pollLog()
    }

    const [version, setVersion] = createSignal(0)
    const [panelOnScreen, setPanelOnScreen] = createSignal(false)
    let model: ModelStats | null = null
    let logRead: LogRead | null = null

    const activeSessions = new Set<string>()
    let lastSession: string | null = null

    /**
     * Which provider each session's current step is served by, from the step's
     * own model. A session we have not heard a step from stays eligible; the
     * first `session.step.started` settles it, and `beginStep` resets the run's
     * metered state either way. Only a known mismatch turns the local feed off
     * for that session, so a remote provider is metered from its own stream.
     */
    const sessionProviders = new Map<string, string | null>()
    function noteProvider(sid: string, stepModel: unknown): void {
      const providerID = (stepModel as { providerID?: unknown } | null | undefined)?.providerID
      sessionProviders.set(sid, typeof providerID === "string" ? providerID : null)
    }
    function metricsEligible(sid: string): boolean {
      if (options.provider === null) return true
      const provider = sessionProviders.get(sid)
      return provider === undefined || provider === null || options.provider.includes(provider)
    }

    // Server busyness is cached instead of recomputed eight times a second: the
    // tick only needs to know whether anything is still decoding.
    let serverBusyUntil = 0
    let metricsInFlight = false
    let propsInFlight = false
    let modelsInFlight = false
    let lastMetrics = 0
    let lastProps = 0
    let lastModels = 0
    let lastLog = 0
    /** When the feed first stopped answering; drives the port re-probe. */
    let linkDownSince = 0
    let probing = false
    let uiTimer: ReturnType<typeof setInterval> | undefined
    let reported = false
    const abort = new AbortController()

    const seenEventIDs = new Set<string>()
    function isNewEvent(e: AnyEvent): boolean {
      if (seenEventIDs.has(e.id)) return false
      seenEventIDs.add(e.id)
      if (seenEventIDs.size > 4_096) {
        const oldest = seenEventIDs.values().next().value
        if (oldest !== undefined) seenEventIDs.delete(oldest)
      }
      return true
    }

    const createdAt = (e: AnyEvent) => (typeof e.created === "number" && Number.isFinite(e.created) ? e.created : Date.now())

    function aborted(err: unknown): boolean {
      return (err as { name?: string } | undefined)?.name === "AbortError"
    }

    function authHeaders(): Record<string, string> {
      const headers: Record<string, string> = { Accept: "application/json" }
      if (options.metricsToken) headers.Authorization = `Bearer ${options.metricsToken}`
      return headers
    }

    async function get(url: string, timeoutMs = 2_000): Promise<{ ok: boolean; status: number; body: unknown }> {
      // A socket that is accepted and never answered would otherwise hold the
      // in-flight flag open and block every later poll.
      const signal = AbortSignal.any([abort.signal, AbortSignal.timeout(timeoutMs)])
      const res = await fetch(url, { headers: authHeaders(), signal })
      if (!res.ok) return { ok: false, status: res.status, body: null }
      return { ok: true, status: res.status, body: await res.json() }
    }

    /**
     * Which port is the live mlx-serve. Nothing on disk names it, so the newest
     * log files' ports are asked in turn and the first that answers is adopted;
     * 503 counts, because that is a server with --metrics off.
     */
    async function probeForFeed(): Promise<void> {
      if (probing) return
      probing = true
      try {
        const dir = logsDir()
        let files: { name: string; mtimeMs: number }[]
        try {
          files = readdirSync(dir).map((name) => {
            try {
              return { name, mtimeMs: statSync(`${dir}/${name}`).mtimeMs }
            } catch {
              return { name, mtimeMs: 0 }
            }
          })
        } catch {
          return // no log directory: nothing to guess from
        }
        for (const port of candidatePorts(files)) {
          const url = metricsUrlForPort(port)
          if (url === feedUrl) continue
          let status = 0
          try {
            const res = await fetch(url, {
              headers: authHeaders(),
              signal: AbortSignal.any([abort.signal, AbortSignal.timeout(PROBE_TIMEOUT_MS)]),
            })
            status = res.status
          } catch {
            continue
          }
          if (!answered(status)) continue
          feedUrl = url
          ensureTail(url)
          lastMetrics = 0
          lastProps = 0
          lastModels = 0
          return
        }
      } finally {
        probing = false
      }
    }

    /** A link that stays down is a server that moved, or never was: look again. */
    function noteLinkDown(now: number): void {
      if (linkDownSince === 0 || now - linkDownSince >= REPROBE_AFTER_MS) {
        linkDownSince = now
        void probeForFeed()
      }
    }

    /** Counters, gauges, histograms: most of the panel, and the turn meter's live feed. */
    async function pollMetrics(): Promise<void> {
      if (metricsInFlight) return
      metricsInFlight = true
      lastMetrics = Date.now()
      const url = feedUrl
      try {
        const res = await get(url)
        const now = Date.now()
        let link: Link = "live"
        if (res.status === 503) link = "disabled"
        else if (!res.ok) link = res.status === 401 || res.status === 403 ? "unauthorized" : "down"
        if (link === "live" || link === "disabled") {
          // This port is the server, so this port's log is the one to tail.
          linkDownSince = 0
          ensureTail(url)
        }
        if (link === "live") {
          service.sample(parseFeed(res.body as RawMetricsJson), now)
          const sample = parseMetricsJson(res.body as MetricsJson, now)
          for (const sessionID of activeSessions) {
            // A session on another provider cannot use this server's numbers:
            // its meter comes from its own stream instead.
            if (metricsEligible(sessionID)) tracker.applyMetrics(sessionID, sample)
          }
          const stats = service.statsAt(now)
          // The gauges are server-wide: another client's work is not a reason to
          // keep polling four times a second with nothing of ours to draw.
          const working = stats !== null && (stats.running > 0 || stats.prefilling > 0 || stats.waiting > 0)
          serverBusyUntil =
            working && holdsServerBusy(panelOnScreen(), tracker.hasActive(now)) ? now + 6_000 : now
        } else {
          service.noteLink(link)
          serverBusyUntil = now
          // 401 is a server that is there with the wrong token; only "down" earns a port scan.
          if (link === "down") noteLinkDown(now)
        }
      } catch (err) {
        if (!aborted(err)) {
          service.noteLink("down")
          noteLinkDown(Date.now())
        }
      } finally {
        metricsInFlight = false
      }
      if (instanceIsOurs()) startUi()
    }

    /** /props: memory headroom and the n-gram warm total. Slow-changing. */
    async function pollProps(): Promise<void> {
      const origin = originOf(feedUrl)
      if (origin === null || propsInFlight) return
      propsInFlight = true
      lastProps = Date.now()
      try {
        const res = await get(`${origin}/props`)
        if (res.ok) service.noteProps(parseProps(res.body as RawPropsJson))
      } catch {
        // Only /metrics.json decides the link state; a missing /props is not "down".
      } finally {
        propsInFlight = false
      }
      if (instanceIsOurs()) startUi()
    }

    /** /v1/models: which model, quantizer and speculative decoder are resident. */
    async function pollModels(): Promise<void> {
      const origin = originOf(feedUrl)
      if (origin === null || modelsInFlight) return
      modelsInFlight = true
      lastModels = Date.now()
      try {
        const res = await get(`${origin}/v1/models`)
        if (res.ok) model = parseModels(res.body)
      } catch {
        // the panel draws without it
      } finally {
        modelsInFlight = false
      }
      if (instanceIsOurs()) startUi()
    }

    /** The log tail: acceptance and sampling. A missing file is a row, not a fault. */
    function pollLog(): void {
      if (tail === null) return
      const now = Date.now()
      lastLog = now
      try {
        logRead = tail.poll(now)
        // With no live feed, a file nobody has written to in minutes belongs to
        // one of the dead runs in the same directory. Its last [spec-stats] line
        // would read as this server's, so the section says `no log file` instead.
        logStale = logIsStale(logRead.status.mtimeMs, now, service.linkState() === "live")
      } catch {
        logRead = null
        logStale = false
      }
    }

    // --- scheduling --------------------------------------------------------

    function busy(now: number): boolean {
      return isBusy(tracker.hasActive(now), now, serverBusyUntil)
    }

    /**
     * Does anything visible need these numbers? The footer meter needs the feed
     * while a turn runs; the slow reads only matter when the panel is on screen.
     */
    function wanted(now: number): boolean {
      return isWanted(instanceIsOurs(), panelOnScreen(), busy(now))
    }

    function stopUi(): void {
      if (uiTimer === undefined) return
      clearInterval(uiTimer)
      uiTimer = undefined
    }

    function tick(): void {
      if (!instanceIsOurs()) {
        stopUi()
        return
      }
      const now = Date.now()
      const running = busy(now)
      const needed = wanted(now)
      if (!needed && !running) {
        stopUi()
        return
      }
      if (needed) {
        setVersion((v) => v + 1)
      }
      reportOnce()

      if (needed && !metricsInFlight) {
        if (isDue(now, lastMetrics, metricsEveryMs(running, options.pollHz, options.idlePollHz))) void pollMetrics()
      }
      if (needed && panelOnScreen()) {
        if (!propsInFlight && isDue(now, lastProps, options.propsSeconds * 1000)) void pollProps()
        if (!modelsInFlight && isDue(now, lastModels, options.modelsSeconds * 1000)) void pollModels()
        if (isDue(now, lastLog, options.logSeconds * 1000)) pollLog()
      }
    }

    function startUi(): void {
      if (uiTimer !== undefined) return
      uiTimer = setInterval(tick, Math.round(1000 / options.refreshHz))
      uiTimer.unref?.()
    }

    /** An integration that failed is worth one interruption; the panel says it forever. */
    function reportOnce(): void {
      if (reported || attachErrors.length === 0) return
      reported = true
      try {
        ctx.ui.toast.show({
          title: "MLX Serve",
          message: `${attachErrors.length} integration(s) failed: ${attachErrors.map((e) => e.where).join(", ")}`,
          variant: "warning",
          duration: 6_000,
        })
      } catch {
        // nothing to toast into; the attach section still carries the detail
      }
    }

    // --- session events ----------------------------------------------------

    const sessionIDOf = (e: AnyEvent) => String(e.data.sessionID ?? "")
    const assistantOf = (e: AnyEvent) => String(e.data.assistantMessageID ?? "")
    const deltaOf = (e: AnyEvent) => (typeof e.data.delta === "string" ? e.data.delta : "")

    function claim(sessionID: string): void {
      if (sessionID === "") return
      activeSessions.add(sessionID)
      lastSession = sessionID
    }

    function onDelta(e: AnyEvent): void {
      if (!instanceIsOurs() || !isNewEvent(e)) return
      const sid = sessionIDOf(e)
      if (sid === "") return
      claim(sid)
      tracker.pushDelta(sid, deltaOf(e), createdAt(e), assistantOf(e))
      startUi()
    }

    function finishStep(e: AnyEvent): void {
      if (!instanceIsOurs() || !isNewEvent(e)) return
      const tokens = e.data.tokens as
        | { input?: number; output?: number; reasoning?: number; cache?: { read?: number }; cacheRead?: number }
        | undefined
      tracker.finishStep(
        sessionIDOf(e),
        assistantOf(e),
        tokens
          ? {
              input: tokens.input,
              output: tokens.output,
              reasoning: tokens.reasoning,
              cacheRead: tokens.cache?.read ?? tokens.cacheRead,
            }
          : undefined,
        createdAt(e),
      )
      startUi()
    }

    function finishRun(e: AnyEvent): void {
      if (!instanceIsOurs() || !isNewEvent(e)) return
      const sid = sessionIDOf(e)
      tracker.finish(sid, createdAt(e))
      activeSessions.delete(sid)
      if (lastSession === sid) lastSession = null
      serverBusyUntil = 0
      startUi()
      // The end of a turn is when the new [spec-stats] line lands in the log and
      // the histograms move. Read them now instead of on the next slow tick.
      pollLog()
      lastMetrics = 0
      void pollMetrics()
      void pollProps()
    }

    function listen(type: string, handler: (e: AnyEvent) => void): () => void {
      try {
        return ctx.data.on(type, handler)
      } catch (err) {
        attachErrors.push({ where: "data.on", detail: type })
        void err
        return () => {}
      }
    }

    const unsubs = [
      listen("session.execution.started", (e) => {
        if (!instanceIsOurs() || !isNewEvent(e)) return
        const sid = sessionIDOf(e)
        if (sid === "") return
        claim(sid)
        tracker.beginRun(sid, createdAt(e))
        startUi()
      }),
      listen("session.text.delta", onDelta),
      listen("session.reasoning.delta", onDelta),
      listen("session.tool.input.delta", onDelta),
      listen("session.step.started", (e) => {
        if (!instanceIsOurs() || !isNewEvent(e)) return
        const sid = sessionIDOf(e)
        if (sid === "") return
        claim(sid)
        noteProvider(sid, e.data.model)
        tracker.beginStep(sid, assistantOf(e), createdAt(e))
        startUi()
      }),
      listen("session.step.streamed", (e) => {
        if (!instanceIsOurs() || !isNewEvent(e)) return
        const sid = sessionIDOf(e)
        if (sid === "") return
        tracker.markStreamed(sid, assistantOf(e), createdAt(e))
        startUi()
      }),
      listen("session.model.selected", (e) => {
        if (!instanceIsOurs() || !isNewEvent(e)) return
        const sid = sessionIDOf(e)
        if (sid === "") return
        noteProvider(sid, e.data.model)
      }),
      listen("session.step.ended", finishStep),
      listen("session.step.failed", finishStep),
      listen("session.execution.succeeded", finishRun),
      listen("session.execution.failed", finishRun),
      listen("session.execution.interrupted", finishRun),
      listen("session.idle", finishRun),
      listen("session.deleted", (e) => {
        if (!instanceIsOurs() || !isNewEvent(e)) return
        const sid = sessionIDOf(e)
        tracker.evict(sid)
        activeSessions.delete(sid)
        sessionProviders.delete(sid)
        if (lastSession === sid) lastSession = null
        startUi()
      }),
    ]

    // --- data for the renderer ---------------------------------------------

    function speedFor(sessionID?: string): SpeedValue | null {
      const sid = sessionID ?? lastSession ?? activeSessions.values().next().value ?? null
      if (sid === null || sid === "") return null
      return tracker.value(sid, Date.now())
    }

    function panelInput(sessionID?: string): PanelInput {
      const now = Date.now()
      const status = logRead?.status ?? null
      return {
        speed: speedFor(sessionID),
        service: service.statsAt(now),
        model,
        // A stale file publishes nothing: not its rows, not its numbers.
        spec: logStale ? null : (tail?.latestSpec() ?? null),
        sampling: logStale ? null : (tail?.latestSampling() ?? null),
        hot: logStale ? null : (tail?.latestHot() ?? null),
        ssd: logStale ? null : (tail?.latestSsd() ?? null),
        diskCacheGb: options.diskCacheGb,
        logDisabled: options.logPathExplicit && options.logPath === null,
        log: status !== null && logStale ? { ...status, bytes: null, mtimeMs: null, error: "no log file" } : status,
        sparkCells: options.sparkCells,
        barCells: options.barCells,
        now,
        link: service.linkState(),
        wiredLimitGb: wiredGb,
        ratioCells: options.ratioCells,
        attachErrors,
      }
    }

    // --- rendering ---------------------------------------------------------

    const dim = () => ctx.theme.text?.subdued ?? ctx.theme.text?.muted ?? ctx.theme.text?.default
    const bright = () => ctx.theme.text?.default ?? ctx.theme.text?.subdued

    function toneColor(tone: string): string | undefined {
      const key = tone === "live" ? "success" : tone === "warn" ? "warning" : tone === "error" ? "error" : "info"
      return ctx.theme.feedback?.[key]?.default ?? dim()
    }

    const Row = (props: { label: string; value: string; note?: string; tone?: string; noteBright?: boolean }) => (
      <text wrapMode="none" truncate>
        {props.label === "" ? null : <span style={{ fg: dim() }}>{props.label} </span>}
        <span style={{ fg: props.tone ? toneColor(props.tone) : bright() }}>{props.value}</span>
        {/* noteBright: the note is a second statistic, not an aside. */}
        {props.note ? <span style={{ fg: props.noteBright ? bright() : dim() }}> {props.note}</span> : null}
      </text>
    )

    /**
     * Keyed by label: the numbers change every frame, the lines do not. An
     * unkeyed `For` sees a fresh array of fresh objects on every repaint and
     * rebuilds every row, which redraws the whole panel eight times a second.
     */
    const Block = (props: { section: SidebarSection }) => (
      <box>
        <text fg={bright()} wrapMode="none" truncate>
          <b>{props.section.title}</b>
          {/* Heading notes are dim asides, unless a second statistic (the Memory
              gauge) or an alert (a dark feed). */}
          {props.section.note === undefined ? null : (
            <span
              style={{
                fg: props.section.noteTone
                  ? toneColor(props.section.noteTone)
                  : props.section.noteBright
                    ? bright()
                    : dim(),
              }}
            >
              {" "}{props.section.note}
            </span>
          )}
        </text>
        <For each={props.section.rows} key={(row) => row.label}>
          {(row) => <Row label={row.label} value={row.value} note={row.note} tone={row.tone} noteBright={row.noteBright} />}
        </For>
      </box>
    )

    /** The sidebar panel: each enabled section that has data. */
    const SidebarPanel = (props: { sessionID?: string }) => {
      // Slow reads run only while this component is mounted.
      createEffect(() => {
        setPanelOnScreen(true)
        onCleanup(() => setPanelOnScreen(false))
        startUi()
      })
      const sections = createMemo(() => {
        version()
        return buildSections(panelInput(props.sessionID), options.sections)
      })
      return (
        <box gap={1}>
          <For each={sections()} key={(section) => section.name}>
            {(section) => <Block section={section} />}
          </For>
        </box>
      )
    }

    /** The turn meter in the prompt footer. */
    const FooterMeter = (props: { sessionID?: string }) => {
      const label = createMemo(() => {
        version()
        if (!instanceIsOurs()) return null
        const value = speedFor(props.sessionID)
        if (value === null) return null
        // A prefill younger than ~5 frames has no tokens yet: drawing a line
        // that changes a blink later is a flicker, not a measurement.
        if (value.phase === "prefill" && value.elapsedMs < 80 && value.prefillTokens === null) return null
        return footerLabel(value, {
          barCells: options.footerBarCells,
        })
      })
      return (
        <Show when={label()}>
          {(text: () => string) => <text fg={dim()}>{text()}</text>}
        </Show>
      )
    }

    const unslots: Array<() => void> = []
    // `prepend` puts the stats above the host's own Context and MCP sections.
    const mounts: Array<{ spec: Record<string, unknown>; render: (input: { sessionID?: string }) => unknown }> = [
      {
        spec: { prepend: "sidebar.content" },
        render: (input) => <SidebarPanel sessionID={input.sessionID} />,
      },
      {
        spec: { append: "prompt.footer.status" },
        render: (input) => <FooterMeter sessionID={input.sessionID} />,
      },
    ]
    for (const mount of mounts) {
      const slotName = String(Object.values(mount.spec)[0] ?? "slot")
      const unslot = attempt(`ui.slot(${slotName})`, () => ctx.ui.slot({ ...mount.spec, render: mount.render }))
      if (unslot !== null) unslots.push(unslot)
    }

    // No log is opened before a server answers: the tail follows the port that
    // did, and until then there is nothing to say which file is this run's.
    void pollMetrics()
    void pollProps()
    void pollModels()
    startUi()
    reportOnce()

    return () => {
      for (const unsub of unsubs) unsub()
      for (const unslot of unslots) unslot()
      stopUi()
      abort.abort()
      if (gen.active === mine)
        setGen((d) => {
          d.active = 0
        })
    }
  },
}

export default definition
