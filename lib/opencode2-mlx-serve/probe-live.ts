import { SpeedTracker, parseMetricsJson, type MetricsJson } from "./tracker.ts"
import { footerLabel } from "./rows.ts"

const URL = process.env.METRICS_URL ?? "http://127.0.0.1:11234/metrics.json"
const TOKEN = process.env.METRICS_TOKEN ?? "mlx-serve"
const SECONDS = Number(process.env.PROBE_SECONDS ?? 5)

const tracker = new SpeedTracker()
const sid = "live-probe"
const started = Date.now()
tracker.beginRun(sid, started)
tracker.beginStep(sid, "probe", started)

async function sample(): Promise<void> {
  const res = await fetch(URL, {
    headers: { Authorization: `Bearer ${TOKEN}`, Accept: "application/json" },
  })
  if (!res.ok) throw new Error(`metrics ${res.status} ${res.statusText}`)
  const json = (await res.json()) as MetricsJson
  const snap = parseMetricsJson(json, Date.now())
  tracker.applyMetrics(sid, snap)
  const v = tracker.value(sid, Date.now())
  const label = v ? footerLabel(v, { barCells: 18 }) : "(no value)"
  console.log(
    [
      `+${((Date.now() - started) / 1000).toFixed(1)}s`,
      `run=${snap.running ? 1 : 0}`,
      `prefill=${snap.prefilling ? 1 : 0}`,
      `pre_live=${snap.prefillLive}`,
      `gen_live=${snap.genLive}`,
      "→",
      label,
    ].join(" "),
  )
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

await sample()
const interval = 250
const end = Date.now() + SECONDS * 1000
while (Date.now() < end) {
  await sleep(interval)
  await sample()
}
