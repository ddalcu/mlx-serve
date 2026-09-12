/**
 * Finding the mlx-serve that is actually running.
 *
 * Nothing on disk names the live port: there is no pidfile and no server config,
 * only `~/.mlx-serve/logs/mlx-serve-<port>.log` and the process's own argv. So
 * the log directory is the candidate list — newest file first — and each
 * candidate is confirmed by asking its `/metrics.json`. 503 counts as an answer:
 * it means the server is there with `--metrics` off, which the panel already has
 * a state for.
 *
 * Pure here (names, ordering, URLs, staleness); the fetch loop lives in tui.tsx.
 */

import { homedir } from "node:os"

/** `mlx-serve-11234.log` → 11234; anything else → null. */
const LOG_NAME = /^mlx-serve-(\d{1,5})\.log$/

export interface LogFile {
  readonly name: string
  readonly mtimeMs: number
}

/** How long a probe of one candidate may take before it is written off. */
export const PROBE_TIMEOUT_MS = 300
/** Candidates tried per probe, newest log first. */
export const PROBE_CANDIDATES = 3
/** A link down this long earns another look for a server on a different port. */
export const REPROBE_AFTER_MS = 30_000
/** With no live feed, a log older than this is somebody else's dead run. */
export const LOG_STALE_MS = 5 * 60_000

export function logsDir(home: string = homedir()): string {
  return `${home}/.mlx-serve/logs`
}

export function portFromLogName(name: string): number | null {
  const m = LOG_NAME.exec(name)
  if (!m?.[1]) return null
  const port = Number(m[1])
  return Number.isInteger(port) && port > 0 && port <= 65535 ? port : null
}

/** The ports worth probing: newest log file first, at most `limit`. */
export function candidatePorts(files: readonly LogFile[], limit = PROBE_CANDIDATES): number[] {
  const scored: { port: number; mtimeMs: number }[] = []
  for (const file of files) {
    const port = portFromLogName(file.name)
    if (port === null) continue
    const mtimeMs = Number.isFinite(file.mtimeMs) ? file.mtimeMs : 0
    scored.push({ port, mtimeMs })
  }
  scored.sort((a, b) => b.mtimeMs - a.mtimeMs)
  const seen = new Set<number>()
  const out: number[] = []
  for (const entry of scored) {
    if (seen.has(entry.port)) continue
    seen.add(entry.port)
    out.push(entry.port)
    if (out.length >= limit) break
  }
  return out
}

/** The loopback feed URL for a port. */
export function metricsUrlForPort(port: number, host = "127.0.0.1"): string {
  return `http://${host}:${port}/metrics.json`
}

/** A server is there if it answers at all: 200, or 503 for `--metrics` off. */
export function answered(status: number): boolean {
  return status === 200 || status === 503
}

/**
 * Whether a log file may be shown as this server's. With no live feed a stale
 * file is one of the dead runs the log directory is full of, and its last
 * `[spec-stats]` line would read as current.
 */
export function logIsStale(mtimeMs: number | null, now: number, feedLive: boolean): boolean {
  if (feedLive || mtimeMs === null) return false
  return now - mtimeMs > LOG_STALE_MS
}
