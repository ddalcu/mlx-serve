/**
 * Tail of the mlx-serve server log.
 *
 * mlx-serve persists its log to `~/.mlx-serve/logs/mlx-serve-<port>.log`
 * (`log.defaultLogPath`, rotating to `<path>.1` at 32 MB) and prints into that
 * file the two numbers no HTTP endpoint carries:
 *
 *   [spec-stats] mode=mtp attempts=55 accepts=89 … per_draft_pct=38.4% depth=6 …
 *   POST /v1/chat/completions (127 msgs, max_tokens=64000 (launch default), temp=1.00, top_p=0.95, top_k=20 …)
 *
 * generate.zig documents `[spec-stats]` as a stable format for external tooling.
 * The file is only read; a server on another host shows up as `no log file`.
 *
 * The tail is incremental: a byte offset, advanced only to the last complete
 * line we decoded, so a half-written line is read again next poll instead of
 * being parsed as garbage, and cutting at a newline can never split a UTF-8
 * sequence. When the file shrinks or the inode changes we rewind (rotation, or a
 * fresh server on the same port), and a backlog larger than `chunkBytes` is
 * skipped forward to the last `backBytes` rather than crawled a chunk per poll.
 *
 * Lines read on the first poll, after a rewind or after a skip are stamped with
 * the file's mtime, not with the clock: they were written before we looked, and
 * the panel ages a row by that stamp.
 */

import { closeSync, openSync, readSync, statSync } from "node:fs"
import { homedir } from "node:os"
import { basename } from "node:path"
import {
  parseCacheTier,
  parseSampling,
  parseSpecStats,
  type CacheTier,
  type SamplingStats,
  type SpecStats,
} from "./stats.ts"

/** `<home>/.mlx-serve/logs/mlx-serve-<port>.log`, mirroring `log.defaultLogPath`. */
export function defaultLogPath(port: number, home: string = homedir()): string {
  return `${home}/.mlx-serve/logs/mlx-serve-${port}.log`
}

/** Port of a URL like `http://127.0.0.1:11234/metrics.json`, or `fallback`. */
export function portFromUrl(url: string, fallback = 80): number {
  const m = /^[a-z][a-z0-9+.-]*:\/\/[^/?#]*?:(\d{1,5})(?:[/?#]|$)/i.exec(url)
  if (m?.[1]) return Number(m[1])
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(url)) return /https:\/\//i.test(url) ? 443 : 80
  return fallback
}

/** A value read from the log, stamped with when we read it. */
export interface Observed<T> {
  readonly value: T
  /** Wall-clock ms when we read the line; the panel ages a row by it. */
  readonly at: number
}

export interface LogStatus {
  readonly path: string
  readonly name: string
  /** null when the file is unreadable. */
  readonly bytes: number | null
  readonly mtimeMs: number | null
  /** null when the file is readable. */
  readonly error: string | null
  /** Lines scanned on this poll. */
  readonly lines: number
  /** Bytes skipped unread because the backlog exceeded `chunkBytes`. */
  readonly dropped: number
}

/** What a poll saw. The values themselves are read back with `latestSpec()` and friends. */
export interface LogRead {
  readonly status: LogStatus
}

export interface TailOptions {
  /** How far back to read on the first poll and after a rewind. */
  readonly backBytes?: number
  /** Largest read per poll; a bigger backlog is skipped forward. */
  readonly chunkBytes?: number
}

const DEFAULT_BACK = 256 * 1024
const DEFAULT_CHUNK = 1024 * 1024
const NEWLINE = 0x0a

function errText(err: unknown): string {
  if (err instanceof Error) return err.message
  return String(err)
}

interface Stat {
  bytes: number | null
  mtimeMs: number | null
  error: string | null
  lines: number
  dropped: number
}

export class LogTail {
  readonly path: string
  private readonly backBytes: number
  private readonly chunkBytes: number
  private offset: number | null = null
  private inode: number | null = null
  private spec: Observed<SpecStats> | null = null
  private sampling: Observed<SamplingStats> | null = null
  private hot: Observed<CacheTier> | null = null
  private ssd: Observed<CacheTier> | null = null

  constructor(path: string, options: TailOptions = {}) {
    this.path = path
    this.backBytes = Math.max(0, options.backBytes ?? DEFAULT_BACK)
    this.chunkBytes = Math.max(4096, options.chunkBytes ?? DEFAULT_CHUNK)
  }

  latestSpec(): Observed<SpecStats> | null {
    return this.spec
  }

  latestSampling(): Observed<SamplingStats> | null {
    return this.sampling
  }

  latestHot(): Observed<CacheTier> | null {
    return this.hot
  }

  latestSsd(): Observed<CacheTier> | null {
    return this.ssd
  }

  /** Reads what was appended since the last poll. Never throws, never blocks long. */
  poll(now: number = Date.now()): LogRead {
    let stat: ReturnType<typeof statSync>
    try {
      stat = statSync(this.path)
    } catch (err) {
      const code = (err as { code?: string }).code
      return this.done({
        bytes: null,
        mtimeMs: null,
        error: code === "ENOENT" ? "no log file" : errText(err),
        lines: 0,
        dropped: 0,
      })
    }

    const first = this.offset === null
    let offset = this.offset as number
    let historic = first
    let dropped = 0
    if (first) {
      offset = Math.max(0, stat.size - this.backBytes)
      this.inode = stat.ino
    } else if (stat.ino !== this.inode || stat.size < offset) {
      // Rotated to `<path>.1`, or a new server truncated the same file.
      offset = Math.max(0, stat.size - this.backBytes)
      this.inode = stat.ino
      historic = true
    } else if (stat.size - offset > this.chunkBytes) {
      // A backlog (the sidebar was hidden for a long session): jump to the tail
      // instead of parsing megabytes of dead lines a chunk per poll. Only the
      // newest line of each kind survives a catch-up anyway.
      const target = Math.max(offset, stat.size - this.backBytes)
      dropped = target - offset
      offset = target
      historic = true
    }

    const cap = Math.min(stat.size, offset + this.chunkBytes)
    const wanted = cap - offset
    if (wanted <= 0) {
      this.offset = offset
      return this.done({ bytes: stat.size, mtimeMs: stat.mtimeMs, error: null, lines: 0, dropped })
    }

    let buffer: Buffer
    try {
      const fd = openSync(this.path, "r")
      try {
        buffer = Buffer.allocUnsafe(wanted)
        const got = readSync(fd, buffer, 0, wanted, offset)
        buffer = got === wanted ? buffer : buffer.subarray(0, got)
      } finally {
        closeSync(fd)
      }
    } catch (err) {
      // Do not advance the offset: the next poll reads these bytes again.
      return this.done({ bytes: stat.size, mtimeMs: stat.mtimeMs, error: errText(err), lines: 0, dropped: 0 })
    }

    const lastNewline = buffer.lastIndexOf(NEWLINE)
    if (lastNewline < 0) {
      if (cap < stat.size) {
        // A single line longer than the whole chunk: walk past it.
        this.offset = stat.size
        return this.done({ bytes: stat.size, mtimeMs: stat.mtimeMs, error: null, lines: 0, dropped: dropped + stat.size - offset })
      }
      this.offset = offset
      return this.done({ bytes: stat.size, mtimeMs: stat.mtimeMs, error: null, lines: 0, dropped })
    }

    const consumed = lastNewline + 1
    const text = buffer.subarray(0, consumed).toString("utf8")
    this.offset = offset + consumed
    // Bytes that were already on disk when we attached were written before now.
    const at = historic ? (stat.mtimeMs || now) : now
    const lines = this.consume(text, at)

    return this.done({
      bytes: stat.size,
      mtimeMs: stat.mtimeMs,
      error: null,
      lines,
      dropped,
    })
  }

  private done(part: Stat): LogRead {
    return { status: { path: this.path, name: basename(this.path), ...part } }
  }

  /** Scans freshly completed lines; the newest line of each kind wins. */
  private consume(text: string, now: number): number {
    let spec: SpecStats | null = null
    let sampling: SamplingStats | null = null
    let hot: CacheTier | null = null
    let ssd: CacheTier | null = null
    let lines = 0
    for (const line of text.split("\n")) {
      if (line.length === 0) continue
      lines++
      if (line.includes("[spec-stats]")) {
        const parsed = parseSpecStats(line)
        if (parsed) spec = parsed
      } else if (line.trimStart().startsWith("POST /v1/")) {
        const parsed = parseSampling(line)
        if (parsed) sampling = parsed
      } else if (line.includes("[hot-cache] resident=") || line.includes("[disk-cache] persisted ")) {
        // Both cache tiers log their own occupancy whenever they insert or evict,
        // which is the only place the SSD tier is visible at all.
        const tier = parseCacheTier(line)
        if (tier?.kind === "hot") hot = tier
        else if (tier?.kind === "ssd") ssd = tier
      }
    }
    if (spec) this.spec = { value: spec, at: now }
    if (sampling) this.sampling = { value: sampling, at: now }
    if (hot) this.hot = { value: hot, at: now }
    if (ssd) this.ssd = { value: ssd, at: now }
    return lines
  }
}
