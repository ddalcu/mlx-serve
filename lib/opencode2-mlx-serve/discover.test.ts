import assert from "node:assert/strict"
import { test } from "node:test"
import {
  LOG_STALE_MS,
  REPROBE_AFTER_MS,
  answered,
  candidatePorts,
  logIsStale,
  logsDir,
  metricsUrlForPort,
  portFromLogName,
} from "./discover.ts"
import {
  holdsServerBusy,
  isBusy,
  isDue,
  isWanted,
  metricsEveryMs,
} from "./schedule.ts"

// --- candidates ------------------------------------------------------------

test("only mlx-serve's own log names carry a port", () => {
  assert.equal(portFromLogName("mlx-serve-11234.log"), 11234)
  assert.equal(portFromLogName("mlx-serve-8098.log"), 8098)
  assert.equal(portFromLogName("mlx-serve-11234.log.1"), null, "a rotated file is not the live one")
  assert.equal(portFromLogName("qwen38flash-nohup.out"), null)
  assert.equal(portFromLogName("mlx-serve-.log"), null)
  assert.equal(portFromLogName("mlx-serve-99999999.log"), null, "not a port number")
})

test("the newest logs are the candidates, at most three", () => {
  const files = [
    { name: "mlx-serve-8098.log", mtimeMs: 100 },
    { name: "audit-qsa-boot.out", mtimeMs: 9_000 },
    { name: "mlx-serve-11234.log", mtimeMs: 9_000 },
    { name: "mlx-serve-22223.log", mtimeMs: 5_000 },
    { name: "mlx-serve-22224.log", mtimeMs: 4_000 },
  ]
  assert.deepEqual(candidatePorts(files), [11234, 22223, 22224], "newest first, non-log files ignored")
  assert.deepEqual(candidatePorts(files, 1), [11234])
  assert.deepEqual(candidatePorts([]), [], "an empty log directory is not an error")
  assert.deepEqual(candidatePorts([{ name: "mlx-serve-1.log", mtimeMs: Number.NaN }]), [1], "a broken mtime sorts last, not away")
})

test("a candidate is addressed on loopback and answers with 200 or 503", () => {
  assert.equal(metricsUrlForPort(11234), "http://127.0.0.1:11234/metrics.json")
  assert.equal(answered(200), true)
  assert.equal(answered(503), true, "--metrics off is still a server")
  assert.equal(answered(404), false)
  assert.equal(answered(401), false, "a wrong token says nothing about the port")
  assert.match(logsDir("/Users/x"), /^\/Users\/x\/\.mlx-serve\/logs$/)
})

test("without a live feed an old log file is not this server's", () => {
  const now = 1_800_000_000_000
  assert.equal(logIsStale(now - 60_000, now, false), false, "a minute old, the server just went quiet")
  assert.equal(logIsStale(now - LOG_STALE_MS - 1, now, false), true, "older than the floor: treat it as no log file")
  assert.equal(logIsStale(now - 86_400_000, now, true), false, "a live feed vouches for the file, whatever its age")
  assert.equal(logIsStale(null, now, false), false, "an unreadable file is already its own status")
})

// --- polling rules ---------------------------------------------------------

test("busy is our own turn or the server still working", () => {
  assert.equal(isBusy(true, 1_000, 0), true, "our turn is running")
  assert.equal(isBusy(false, 1_000, 6_000), true, "the server is still busy")
  assert.equal(isBusy(false, 7_000, 6_000), false)
})

test("nothing is polled for a plugin instance that has been superseded", () => {
  assert.equal(isWanted(false, true, true), false, "an old instance draws nothing, so it reads nothing")
  assert.equal(isWanted(true, false, false), false, "hidden panel, idle turn")
  assert.equal(isWanted(true, true, false), true, "the panel is on screen")
  assert.equal(isWanted(true, false, true), true, "the footer meter needs the feed")
})

test("the poll rate follows the work, and a read is due on its own interval", () => {
  assert.equal(metricsEveryMs(true, 4, 1), 250)
  assert.equal(metricsEveryMs(false, 4, 1), 1000)
  assert.equal(isDue(1_000, 750, 250), true)
  assert.equal(isDue(1_000, 800, 250), false)
  assert.equal(isDue(1_000, 0, 250), true, "never read is always due")
})

test("another client's work does not hold the fast poll open by itself", () => {
  assert.equal(holdsServerBusy(false, false), false, "nothing of ours on screen or in flight")
  assert.equal(holdsServerBusy(true, false), true, "the panel is drawing the server's own numbers")
  assert.equal(holdsServerBusy(false, true), true, "our turn is running behind a hidden sidebar")
  assert.ok(REPROBE_AFTER_MS >= 30_000, "a dead link is re-probed, but not every tick")
})
