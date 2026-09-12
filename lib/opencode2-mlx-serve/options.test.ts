import assert from "node:assert/strict"
import { test } from "node:test"
import {
  clamp,
  DEFAULTS,
  DEFAULT_SECTIONS,
  originOf,
  resolveOptions,
} from "./options.ts"
import { ALL_SECTIONS } from "./rows.ts"

// --- options ---------------------------------------------------------------

test("resolveOptions defaults to this machine's mlx-serve", () => {
  const o = resolveOptions(undefined)
  assert.equal(o.metricsUrl, "http://127.0.0.1:11234/metrics.json")
  assert.equal(o.metricsToken, "mlx-serve")
  assert.deepEqual(o.sections, [...DEFAULT_SECTIONS])
  assert.equal(o.sections.includes("turn"), false, "the speed meter lives in the footer; the panel must not repeat it")
  assert.equal(o.sections.includes("throughput"), true)
  assert.match(o.logPath ?? "", /\.mlx-serve[\\/]logs[\\/]mlx-serve-11234\.log$/, "the log path follows the port")
  assert.equal(resolveOptions({ metricsUrl: "http://192.168.1.9:8080" }).logPath?.includes("mlx-serve-8080"), true)
})

test("resolveMetricsUrl shapes are all accepted", () => {
  for (const url of ["http://host:11234", "http://host:11234/", "http://host:11234/metrics", "http://host:11234/metrics.json"]) {
    assert.equal(resolveOptions({ metricsUrl: url }).metricsUrl, "http://host:11234/metrics.json")
  }
})

test("logPath off turns the tail off, an explicit path is honoured", () => {
  assert.equal(resolveOptions({ logPath: "off" }).logPath, null)
  assert.equal(resolveOptions({ logPath: "none" }).logPath, null)
  assert.equal(resolveOptions({ logPath: "/srv/logs/mlx.log" }).logPath, "/srv/logs/mlx.log")
  assert.equal(resolveOptions({ logPath: "  " }).logPath, null, "whitespace is not a path")
  assert.equal(resolveOptions({ logPath: "OFF" }).logPath, null, "case-insensitive, like the command args")
  assert.equal(resolveOptions({ logPath: 42 as unknown }).logPath, null, "a non-string is not a path either")
})

test("a junk config cannot make the panel spin or stall", () => {
  const o = resolveOptions({ refreshHz: 100_000, pollHz: -5, idlePollHz: "4", bytesPerToken: Number.NaN, sparkCells: 9999, barCells: 9999, footerBarCells: 9999 })
  assert.equal(o.refreshHz, 30, "clamped to the host's ceiling")
  assert.equal(o.pollHz, 1, "a negative rate clamps to the floor instead of stopping the polls")
  assert.equal(o.idlePollHz, DEFAULTS.idlePollHz, "a string rate falls back")
  assert.equal(o.bytesPerToken, DEFAULTS.bytesPerToken)
  assert.equal(o.sparkCells, 60)
  assert.equal(o.barCells, 12, "the prefill bar is capped at 12 blocks")
  assert.equal(o.footerBarCells, 12)
  assert.equal(resolveOptions({}).footerBarCells, 12, "the footer bar ships at the cap")
})

test("turn is opt-in for people who want it in the panel too", () => {
  assert.deepEqual(resolveOptions({ sections: ["throughput", "turn"] }).sections, ["throughput", "turn"])
  assert.deepEqual([...DEFAULT_SECTIONS, "turn"].sort(), [...ALL_SECTIONS].sort(), "the default list is ALL minus turn")
})

test("sections is the user's list, in their order, minus typos", () => {
  assert.deepEqual(resolveOptions({ sections: ["log", "turn"] }).sections, ["log", "turn"])
  assert.deepEqual(resolveOptions({ sections: ["bogus"] }).sections, [], "a list of typos is an empty panel, not the default")
  assert.deepEqual(resolveOptions({ sections: "turn" }).sections, [...DEFAULT_SECTIONS], "a string is not a list")
  assert.deepEqual(resolveOptions({ sections: [] }).sections, [], "an explicit empty list draws only the header")
})

test("clamp keeps the fallback", () => {
  assert.equal(clamp(5, 1, 2, 3), 3)
  assert.equal(clamp(0, 1, 2, 3), 2)
  assert.equal(clamp("3", 7, 1, 9), 7)
  assert.equal(clamp(Number.POSITIVE_INFINITY, 7, 1, 9), 7)
})

// --- derived slow-endpoint URLs -------------------------------------------

test("originOf recovers the server root from the feed URL", () => {
  assert.equal(originOf("http://127.0.0.1:11234/metrics.json"), "http://127.0.0.1:11234")
  assert.equal(originOf("https://gpu.example:8443/metrics"), "https://gpu.example:8443")
  assert.equal(originOf("http://localhost/metrics.json"), "http://localhost")
  assert.equal(originOf("not a url"), null, "no origin means no /props, not a crash")
})

test("the options say whether the user pinned the log path", () => {
  const auto = resolveOptions(undefined)
  assert.equal(auto.logPathExplicit, false, "the tail follows whichever port answers")
  const pinned = resolveOptions({ metricsUrl: "http://127.0.0.1:8098", logPath: "off" })
  assert.equal(pinned.logPathExplicit, true)
  assert.equal(pinned.logPath, null)
  assert.equal(resolveOptions({ metricsUrl: "   " }).metricsUrl, DEFAULTS.metricsUrl, "whitespace is not a URL")
})

test("provider names whose sessions the local feed is allowed to meter", () => {
  assert.deepEqual(resolveOptions(undefined).provider, ["mlx-serve", "mlx"], "both ids in the wild")
  assert.deepEqual(resolveOptions({ provider: "anthropic" }).provider, ["anthropic"], "a string names one")
  assert.deepEqual(resolveOptions({ provider: ["a", "b"] }).provider, ["a", "b"], "a list names several")
  assert.deepEqual(resolveOptions({ provider: "  ollama  " }).provider, ["ollama"], "trimmed")
  assert.equal(resolveOptions({ provider: null }).provider, null, "null accepts the feed for every session")
  assert.deepEqual(resolveOptions({ provider: "" }).provider, ["mlx-serve", "mlx"], "a blank value is the default, not a wildcard")
  assert.deepEqual(resolveOptions({ provider: [] }).provider, ["mlx-serve", "mlx"])
  assert.deepEqual(resolveOptions({ provider: 7 as unknown }).provider, ["mlx-serve", "mlx"])
})
