// website_benchmarks_logic.mjs — unit-tests the pure logic embedded in
// website/benchmarks/index.html: row validation (v1 + v2), RTDB payload
// parsing, median, settings signature, cell families and the
// speculation-headroom view, and filtering.
// Invoked by test_website_pages.sh when node is available; exits non-zero on
// the first failed assertion.
//
// Like website_tier_list_logic.mjs it evals the page's module script up to the
// DOM-dependent rendering section, so the code under test is the exact code the
// browser runs — no copies.
//
// The grouping logic here MIRRORS BenchmarkStore / BenchmarkSettings in the
// Swift app. If the two diverge, the app and the website quote different
// numbers for the same data, so both sides are tested and the duplication is
// deliberate.
import { readFileSync } from "node:fs";

const html = readFileSync("website/benchmarks/index.html", "utf8");
const script = html.split('<script type="module">')[1]?.split("</script>")[0];
if (!script) { console.error("ASSERT FAIL: module script not found in page"); process.exit(1); }
const pure = script.split("// ── rendering")[0];
if (pure.length === script.length) { console.error("ASSERT FAIL: rendering marker missing"); process.exit(1); }

const asserts = `
function assert(c, m) { if (!c) { console.error("ASSERT FAIL: " + m); process.exit(1); } }

const SETTINGS = { kv_quant: "8", decode_attn_quant: "false", mtp_loaded: "false",
                   mtp_default_on: "false", pld_default_on: "true", drafter: "none",
                   n_ctx: "49152", version: "26.9.4", prefix_cache_mem: "2147483648" };

function row(over) {
  return Object.assign({
    schemaVersion: 2, sessionId: "s1", suiteId: "ctx-v1-4k", armId: "configured", armLabel: "As configured",
    modelId: "mlx-community/Qwen3.6-27B-4bit", isLossy: true,
    prefillTps: 900, decodeTps: 50, ceilingDecodeTps: 90, targetTokens: 4096, promptTokens: 4100,
    settings: SETTINGS,
    hardware: { chip: "Apple M4 Max", gpuCores: 40, ramGB: 128, osVersion: "27.0", onBattery: false },
  }, over || {});
}

function v1row(over) {
  return Object.assign({
    schemaVersion: 1, sessionId: "old", suiteId: "standard-v1", armId: "defaults", armLabel: "Defaults",
    modelId: "mlx-community/Qwen3.6-27B-4bit", isLossy: false, flags: {},
    prefillTps: 900, decodeTps: 40, promptTokens: 2048,
    hardware: { chip: "Apple M4 Max", gpuCores: 40, ramGB: 128 },
  }, over || {});
}

// ── row validation: an open database means anything can arrive ─────────────
assert(isValidRow(row()), "a well-formed v2 row validates");
assert(!isValidRow(v1row()), "a pre-release v1 row is dropped: no rung, no column");
assert(!isValidRow(null), "null is not a row");
assert(!isValidRow({}), "an empty object is not a row");
assert(!isValidRow(row({ decodeTps: 0 })), "zero decode is not a measurement");
assert(!isValidRow(row({ decodeTps: -5 })), "negative decode is rejected");
assert(!isValidRow(row({ decodeTps: "fast" })), "a string decode is rejected");
assert(!isValidRow(row({ hardware: null })), "a row with no hardware is unfilterable");
assert(!isValidRow(row({ hardware: { chip: "", ramGB: 8 } })), "a blank chip is unfilterable");
assert(!isValidRow(row({ modelId: "" })), "a blank model is unusable");
assert(!isValidRow(row({ targetTokens: undefined })), "a v2 row without its rung is rejected");
assert(!isValidRow(row({ settings: undefined })), "a v2 row without settings is rejected");
assert(!isValidRow(row({ targetTokens: "4k" })), "a non-integer rung is rejected");
assert(isValidRow(row({ note: "david, fans on max" })), "a note is optional free text");
assert(!isValidRow(row({ note: 42 })), "a non-string note is rejected");
assert(isValidRow(row({ driftPercent: -4.6, driftDecodeTps: 58.4 })), "drift fields are optional numbers");
assert(!isValidRow(row({ driftPercent: "steady" })), "a non-numeric drift is rejected");

// ── RTDB payload shape: an OBJECT keyed by push id, or null when empty ─────
assert(parseRows(null).length === 0, "an empty database renders as no rows");
assert(parseRows({}).length === 0, "an empty object yields no rows");
assert(parseRows({ "-Na": row(), "-Nb": row() }).length === 2, "push-keyed object yields its rows");
assert(parseRows({ "-Na": row(), "-Njunk": { nonsense: true } }).length === 1,
       "a malformed row is skipped, the good one survives");

// ── median ────────────────────────────────────────────────────────────────
assert(median([]) === 0, "median of nothing is 0");
assert(median([42]) === 42, "median of one");
assert(median([10, 20]) === 15, "median of two averages");
assert(median([50, 51, 20]) === 50, "one slow run does not drag the median");

// ── settings signature (mirrors BenchmarkSettings.signature) ──────────────
assert(settingsSignature(SETTINGS) === "kv8|daq0|mtp0|pld1|none", "signature spells what changes speed");
assert(settingsSignature(undefined) === "kv?|daq?|mtp?|pld?|?", "missing settings have an unknown signature");
assert(settingsSignature(Object.assign({}, SETTINGS, { version: "27.0.0", prefix_cache_mem: "1" }))
       === settingsSignature(SETTINGS), "server version and cache size do not change the signature");
assert(settingsSignature(Object.assign({}, SETTINGS, { kv_quant: "off" })) !== settingsSignature(SETTINGS),
       "kv quant changes the signature");
const chips = settingsChips(SETTINGS);
assert(chips.join(",") === "KV 8-bit,PLD,MTP off,ctx 48K", "chips name what matters: " + chips.join(","));
assert(settingsChips(undefined).length === 0, "no settings, no chips");

// ── family grouping: only genuinely comparable rows share a median ────────
const m4max40 = row();
const m4max32 = row({ hardware: { chip: "Apple M4 Max", gpuCores: 32, ramGB: 128 } });
assert(familyKey(m4max40) !== familyKey(m4max32),
       "GPU core count separates families — 32 and 40 core M4 Max are different machines");
assert(familyKey(row({ decodeTps: 99 })) === familyKey(row()),
       "the measurement itself is not part of the key");
assert(familyKey(row({ modelId: "other" })) !== familyKey(row()), "model separates families");
assert(familyKey(row({ settings: Object.assign({}, SETTINGS, { kv_quant: "off" }) })) !== familyKey(row()),
       "settings signature separates families");
assert(familyKey(row({ engineVersion: "26.9.0" })) === familyKey(row({ engineVersion: "26.8.1" })),
       "engine version does not fragment families");
assert(familyKey(row({ suiteId: "ctx-v1-512", targetTokens: 512 })) === familyKey(row()),
       "the rung is a column, not a family");

// ── cell families: one row per machine × model × settings, a column per rung
const fam = aggregateFamilies([
  row({ sessionId: "a", suiteId: "ctx-v1-512", targetTokens: 512, decodeTps: 60, ceilingDecodeTps: 120 }),
  row({ sessionId: "a", suiteId: "ctx-v1-4k", targetTokens: 4096, decodeTps: 50, ceilingDecodeTps: 90 }),
  row({ sessionId: "b", suiteId: "ctx-v1-512", targetTokens: 512, decodeTps: 70, ceilingDecodeTps: 100 }),
  row({ sessionId: "c", suiteId: "ctx-v1-512", targetTokens: 512, decodeTps: 30,
        settings: Object.assign({}, SETTINGS, { kv_quant: "off" }) }),
]);
assert(fam.length === 2, "two settings signatures make two families");
const kv8 = fam.find((f) => f.settings.kv_quant === "8");
assert(kv8.rungs.get(512).decodeTps === 65, "per-rung median across sessions");
assert(kv8.rungs.get(512).sampleCount === 2, "per-rung n");
assert(kv8.rungs.get(4096).decodeTps === 50 && kv8.rungs.get(4096).sampleCount === 1, "a rung one session reached");
assert(kv8.sessionCount === 2, "n on the row is sessions, not rungs");
const noted = aggregateFamilies([row({ note: " david " }), row({ sessionId: "b", note: "david" }), row({ sessionId: "c", note: "m4 fanless" })]);
assert(noted[0].notes.join("|") === "david|m4 fanless", "notes are trimmed and de-duplicated: " + noted[0].notes.join("|"));
assert(aggregateFamilies([row()])[0].notes.length === 0, "no note, no notes");
assert(fam[0] === kv8, "families sort by the fastest smallest rung");


// ── speculation headroom: ceiling ÷ decode per rung, inside one session ────
const head = headroomByCell([
  row({ sessionId: "a", suiteId: "ctx-v1-512", targetTokens: 512, decodeTps: 50, ceilingDecodeTps: 100 }),
  row({ sessionId: "a", suiteId: "ctx-v1-4k", targetTokens: 4096, decodeTps: 40, ceilingDecodeTps: 0 }),
]);
assert(head.length === 1, "a family with any measurable rung is reported");
assert(Math.abs(head[0].rungs.get(512).headroom - 2.0) < 1e-9, "headroom is ceiling over decode");
assert(head[0].rungs.get(4096).headroom === null, "a failed ceiling run never yields a ratio");
assert(headroomByCell([row({ ceilingDecodeTps: 0 })]).length === 0, "a family with no ceiling anywhere is dropped from the view");

// ── filters ───────────────────────────────────────────────────────────────
const mixed = [
  row({ hardware: { chip: "Apple M4 Max", gpuCores: 40, ramGB: 128 } }),
  row({ hardware: { chip: "Apple M4", gpuCores: 10, ramGB: 16 } }),
  row({ isLossy: false, settings: Object.assign({}, SETTINGS, { kv_quant: "off" }) }),
];
assert(applyFilters(mixed, { chip: "Apple M4" }).length === 1, "chip filter");
assert(applyFilters(mixed, { ram: "16" }).length === 1, "memory filter");
assert(applyFilters(mixed, { quality: "lossless" }).length === 1, "lossless filter drops lossy rows");
assert(applyFilters(mixed, {}).length === 3, "no filters keeps everything");
assert(applyFilters(mixed, { model: "nope" }).length === 0, "unknown model matches nothing");

// ── the fetch URL must actually be RTDB's query grammar ───────────────────
const url = fetchURL();
assert(url.includes("%22%24key%22"), 'orderBy="$key" must be URL-encoded or RTDB 400s');
assert(url.includes("limitToLast="), "fetch is bounded, never the whole database");

console.log("website benchmarks logic: all assertions passed");
`;

const module = pure + asserts;
try {
  // eslint-disable-next-line no-new-func
  new Function(module)();
} catch (e) {
  console.error("ASSERT FAIL: " + (e && e.message ? e.message : String(e)));
  process.exit(1);
}
