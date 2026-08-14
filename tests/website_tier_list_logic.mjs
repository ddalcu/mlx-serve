// website_tier_list_logic.mjs — unit-tests the pure logic embedded in
// website/llm-tier-list/index.html: Wilson-score tiering, vote sanitization,
// tally aggregation, and the Hugging Face popularity pipeline (canonical
// top-level model grouping, quant/finetune roll-up rules, size→RAM tiers).
// Invoked by test_website_pages.sh when node is available; exits non-zero on
// the first failed assertion.
//
// It evals the page's module script up to the DOM-dependent rendering section,
// so the code under test is the exact code the browser runs — no copies.
import { readFileSync } from "node:fs";

const html = readFileSync("website/llm-tier-list/index.html", "utf8");
const script = html.split('<script type="module">')[1]?.split("</script>")[0];
if (!script) { console.error("ASSERT FAIL: module script not found in page"); process.exit(1); }
const pure = script.split("// ── rendering")[0];
if (pure.length === script.length) { console.error("ASSERT FAIL: rendering marker missing"); process.exit(1); }

// ── dead-control guard ────────────────────────────────────────────────────
// A typo'd getElementById is a control that silently never works, and every
// byte-level assertion in test_website_pages.sh still passes. Same check the
// built-in console page runs against app.js (tests/html_console_test.mjs).
{
  const ids = new Set();
  for (const m of html.matchAll(/\bid=(?:"([^"]+)"|([\w-]+))/g)) ids.add(m[1] || m[2]);
  const referenced = new Set();
  for (const m of script.matchAll(/\$\("([\w-]+)"\)/g)) referenced.add(m[1]);
  for (const m of script.matchAll(/getElementById\("([\w-]+)"\)/g)) referenced.add(m[1]);
  if (referenced.size < 15) { console.error("ASSERT FAIL: the id scan found suspiciously little"); process.exit(1); }
  const missing = [...referenced].filter((id) => !ids.has(id));
  if (missing.length) {
    console.error("ASSERT FAIL: script reaches for ids absent from the markup: " + missing.join(", "));
    process.exit(1);
  }
  // the chooser ships CLOSED: a modal whose hidden attribute is only added by
  // JS covers the page for anyone who loads it with scripting stalled
  if (!/<div class="modal" id="model-modal" hidden>/.test(html)) {
    console.error("ASSERT FAIL: the model modal must be hidden in the markup, not just by script");
    process.exit(1);
  }
  // ONE tiering chokepoint: the board must not reach past tierForModel to the
  // vote-only tierFor, or flipping TIER_MODE would leave half the page behind
  const boardRender = script.split("// ── rendering")[1].split("// ── quant playground")[0];
  if (/[^r]\btierFor\(/.test(boardRender)) {
    console.error("ASSERT FAIL: board rendering calls tierFor directly — go through tierForModel");
    process.exit(1);
  }
  if (!/tierForModel\(/.test(boardRender)) {
    console.error("ASSERT FAIL: board rendering must tier through tierForModel");
    process.exit(1);
  }
}

const asserts = `
function assert(c, m) { if (!c) { console.error("ASSERT FAIL: " + m); process.exit(1); } }

// ── Wilson-score tiering: needs MIN_VOTES to rank, consistency to climb ────
assert(wilsonLower(0, 0) === 0, "wilson of no votes is 0");
assert(wilsonLower(8, 0) > wilsonLower(3, 0), "confidence grows with sample size");
assert(tierFor(0, 0) === "U", "no votes stays unranked");
assert(tierFor(1, 0) === "C", "first upvote enters cautiously at C (MIN_VOTES=1)");
assert(tierFor(2, 0) === "B", "2-0 climbs to B");
assert(tierFor(3, 0) === "A", "3-0 ranks A (not yet S)");
assert(tierFor(8, 0) === "S", "8-0 reaches S");
assert(tierFor(0, 5) === "D", "0-5 is D");

// ── HF pipeline: quants roll UP, finetunes drop OUT, one entry per model ───
const NOW = Date.parse("2026-07-20T00:00:00Z");
const FIX_TEXT = [
  // official instruct model (finetune-of-own-Base is the normal official shape)
  { id: "Qwen/Qwen3.6-27B", downloads: 500000,
    tags: ["transformers", "safetensors", "qwen3_5", "text-generation",
           "base_model:finetune:Qwen/Qwen3.6-27B-Base", "license:apache-2.0"],
    safetensors: { total: 27000000000 }, createdAt: "2026-02-01T00:00:00.000Z" },
  // its pretrain sibling — must merge into the same top-level entry
  { id: "Qwen/Qwen3.6-27B-Base", downloads: 90000,
    tags: ["transformers", "qwen3_5", "text-generation"],
    safetensors: { total: 27000000000 }, createdAt: "2026-02-01T00:00:00.000Z" },
  // pure quantized conversions — downloads roll up into the official entry
  { id: "mlx-community/Qwen3.6-27B-4bit", downloads: 40000,
    tags: ["mlx", "qwen3_5", "text-generation", "base_model:Qwen/Qwen3.6-27B",
           "base_model:quantized:Qwen/Qwen3.6-27B"],
    safetensors: { total: 27000000000 }, createdAt: "2026-02-10T00:00:00.000Z" },
  { id: "mlx-community/Qwen3.6-27B-8bit", downloads: 10000,
    tags: ["mlx", "qwen3_5", "base_model:quantized:Qwen/Qwen3.6-27B"],
    safetensors: { total: 27000000000 }, createdAt: "2026-02-10T00:00:00.000Z" },
  // the official org's OWN quant release (live-observed NVFP4 class) — must
  // merge into the base entry, not fork; packed param count must not shrink
  // the size estimate
  { id: "Qwen/Qwen3.6-27B-NVFP4", downloads: 60000,
    tags: ["transformers", "qwen3_5", "text-generation"],
    safetensors: { total: 14000000000 }, createdAt: "2026-04-01T00:00:00.000Z" },
  // community finetune: popular, but NOT a top-level model — rejected
  { id: "coolhacker/Dolphin-Qwen3.6-27B", downloads: 999999,
    tags: ["qwen3_5", "text-generation", "base_model:finetune:Qwen/Qwen3.6-27B"],
    safetensors: { total: 27000000000 }, createdAt: "2026-03-01T00:00:00.000Z" },
  // quant OF a community finetune — also rejected (target org not official)
  { id: "mlx-community/Dolphin-Qwen3.6-27B-4bit", downloads: 888888,
    tags: ["mlx", "qwen3_5", "base_model:quantized:coolhacker/Dolphin-Qwen3.6-27B"],
    safetensors: { total: 27000000000 }, createdAt: "2026-03-02T00:00:00.000Z" },
  // unknown architecture, not a GGUF release — rejected
  { id: "SomeOrg/WeirdArch-70B", downloads: 777777,
    tags: ["superarch", "text-generation"],
    safetensors: { total: 70000000000 }, createdAt: "2026-01-01T00:00:00.000Z" },
  // modern arch beyond mlx-serve's native dispatch — popularity gate keeps it
  { id: "openai/gpt-oss-20b", downloads: 7200000,
    tags: ["transformers", "safetensors", "gpt_oss", "text-generation"],
    safetensors: { total: 21000000000 }, createdAt: "2025-08-05T00:00:00.000Z" },
  // LEGACY arch from an official org — must stay off the board
  { id: "Qwen/Qwen2.5-7B-Instruct", downloads: 11500000,
    tags: ["transformers", "safetensors", "qwen2", "text-generation"],
    safetensors: { total: 7600000000 }, createdAt: "2024-09-16T00:00:00.000Z" },
  // pre-2024 model on a still-listed arch — date floor must reject it
  { id: "meta-llama/Llama-2-7b-hf", downloads: 700000,
    tags: ["transformers", "safetensors", "llama", "text-generation"],
    safetensors: { total: 6700000000 }, createdAt: "2023-07-18T00:00:00.000Z" },
  // new-lab official release (Ornith = deepreinforce-ai, arch qwen3_5_moe)
  { id: "deepreinforce-ai/Ornith-1.0-35B", downloads: 1350838,
    tags: ["transformers", "safetensors", "qwen3_5_moe", "text-generation"],
    safetensors: { total: 35000000000 }, createdAt: "2026-06-25T00:00:00.000Z" },
  // official-org GGUF release with NO arch tag (Bonsai class) — gguf pass-through
  { id: "prism-ml/Bonsai-27B-gguf", downloads: 1262894,
    tags: ["llama.cpp", "gguf", "conversational", "1-bit"],
    createdAt: "2026-07-04T00:00:00.000Z" },
];
const FIX_VISION = [
  // recent official vision model → vision tag + New badge
  { id: "google/gemma-4-e2b-it", downloads: 800000,
    tags: ["transformers", "gemma4", "image-text-to-text",
           "base_model:finetune:google/gemma-4-e2b"],
    safetensors: { total: 5000000000 }, createdAt: "2026-07-05T00:00:00.000Z" },
];

const groups = groupHfModels(FIX_TEXT, FIX_VISION, NOW);
const byId = Object.fromEntries(groups.map((g) => [g.id, g]));

const qwen = byId["qwen3.6-27b"];
assert(qwen, "official model present under its normalized id");
assert(qwen.downloads === 700000, "quant + Base + official-quant downloads rolled up (got " + (qwen && qwen.downloads) + ")");
assert(qwen.params === "27B", "packed-quant param count doesn't shrink the size (got " + (qwen && qwen.params) + ")");
assert(prettyName("gemma-3-270m") === "Gemma 3 270M", "sub-B param suffix uppercased");
assert(prettyName("gpt-oss-20b") === "GPT OSS 20B", "short alpha tokens uppercased");
assert(normalizeBaseName("MiniMax-M3-MXFP8") === "minimax-m3", "mxfp8 quant suffix stripped");
assert(qwen.name === "Qwen3.6 27B", "pretty top-level name (got " + (qwen && qwen.name) + ")");
assert(qwen.minRam === 24, "27B at ~4-bit lands on the 24 GB tier (got " + (qwen && qwen.minRam) + ")");
assert(groups.filter((g) => g.id.includes("qwen3.6-27b")).length === 1, "one entry per top-level model");
assert(!groups.some((g) => g.id.includes("dolphin")), "community finetunes rejected");
assert(!groups.some((g) => g.id.includes("weirdarch")), "unknown non-GGUF architectures rejected");
assert(byId["gpt-oss-20b"], "modern arch beyond native dispatch is listed (gpt_oss)");
assert(!byId["qwen2.5-7b"], "legacy archs stay off the board even from official orgs");
assert(!byId["llama-2-7b"], "pre-2024 releases rejected by the date floor");
assert(byId["ornith-1.0-35b"] && byId["ornith-1.0-35b"].name === "Ornith 1.0 35B",
  "new-lab org (deepreinforce-ai) is listed");
assert(byId["bonsai-27b"] && byId["bonsai-27b"].isNew,
  "official-org GGUF release without an arch tag passes (Bonsai class)");

const gemma = byId["gemma-4-e2b"];
assert(gemma, "vision-pipeline model present (-it stripped from id)");
assert(gemma.name === "Gemma 4 E2B", "vision model pretty name (got " + (gemma && gemma.name) + ")");
assert(gemma.tags.includes("vision"), "image-text-to-text marks vision");
assert(gemma.isNew === true, "recently created model flagged New");
assert(qwen.isNew === false, "old model not flagged New");
assert(gemma.minRam === 8, "5B at ~4-bit fits the 8 GB tier");

// vote ids must be Firestore-map-key-safe and org-free
for (const g of groups) {
  assert(/^[a-z0-9][a-z0-9._+-]*$/.test(g.id), "id is a safe stable key: " + g.id);
}

// seed union: pinned models HF can't surface (GGUF-only) survive, no dupes
const merged = mergeWithSeed(groups, SEED_MODELS);
assert(merged.some((m) => m.id === "deepseek-v4-flash"), "pinned seed survives merge when not found dynamically");
assert(merged.filter((m) => m.id === "qwen3.6-27b").length === 1, "dynamic entry wins over same-id seed");

// ── vote sanitization: at most one exact ±1 per KNOWN model per account ────
setActiveModels(merged);
const s = sanitize({ "qwen3.6-27b": 1, "bogus-model": 1, "gemma-4-e2b": 2, "deepseek-v4-flash": -1 });
assert(JSON.stringify(s) === JSON.stringify({ "qwen3.6-27b": 1, "deepseek-v4-flash": -1 }),
  "sanitize drops unknown ids and non-plus-minus-1 values");

const t = talliesFrom([{ "qwen3.6-27b": 1 }, { "qwen3.6-27b": 1, "deepseek-v4-flash": -1 }]);
assert(t["qwen3.6-27b"].up === 2 && t["qwen3.6-27b"].down === 0 && t["deepseek-v4-flash"].down === 1,
  "tallies aggregate across voter docs");

// ── unranked table: text filter + vote-independent ordering ────────────────
const probe = { id: "qwen3.6-27b", name: "Qwen3.6 27B", params: "27B", tags: ["vision"], downloads: 5 };
assert(matchesQuery(probe, ""), "empty query matches everything");
assert(matchesQuery(probe, "  qWeN "), "name match is case/whitespace-insensitive");
assert(matchesQuery(probe, "vision"), "tags are searchable");
assert(matchesQuery(probe, "27b"), "params are searchable");
assert(!matchesQuery(probe, "llama"), "non-matches are filtered out");
const order = unrankedOrder([
  { id: "a", name: "A", downloads: 5 },
  { id: "b", name: "B", downloads: 50 },
  { id: "c", name: "C", downloads: 0 },
  { id: "d", name: "D", downloads: 5 },
]).map((m) => m.id);
assert(JSON.stringify(order) === JSON.stringify(["b", "a", "d", "c"]),
  "unranked order is downloads-desc + name tiebreak — votes can never reorder it (got " + order + ")");

// ── promotion detection: celebrate real U→tier moves, never a load storm ───
const prev = {};
// first render with real tallies only BASELINES (track=false): models that
// are already ranked on page load must not celebrate
let promos = promotionsSince(prev, [{ id: "m1", tier: "A" }, { id: "m2", tier: "U" }], false);
assert(promos.length === 0, "baseline render never celebrates");
// a later render where m2 earned its votes — that's a real promotion
promos = promotionsSince(prev, [{ id: "m1", tier: "A" }, { id: "m2", tier: "B" }], true);
assert(promos.length === 1 && promos[0].id === "m2", "U-to-ranked transition detected");
// staying ranked or moving between tiers is not a promotion
promos = promotionsSince(prev, [{ id: "m1", tier: "S" }, { id: "m2", tier: "B" }], true);
assert(promos.length === 0, "tier-to-tier moves don't fire the effect");

// ── seed data integrity: unique ids, labeled tags, minRam on a filter tier ─
const ids = SEED_MODELS.map((m) => m.id);
assert(new Set(ids).size === ids.length, "seed ids are unique");
const ramTiers = new Set([8, 16, 24, 32, 48, 64, 96, 128, 192]);
for (const m of SEED_MODELS) {
  assert(ramTiers.has(m.minRam), "seed minRam matches a filter tier: " + m.id);
  for (const tag of m.tags) assert(TAG_LABELS[tag], "seed tag has a label: " + tag);
}

// ════════════════════════════════════════════════════════════════════════
// QUANT PLAYGROUND — the setup calculator above the board.
// ════════════════════════════════════════════════════════════════════════

// ── numeric params ride WITH the model: the display label ("26B (4B act)")
//    is prose, the calculator needs floats and must never re-parse it ─────
for (const m of SEED_MODELS) {
  assert(typeof m.paramsB === "number" && m.paramsB > 0, "seed carries numeric paramsB: " + m.id);
  assert(typeof m.activeB === "number" && m.activeB > 0 && m.activeB <= m.paramsB,
    "seed carries numeric activeB <= paramsB: " + m.id);
}
assert(Math.abs(byId["qwen3.6-27b"].paramsB - 27) < 0.6, "HF group exports paramsB");
assert(byId["ornith-1.0-35b"].activeB === byId["ornith-1.0-35b"].paramsB,
  "MoE with no -aNb token falls back to dense active count");

// ── Mac configs: the binned die is a REAL config, not a footnote. Apple only
//    sells the low-RAM Max with fewer GPU cores and less bandwidth, so the
//    RAM choice picks the die ───────────────────────────────────────────────
const m4max = MAC_CHIPS.find((c) => c.id === "m4-max");
assert(m4max, "M4 Max is a listed chip");
assert(chipConfig(m4max, 128).bw === 546, "full M4 Max die is 546 GB/s");
assert(chipConfig(m4max, 36).bw === 410, "36 GB M4 Max is the binned 410 GB/s die");
assert(chipConfig(m4max, 36).cores < chipConfig(m4max, 128).cores, "binned die has fewer GPU cores");
assert(chipConfig(m4max, 128).tflops > chipConfig(m4max, 36).tflops, "tflops follow the die");
for (const c of MAC_CHIPS) {
  assert(c.ram.length && c.ram.every((r) => r > 0), "chip lists RAM options: " + c.id);
  assert(chipConfig(c, c.ram[0]).bw > 0 && chipConfig(c, c.ram[0]).tflops > 0,
    "every chip resolves to a usable config: " + c.id);
}
// Ultras that Apple never shipped must not be inventable hardware
assert(!MAC_CHIPS.some((c) => c.id === "m4-ultra"), "no M4 Ultra (never shipped)");
assert(chipFor("M4", "Ultra") === null, "an unshipped combination resolves to nothing");
assert(chipFor("M1", "") && chipFor("M1", "").id === "m1", "base tier is the empty string");
// changing chip must never strand a RAM size you cannot buy that chip with
assert(nearestRam(chipFor("M1", ""), 512) === 16, "512 GB clamps to the biggest M1 config");
assert(nearestRam(chipFor("M3", "Ultra"), 64) === 96, "clamps up to the smallest Ultra config");
assert(nearestRam(chipFor("M4", "Max"), 64) === 64, "an available size is kept as-is");

// ── quantization: bytes per weight includes the affine scale+bias overhead,
//    and 4-bit must agree with the sizing constant the board already uses ──
assert(Math.abs(bytesPerParam(4) - 0.5625) < 1e-9, "4-bit = 4.5 bits/weight incl. gs64 scales+biases");
assert(bytesPerParam(16) === 2, "bf16 is 2 bytes");
const bitsLadder = [2, 3, 4, 6, 8, 16];
for (let i = 1; i < bitsLadder.length; i++) {
  assert(bytesPerParam(bitsLadder[i]) > bytesPerParam(bitsLadder[i - 1]), "bytes/param is monotone in bits");
  assert(quantQuality(bitsLadder[i], 30) >= quantQuality(bitsLadder[i - 1], 30), "quality is monotone in bits");
}
assert(quantQuality(16, 30) === 1, "bf16 is the quality reference");
assert(quantQuality(2, 4) < quantQuality(2, 200), "low bits hurt a small model more than a big one");

// ── SPEED MODEL CALIBRATION ───────────────────────────────────────────────
// These four cells are OUR OWN measurements from docs/perf-csvs/all-26.7.12.csv (git history)
// (mlx-serve, spec=none, Apple-M4-Max-128gb). The calculator must reproduce
// them: it is a roofline fit, and this is the fit's evidence. Change a
// constant and this test says which measurement you stopped matching.
const M4MAX = { chip: MAC_CHIPS.find((c) => c.id === "m4-max"), ram: 128 };
const CAL = [
  // model                                     paramsB activeB  moe    decode prefill
  { name: "gemma-4-31b-4bit",                   p: 31,   a: 31,  moe: 0, d: 25.2,  f: 206.8 },
  { name: "qwen3.6-27b-oQ4e",                   p: 27.8, a: 27.8, moe: 0, d: 28.0,  f: 237.2 },
  { name: "qwen3.6-35b-a3b-oQ4",                p: 36,   a: 3,   moe: 1, d: 152.4, f: 1462.3 },
  { name: "gemma-4-26b-a4b-qat",                p: 25.2, a: 3.8, moe: 1, d: 118.2, f: 1434.6 },
  // MatFormer/PLE outlier: billed on TOTAL params because a decode step
  // reads the elastic model's full weights, not its 4B "active" count.
  { name: "gemma-4-e4b",                        p: 8,    a: 4,   moe: 0, d: 114.8, f: 2159.9, tol: 0.20 },
];
for (const c of CAL) {
  const m = { paramsB: c.p, activeB: c.a, tags: c.moe ? ["moe"] : [], id: c.name };
  const hw = chipConfig(M4MAX.chip, M4MAX.ram);
  const opts = { bits: 4, kvBits: 16, ctx: 851, spec: "none", workload: "code" };
  const d = estimateDecodeTps(m, hw, opts);
  const f = estimatePrefillTps(m, hw, opts);
  assert(Math.abs(d - c.d) / c.d < (c.tol || 0.12),
    "decode within tolerance of measured " + c.name + ": model " + d.toFixed(1) + " vs bench " + c.d);
  assert(Math.abs(f - c.f) / c.f < 0.30,
    "prefill within 30% of measured " + c.name + ": model " + f.toFixed(0) + " vs bench " + c.f);
}
// the E-series rule must be a NAMED exception, not a global change of billing
assert(estimateDecodeTps({ id: "gemma-4-e4b", paramsB: 8, activeB: 4, tags: [] },
         chipConfig(M4MAX.chip, 128), { bits: 4, kvBits: 16, ctx: 851, spec: "none", workload: "code" })
     < estimateDecodeTps({ id: "some-other-8b", paramsB: 8, activeB: 4, tags: [] },
         chipConfig(M4MAX.chip, 128), { bits: 4, kvBits: 16, ctx: 851, spec: "none", workload: "code" }),
  "MatFormer billing applies to the E-series only");

// ── bandwidth roofline: decode scales with bandwidth, not core count ──────
{
  const m = { paramsB: 27.8, activeB: 27.8, tags: [], id: "x" };
  const o = { bits: 4, kvBits: 16, ctx: 1024, spec: "none", workload: "code" };
  const slow = estimateDecodeTps(m, chipConfig(MAC_CHIPS.find((c) => c.id === "m4-pro"), 64), o);
  const fast = estimateDecodeTps(m, chipConfig(MAC_CHIPS.find((c) => c.id === "m4-max"), 128), o);
  assert(fast > slow, "more bandwidth = faster decode");
  const ratio = fast / slow, bwRatio = 546 / 273;
  assert(Math.abs(ratio - bwRatio) / bwRatio < 0.05, "decode is bandwidth-proportional (got " + ratio.toFixed(2) + "x)");
  // halving the weight bytes must roughly double decode
  const q8 = estimateDecodeTps(m, chipConfig(MAC_CHIPS.find((c) => c.id === "m4-max"), 128), { ...o, bits: 8 });
  assert(q8 < fast && fast / q8 > 1.6, "8-bit weights roughly halve decode speed");
}

// ── context decay: KV reads join the per-token byte bill. Pinned against the
//    64k ladder in docs/perf-csvs/mtp-ladder-26.7.12.csv, git history (27B dense, spec off:
//    51.8 tok/s @1k -> 28.83 @64k, i.e. 0.56x) ─────────────────────────────
{
  const m = { paramsB: 27.8, activeB: 27.8, tags: [], id: "x" };
  const hw = chipConfig(MAC_CHIPS.find((c) => c.id === "m4-max"), 128);
  const base = { bits: 4, kvBits: 16, spec: "none", workload: "code" };
  const at1k = estimateDecodeTps(m, hw, { ...base, ctx: 1024 });
  const at64k = estimateDecodeTps(m, hw, { ...base, ctx: 65536 });
  const decay = at64k / at1k;
  assert(Math.abs(decay - 0.556) < 0.08, "64k decode decay matches the measured ladder (got " + decay.toFixed(3) + ")");
  const kv8 = estimateDecodeTps(m, hw, { ...base, ctx: 65536, kvBits: 8 });
  assert(kv8 > at64k, "quantizing the KV cache buys back long-context decode");
  assert(estimatePrefillTps(m, hw, { ...base, ctx: 1024 }) === estimatePrefillTps(m, hw, { ...base, ctx: 65536 }),
    "prefill tok/s is compute-bound: independent of context length");
}

// ── memory: weights + KV + engine overhead, against the 75% usable rule the
//    board's own minRam tier already applies ────────────────────────────────
{
  const m = { paramsB: 27.8, activeB: 27.8, tags: [], id: "x" };
  const small = estimateMemoryGb(m, { bits: 4, kvBits: 16, ctx: 4096 });
  const big = estimateMemoryGb(m, { bits: 8, kvBits: 16, ctx: 4096 });
  assert(Math.abs(small.weights - 27.8 * 0.5625) < 0.01, "weights = params x bytes/param");
  assert(big.weights > small.weights && big.total > small.total, "more bits = bigger footprint");
  assert(small.kv > 0 && small.total > small.weights + small.kv, "footprint carries KV + engine overhead");
  const longCtx = estimateMemoryGb(m, { bits: 4, kvBits: 16, ctx: 131072 });
  assert(longCtx.kv > small.kv * 10, "KV grows with context");
  assert(fitsRam(small.total, 32) === true, "a 15.6 GB setup fits a 32 GB Mac");
  assert(fitsRam(small.total, 16) === false, "the same setup does not fit 16 GB");
  assert(fitsRam(24, 32) === false, "usable RAM bites before the raw total does");
}

// ── USABLE RAM: a fixed OS/driver reserve, NOT a flat percentage ──────────
// A flat 75% is roughly right at 8 GB and badly wrong at 128 GB, where the
// absolute overhead is a much smaller share. The anchor is Metal's own
// max_recommended_working_set_size, which src/server.zig records as ~115 GB
// on a 128 GB Mac (see the #64 note there) — not the 96 GB a 75% rule gives.
{
  assert(Math.abs(usableRamGb(128) - 115) <= 1,
    "128 GB Mac gives ~115 GB usable, matching the engine's observed working set (got " + usableRamGb(128) + ")");
  assert(usableRamGb(8) >= 4.5 && usableRamGb(8) <= 6, "small Macs keep a proportionally larger reserve");
  const tiers = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512];
  for (let i = 1; i < tiers.length; i++) {
    assert(usableRamGb(tiers[i]) > usableRamGb(tiers[i - 1]), "usable RAM grows with RAM");
    assert(usableRamGb(tiers[i]) < tiers[i], "usable RAM never reaches the full stick");
    assert(usableRamGb(tiers[i]) / tiers[i] > usableRamGb(tiers[i - 1]) / tiers[i - 1] - 1e-9,
      "the usable FRACTION improves with size, it never gets worse");
  }
  // one rule, two callers: the board's RAM tier and the playground's verdict
  // must never disagree about whether a model fits
  assert(minRamFor(115) === 128,
    "our 115.4 GB DeepSeek mirror is a 128 GB model, not a 192 GB one (got " + minRamFor(115) + ")");
  const dsv4 = SEED_MODELS.find((m) => m.id === "deepseek-v4-flash");
  assert(dsv4.minRam === 128, "the DeepSeek seed is placed on the Mac it actually runs on");
  assert(dsv4.minRam === minRamFor(dsv4.sizeGb), "seed minRam agrees with the shared rule");

  // past the recommended working set but physically resident = TIGHT, not
  // impossible: it runs with a raised wired limit and a quiet machine
  assert(ramVerdict(90, 128) === "ok", "comfortably inside the working set");
  assert(ramVerdict(117, 128) === "tight", "just past the recommended set still runs");
  assert(ramVerdict(160, 128) === "no", "beyond physical RAM is beyond physical RAM");
  assert(fitsRam(117, 128) === false, "fitsRam stays the COMFORTABLE line");
}

// ── mixed-precision: the recipe that puts a 284B model on a 128 GB Mac ────
// Large-MoE mirrors do not ship at a uniform width — ours is
// DeepSeek-V4-Flash-0731-MLX-Serve-mixed-2-3-8bit: low-bit experts, 8-bit
// spine, 115.4 GB. A uniform ladder cannot express it, so the page would
// claim the model needs hardware it does not need.
{
  assert(QUANT_BITS.includes("mix"), "the ladder offers a mixed-precision option");
  assert(bytesPerParam("mix") < bytesPerParam(3), "mixed is CHEAPER than uniform 3-bit: most experts sit at 2");
  assert(bytesPerParam("mix") > bytesPerParam(2), "...and dearer than uniform 2-bit: the spine stays 8-bit");
  assert(quantQuality("mix", 284) > quantQuality(3, 284), "calibrated mixed beats uniform 3-bit on quality");
  assert(quantQuality("mix", 284) < quantQuality(4, 284), "...and does not beat uniform 4-bit");

  // it is an EXPERT-quantization recipe: meaningless on a dense model
  assert(quantAvailable("mix", { tags: ["moe"] }) === true, "mixed is offered for an MoE");
  assert(quantAvailable("mix", { tags: [] }) === false, "mixed is not offered for a dense model");
  assert(quantAvailable(4, { tags: [] }) === true, "uniform widths are always offered");

  const dsv4 = SEED_MODELS.find((m) => m.id === "deepseek-v4-flash");
  const mem = estimateMemoryGb(dsv4, { bits: "mix", kvBits: 16, ctx: 8192 });
  assert(Math.abs(mem.weights - 115.4) < 3,
    "mixed reproduces the published mirror size (got " + mem.weights.toFixed(1) + " GB vs 115.4)");
  assert(ramVerdict(mem.total, 128) === "tight",
    "DeepSeek runs on a 128 GB Mac at mixed precision, tightly (got " + ramVerdict(mem.total, 128) + ")");
  assert(ramVerdict(estimateMemoryGb(dsv4, { bits: 4, kvBits: 16, ctx: 8192 }).total, 128) === "no",
    "...and genuinely does not at uniform 4-bit");
}

// ── speculative decoding: measured multipliers, and MTP can't be claimed on
//    a checkpoint that has no head ──────────────────────────────────────────
{
  const moe = { paramsB: 36, activeB: 3, tags: ["moe"], id: "x" };
  assert(specMultiplier("none", "prose", moe) === 1, "spec off is 1.0x");
  assert(specMultiplier("mtp", "agent", moe) > specMultiplier("mtp", "prose", moe),
    "echo-shaped agent traffic drafts better than prose");
  assert(specMultiplier("mtp", "code", moe) > specMultiplier("pld", "code", moe),
    "a trained MTP head beats prompt-lookup on code");
  assert(specMultiplier("pld", "prose", moe) >= 1, "PLD never slows a request in the model");
  // an MTP head ships inside a checkpoint — it is not a switch you can flip
  assert(specSupported("mtp", { id: "qwen3.6-27b" }) === true, "qwen3.6 ships a native MTP head");
  assert(specSupported("mtp", { id: "ling-3.0-flash" }) === true, "ling ships a native MTP head");
  assert(specSupported("mtp", { id: "llama-3.3-70b" }) === false, "no MTP head on llama 3.3");
  assert(specSupported("pld", { id: "llama-3.3-70b" }) === true, "PLD is model-agnostic");
  assert(specSupported("drafter", { id: "anything-9b" }) === true, "drafter is model-agnostic");
  const hw = chipConfig(MAC_CHIPS.find((c) => c.id === "m4-max"), 128);
  const o = { bits: 4, kvBits: 16, ctx: 2048, workload: "code" };
  const off = estimateDecodeTps(moe, hw, { ...o, spec: "none" });
  const on = estimateDecodeTps(moe, hw, { ...o, spec: "mtp" });
  assert(on > off * 1.3, "MTP lifts the reported decode rate");
  assert(estimatePrefillTps(moe, hw, { ...o, spec: "mtp" }) === estimatePrefillTps(moe, hw, { ...o, spec: "none" }),
    "speculation is a decode lever: prefill is untouched");
}

// ── intelligence: a published score is quoted, an unpublished one is an
//    ESTIMATE and must say so — never a fabricated citation ────────────────
{
  const known = intelligenceFor({ id: "gemma-4-31b", paramsB: 30.7, activeB: 30.7 }, 16);
  assert(known.score === AA_INDEX["gemma-4-31b"], "published AA score quoted verbatim at bf16");
  assert(known.est === false, "a published score is not flagged as an estimate");
  const unknown = intelligenceFor({ id: "totally-made-up-9b", paramsB: 9, activeB: 9 }, 16);
  assert(unknown.est === true, "an unscored model is flagged as an estimate");
  assert(unknown.score > 0, "estimate is still a usable number");
  const q4 = intelligenceFor({ id: "gemma-4-31b", paramsB: 30.7, activeB: 30.7 }, 4);
  const q2 = intelligenceFor({ id: "gemma-4-31b", paramsB: 30.7, activeB: 30.7 }, 2);
  assert(q4.score < known.score && q4.score > q2.score, "quantization only ever costs intelligence");
  assert(q4.score / known.score > 0.9, "4-bit is close to lossless");
  assert(q2.score / known.score < 0.8, "2-bit is a real quality cliff");
  for (const [id, v] of Object.entries(AA_INDEX)) {
    assert(typeof v === "number" && v > 0 && v <= 100, "AA score in range: " + id);
  }
  // the bar's ceiling is derived, so a stronger model rescales it
  assert(AA_MAX === Math.max(...Object.values(AA_INDEX)), "AA_MAX is derived from the table");
  for (const id of Object.keys(AA_INDEX)) {
    const m = { id, paramsB: 30, activeB: 30 };
    for (const b of [2, 4, 16]) {
      assert(intelligenceFor(m, b).score <= AA_MAX,
        "no model can overflow the intelligence bar: " + id + " @" + b + "-bit");
    }
  }
}

// ── BOARD TIERING: intelligence now, votes once there are votes ───────────
// With an empty vote table a Wilson board is all-U, so the board ranks on the
// published AA index until the community has voted. tierForModel is the ONE
// chokepoint; flipping TIER_MODE back to "votes" restores the old behaviour
// without touching anything else.
{
  assert(TIER_MODE === "intelligence" || TIER_MODE === "votes", "TIER_MODE names a real mode");
  const m = (id) => ({ id, paramsB: 30, activeB: 30 });
  const noVotes = { up: 0, down: 0 };

  // intelligence mode: the score decides, votes are ignored
  assert(tierForModel(m("deepseek-v4-flash"), noVotes, "intelligence") === "S", "50 on the index is S");
  assert(tierForModel(m("qwen3.6-27b"), noVotes, "intelligence") === "A", "37 is A");
  assert(tierForModel(m("gemma-4-31b"), noVotes, "intelligence") === "B", "29 is B");
  assert(tierForModel(m("gemma-4-12b"), noVotes, "intelligence") === "C", "22 is C");
  assert(tierForModel(m("llama-3.3-70b"), noVotes, "intelligence") === "D", "9 is D");
  assert(tierForModel(m("deepseek-v4-flash"), { up: 0, down: 99 }, "intelligence") === "S",
    "downvotes cannot move a model while the board ranks on intelligence");
  // a model AA has not scored is NOT placed on an estimate — it waits below
  assert(tierForModel(m("some-unscored-model"), noVotes, "intelligence") === "U",
    "an estimated score is not good enough to tier a model");

  // votes mode: unchanged Wilson behaviour, ready to switch back on
  assert(tierForModel(m("deepseek-v4-flash"), noVotes, "votes") === "U", "no votes = unranked in votes mode");
  assert(tierForModel(m("llama-3.3-70b"), { up: 8, down: 0 }, "votes") === "S",
    "in votes mode a low-intelligence model can still be voted to S");
  for (const up of [0, 1, 2, 3, 8]) {
    assert(tierForModel(m("x"), { up, down: 0 }, "votes") === tierFor(up, 0),
      "votes mode delegates to the wilson tier exactly");
  }

  // the tiers must actually spread the current snapshot across the board
  const spread = {};
  for (const id of Object.keys(AA_INDEX)) {
    const t = tierForModel({ id, paramsB: 30, activeB: 30 }, noVotes, "intelligence");
    spread[t] = (spread[t] || 0) + 1;
  }
  for (const t of ["S", "A", "B", "C", "D"]) {
    assert(spread[t] >= 2, "tier " + t + " is not empty on the current AA snapshot (" + JSON.stringify(spread) + ")");
  }
  assert(!spread.U, "every AA-scored model lands on the board");
}

// ── card sorting: ties resolve by NAME so the grid is deterministic ───────
{
  const e = (id, name, downloads, iq, decode) => ({ model: { id, name, downloads }, iq, decode });
  const set = [
    e("b", "Beta", 10, 30, 50),
    e("a", "Alpha", 10, 20, 90),
    e("c", "Gamma", 99, 25, 10),
  ];
  const ids = (mode) => sortModelEntries(set, mode).map((x) => x.model.id).join("");
  assert(ids("popular") === "cab", "popularity first, then name for the two tied on downloads");
  assert(ids("name") === "abc", "name sorts A-Z");
  assert(ids("intel") === "bca", "intelligence sorts high-to-low");
  assert(ids("speed") === "abc", "speed sorts fast-to-slow");
  assert(sortModelEntries(set, "nonsense").map((x) => x.model.id).join("") === "abc",
    "an unknown mode still returns a stable order, never an arbitrary one");
  assert(sortModelEntries(set, "name") !== set, "sorting does not mutate the caller's array");
  assert(SORT_MODES.every(([m]) => ids(m).length === 3), "every offered sort mode works");
}

// ── the calculator must answer for EVERY model on the board ───────────────
{
  const hw = chipConfig(MAC_CHIPS.find((c) => c.id === "m3-max"), 64);
  for (const m of merged) {
    const o = { bits: 4, kvBits: 16, ctx: 8192, spec: "none", workload: "code" };
    const d = estimateDecodeTps(m, hw, o), f = estimatePrefillTps(m, hw, o);
    const mem = estimateMemoryGb(m, o), iq = intelligenceFor(m, 4);
    assert(isFinite(d) && d > 0, "decode is a finite positive number for " + m.id);
    assert(isFinite(f) && f > 0, "prefill is a finite positive number for " + m.id);
    assert(isFinite(mem.total) && mem.total > 0, "footprint is finite for " + m.id);
    assert(isFinite(iq.score) && iq.score > 0, "intelligence is finite for " + m.id);
  }
}

console.log("tier-list logic OK (" + groups.length + " fixture groups, " + SEED_MODELS.length + " seeds, " +
  MAC_CHIPS.length + " Mac configs, " + Object.keys(AA_INDEX).length + " AA scores)");
`;

eval(pure + asserts);

// ══════════════════════════════════════════════════════════════════════════
// PLAYGROUND WIRING — the pure model above can be perfect while the panel
// still renders nothing. This runs the REAL renderRig() against a DOM stub,
// so a ReferenceError or a wrong element reaches CI instead of the page.
// ══════════════════════════════════════════════════════════════════════════
const rigTail = script.split("// ── quant playground: wiring")[1]?.split("// ── filters")[0];
if (!rigTail) { console.error("ASSERT FAIL: playground wiring markers missing"); process.exit(1); }
// the split ate the "//" off the marker line, leaving its box-drawing rule as
// bare source — hand it back so the slice still starts inside a comment
const rigSrc = "//" + rigTail;

const stub = `
function elem(tag) {
  return {
    tag, children: [], _html: "", textContent: "", value: "", disabled: false,
    selected: false, dataset: {}, style: {}, handlers: {},
    className: "",
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
      toggle(c, on) { if (on === undefined) on = !this._s.has(c); on ? this._s.add(c) : this._s.delete(c); },
    },
    hidden: false, focus() { DOM._focused = this; },
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = v; this.children = []; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(ev, fn) { this.handlers[ev] = fn; },
    setAttribute() {},
    // the card builder reaches back into markup it just wrote
    querySelector(sel) { return (this._q = this._q || elem("span")); },
  };
}
const DOM = {};
const document = {
  getElementById(id) { return (DOM[id] = DOM[id] || elem("div")); },
  createElement(tag) { return elem(tag); },
  querySelectorAll() { return []; },
  addEventListener(ev, fn) { (this.handlers = this.handlers || {})[ev] = fn; },
  get activeElement() { return DOM._focused || null; },
  body: elem("body"),
};
function render() {}          // board render: not under test here
function assert(c, m) { if (!c) { console.error("ASSERT FAIL: " + m); process.exit(1); } }
`;

const rigAsserts = `
// boot the panel the way the page's boot block does
rig.modelId = "qwen3.6-27b";
renderRig();

const num = (id) => parseFloat(String(DOM[id].textContent || DOM[id]._html).replace(/[^0-9.]/g, ""));
assert(DOM["rig-gen"].children.length === 5, "five chip generations offered");
assert(DOM["rig-tier"].children.length === 4, "four chip tiers offered");
assert(DOM["rig-tier"].children[3].textContent === "Ultra" && DOM["rig-tier"].children[3].disabled === true,
  "M4 has no Ultra: the tier is shown but not selectable");
assert(DOM["rig-tier"].children[2].disabled === false, "M4 Max is selectable");
assert(DOM["rig-ram"].children.length > 1, "RAM options rendered for the selected chip");
assert(DOM["rig-bits"].children.length === QUANT_BITS.length, "every quantization rung is offered");
// qwen3.6-27b is dense, so the mixed-expert rung must be visible but dead
assert(DOM["rig-bits"].children.find((b) => b.textContent === "Mixed").disabled === true,
  "mixed precision is not selectable on a dense model");
assert(DOM["rig-model-name"].textContent === "Qwen3.6 27B", "trigger names the selected model");
assert(DOM["rig-model-meta"].textContent.length > 0, "trigger carries the model's params");
assert(/GPU/.test(DOM["rig-spec"].textContent) && /GB\\/s/.test(DOM["rig-spec"].textContent),
  "chip spec line names cores and bandwidth (got: " + DOM["rig-spec"].textContent + ")");

// the three headline numbers must actually be numbers
assert(num("out-decode") > 0, "decode stat rendered");
assert(num("out-prefill") > 0, "prefill stat rendered");
assert(num("out-iq") > 0, "intelligence stat rendered");
assert(DOM["out-verdict"].textContent.length > 20, "verdict sentence rendered");
assert(DOM["out-mem"].textContent.includes("usable"), "memory legend rendered");
// intelligence reads as a BAR against a fixed ceiling, not a bare number
assert(DOM["out-iq-bar"].children.length === 2, "intelligence bar has a kept and a lost segment");
assert(DOM["out-iq-legend"].textContent.includes(" of " + AA_MAX), "intelligence legend states the ceiling");
assert(DOM["out-cmd"].textContent.startsWith("mlx-serve run "), "runnable command rendered");
assert(DOM["rig-foot"].innerHTML.includes("artificialanalysis.ai"), "intelligence source is cited");
// the data here goes stale, so the page has to ask for fixes and point at the
// EXACT file that holds them
assert(DOM["rig-foot"].innerHTML.includes(
  "https://github.com/ddalcu/mlx-serve/edit/main/website/llm-tier-list/index.html"),
  "the footnote links straight to the file a contributor would edit");
assert(/pull request/i.test(DOM["rig-foot"].innerHTML), "...and says what to send");

// ── the controls must actually DO something ──────────────────────────────
const decode4 = num("out-decode");
DOM["rig-bits"].children.find((b) => b.textContent === "8-bit").handlers.click();
const decode8 = num("out-decode");
assert(decode8 < decode4, "moving 4-bit -> 8-bit slows decode (" + decode4 + " -> " + decode8 + ")");
DOM["rig-bits"].children.find((b) => b.textContent === "4-bit").handlers.click();

// the bar must show the PRICE of a quant, not just restate the score
const pct = (el) => parseFloat(el.style.width);
const lost4 = pct(DOM["out-iq-bar"].children[1]), kept4 = pct(DOM["out-iq-bar"].children[0]);
DOM["rig-bits"].children.find((b) => b.textContent === "2-bit").handlers.click();
assert(pct(DOM["out-iq-bar"].children[1]) > lost4, "2-bit visibly forfeits more of the bar than 4-bit");
assert(pct(DOM["out-iq-bar"].children[0]) < kept4, "...and keeps less of it");
DOM["rig-bits"].children.find((b) => b.textContent === "bf16").handlers.click();
assert(pct(DOM["out-iq-bar"].children[1]) === 0, "bf16 forfeits nothing");
assert(DOM["out-iq-legend"].textContent.includes("no quantization loss"), "bf16 legend says so");
DOM["rig-bits"].children.find((b) => b.textContent === "4-bit").handlers.click();

const before = num("out-decode");
DOM["rig-mode"].children.find((b) => b.textContent === "MTP").handlers.click();
assert(num("out-decode") > before, "enabling MTP raises the reported rate");

// switching to a chip generation with no Ultra must not strand the tier
DOM["rig-gen"].children.find((b) => b.textContent === "M5").handlers.click();
assert(DOM["rig-tier"].children[3].disabled === true, "M5 Ultra is offered but disabled (not shipped)");
assert(num("out-decode") > 0, "panel still computes after switching generation");
DOM["rig-gen"].children.find((b) => b.textContent === "M1").handlers.click();
DOM["rig-tier"].children.find((b) => b.textContent === "Base").handlers.click();
assert(DOM["rig-ram"].children.every((b) => /^(8|16) GB$/.test(b.textContent)),
  "an M1 cannot be configured with RAM Apple never sold it with");

// a model with no MTP head must not be able to claim one
DOM["rig-gen"].children.find((b) => b.textContent === "M4").handlers.click();
DOM["rig-tier"].children.find((b) => b.textContent === "Max").handlers.click();
rig.modelId = "llama-3.3-70b";
rig.spec = "mtp";
renderRig();
assert(rig.spec === "none", "MTP is dropped for a checkpoint that ships no head");
assert(DOM["rig-mode"].children.find((b) => b.textContent === "MTP").disabled === true,
  "the MTP control is disabled, not silently ignored");

// a model that cannot fit must say so rather than quoting a speed
rig.modelId = "hy3";
DOM["rig-ram"].children.find((b) => b.textContent === "36 GB").handlers.click();
assert(/Won't fit/.test(DOM["out-verdict"].textContent),
  "a 295B model on a 36 GB Mac is refused (got: " + DOM["out-verdict"].textContent + ")");

// ── MODEL CHOOSER MODAL ───────────────────────────────────────────────────
// The point of the grid is that every card is scored on the CURRENT Mac and
// settings — a list of names would not need a modal.
{
  const el = (id) => document.getElementById(id);
  rig.modelId = "qwen3.6-27b"; rig.bits = 4; rig.ram = 64;
  rig.gen = "M4"; rig.tier = "Max"; rig.spec = "none";
  renderRig();

  el("rig-model").handlers.click();
  assert(el("model-modal").hidden === false, "the trigger opens the modal");
  assert(document.body.classList.contains("modal-open"), "page scroll is locked behind the modal");
  const cards = el("model-grid").children;
  assert(cards.length === activeModels.length, "every model gets a card");
  assert(cards.some((c) => / class="mcard on"|mcard on/.test(c.className)),
    "the current model's card is marked selected");
  assert(el("model-modal-sub").textContent.includes("M4 Max") &&
         el("model-modal-sub").textContent.includes("64 GB"),
    "the modal states what the cards are scored on (got: " + el("model-modal-sub").textContent + ")");

  const card = (id) => cards.find((c) => c.dataset.model === id);
  const qwen = card("qwen3.6-27b");
  assert(/mcard-num/.test(qwen.innerHTML), "card carries stat tiles");
  assert((qwen.innerHTML.match(/mcard-num"/g) || []).length === 3, "three numbers per card: tok/s, prompt, intel");
  assert(/iq-have/.test(qwen.innerHTML) && /iq-lost/.test(qwen.innerHTML), "card carries the intelligence bar");
  assert(/Fits 64 GB/.test(qwen.innerHTML), "a card that fits says so");
  assert(/mcard nofit/.test(card("deepseek-v4-flash").className), "a model that cannot fit is dimmed");
  assert(/Needs /.test(card("deepseek-v4-flash").innerHTML), "...and states the Mac it would need");

  // ── sort ────────────────────────────────────────────────────────────────
  const gridIds = () => el("model-grid").children.map((c) => c.dataset.model);
  const byId2 = Object.fromEntries(activeModels.map((m) => [m.id, m]));
  const hw2 = chipConfig(chipFor(rig.gen, rig.tier), rig.ram);
  const o2 = { bits: rig.bits, kvBits: rig.kvBits, ctx: rig.ctx, spec: "none", workload: rig.workload };
  const pick = (id) => el("model-sort").children.find((b) => b.textContent === id).handlers.click();

  pick("Name");
  const names = gridIds().map((id) => byId2[id].name);
  assert(JSON.stringify(names) === JSON.stringify(names.slice().sort((a, b) => a.localeCompare(b))),
    "sorted by name: " + names.join(", "));

  pick("Speed");
  const speeds = gridIds().map((id) => estimateDecodeTps(byId2[id], hw2, o2));
  assert(speeds.every((v, i) => i === 0 || speeds[i - 1] >= v),
    "sorted by decode speed, fastest first: " + speeds.map((v) => v.toFixed(0)).join(", "));

  pick("Intelligence");
  const iqs = gridIds().map((id) => intelligenceFor(byId2[id], rig.bits).score);
  assert(iqs.every((v, i) => i === 0 || iqs[i - 1] >= v),
    "sorted by intelligence, best first: " + iqs.map((v) => v.toFixed(0)).join(", "));

  pick("Popular");
  assert(JSON.stringify(gridIds()) === JSON.stringify(unrankedOrder(activeModels).map((m) => m.id)),
    "popularity restores the board's own ordering");
  // the chosen sort survives a reopen — it is a preference, not a per-open reset
  pick("Name");
  closeModelModal();
  el("rig-model").handlers.click();
  assert(gridIds()[0] === byId2[gridIds()[0]].id &&
    JSON.stringify(gridIds().map((id) => byId2[id].name)) ===
    JSON.stringify(gridIds().map((id) => byId2[id].name).slice().sort((a, b) => a.localeCompare(b))),
    "the sort choice persists across opens");
  pick("Popular");

  // search
  el("model-modal-search").handlers.input({ target: { value: "gemma" } });
  assert(el("model-grid").children.length > 0 &&
         el("model-grid").children.every((c) => /gemma/.test(c.dataset.model)),
    "the filter narrows the grid");
  el("model-modal-search").handlers.input({ target: { value: "" } });

  // picking a card selects it and closes
  el("model-grid").children.find((c) => c.dataset.model === "gemma-4-12b").handlers.click();
  assert(el("model-modal").hidden === true, "choosing a card closes the modal");
  assert(!document.body.classList.contains("modal-open"), "scroll lock released");
  assert(rig.modelId === "gemma-4-12b", "the choice is applied");
  assert(el("rig-model-name").textContent === "Gemma 4 12B", "the trigger follows the choice");

  // cards re-score when the hardware changes
  el("rig-model").handlers.click();
  const before = el("model-grid").children.find((c) => c.dataset.model === "qwen3.6-27b").innerHTML;
  closeModelModal();
  el("rig-gen").children.find((b) => b.textContent === "M1").handlers.click();
  el("rig-tier").children.find((b) => b.textContent === "Base").handlers.click();
  el("rig-model").handlers.click();
  const after = el("model-grid").children.find((c) => c.dataset.model === "qwen3.6-27b").innerHTML;
  assert(before !== after, "cards are re-scored against the selected Mac, not cached");
  assert(/mcard nofit/.test(el("model-grid").children.find((c) => c.dataset.model === "qwen3.6-27b").className),
    "a 27B does not fit a 16 GB M1, and its card says so");

  // escape closes
  document.handlers.keydown({ key: "Escape" });
  assert(el("model-modal").hidden === true, "escape closes the modal");
  el("rig-model").handlers.click();
  el("model-modal-backdrop").handlers.click();
  assert(el("model-modal").hidden === true, "clicking the backdrop closes the modal");

  rig.gen = "M4"; rig.tier = "Max"; rig.ram = 64; rig.modelId = "qwen3.6-27b"; renderRig();
}

// ── NO LAYOUT SHIFT ───────────────────────────────────────────────────────
// The panel is a 3-column grid, so ANY column growing a line moves the whole
// board below it. The CSS reserves height for the blocks whose text varies;
// these bounds are what those reservations are sized for. Copy that outgrows
// them starts the jumping again, so the limits live here and not in a comment.
{
  // the footnote must not depend on the selection at all
  const foots = new Set();
  for (const gen of MAC_GENS) {
    DOM["rig-gen"].children.find((b) => b.textContent === gen).handlers.click();
    foots.add(DOM["rig-foot"].innerHTML);
  }
  assert(foots.size === 1, "the footnote is constant: conditional copy there resizes the whole page");

  // sweep the controls and bound every block that the CSS reserves space for
  const LIMITS = { "out-verdict": 132, "out-mem": 92, "rig-spec": 88, "out-iq-legend": 92 };
  const worst = {};
  for (const chip of MAC_CHIPS) {
    rig.gen = chip.gen; rig.tier = chip.tier;
    for (const ram of chip.ram) {
      rig.ram = ram;
      for (const mid of ["gemma-4-e2b", "qwen3.6-27b", "hy3", "deepseek-v4-flash"]) {
        rig.modelId = mid;
        for (const bits of QUANT_BITS) {
          for (const ctx of [4096, 131072]) {
            rig.bits = bits; rig.ctx = ctx; rig.kvBits = 16; rig.spec = "none";
            renderRig();
            for (const id of Object.keys(LIMITS)) {
              const t = DOM[id].textContent;
              if (!worst[id] || t.length > worst[id].length) worst[id] = t;
            }
          }
        }
      }
    }
  }
  for (const [id, cap] of Object.entries(LIMITS)) {
    assert(worst[id].length <= cap,
      id + " stays within its reserved height (" + worst[id].length + " > " + cap + " chars): " + worst[id]);
  }
}
console.log("playground wiring OK (renderRig drives " + Object.keys(DOM).length + " elements)");
`;

eval(pure + stub + rigSrc + rigAsserts);
