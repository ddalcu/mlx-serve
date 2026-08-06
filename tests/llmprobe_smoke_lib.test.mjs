// Unit tests for the pure decision layer of tests/llmprobe_smoke_test.sh.
//
// The shell script owns only process lifecycle (boot a server, run npx
// llmprobe, kill it). Everything that DECIDES something — which models are
// probeable, which family they belong to, which spec/KV cells to run, whether
// a cell actually engaged the spec path it claimed — lives here, where it can
// be tested without weights.
//
//   node --test tests/llmprobe_smoke_lib.test.mjs

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  familyKey,
  isProbeable,
  selectRepresentatives,
  planCells,
  parseSpecStats,
  cellVerdict,
  buildSummary,
  findDrafters,
  pickDrafterFor,
  hasMtpSidecar,
} from "./llmprobe_smoke_lib.mjs";

const model = (id, arch, extra = {}) => ({
  id,
  capabilities: ["chat", "tool_use", "streaming", "json_schema"],
  bytes_on_disk: 8e9,
  meta: { architecture: arch, is_moe: false, ...(extra.meta ?? {}) },
  ...extra,
});

// ── family grouping ────────────────────────────────────────────────────────

test("familyKey folds _text siblings onto the base arch and splits MoE out", () => {
  assert.equal(familyKey(model("a", "gemma4")), "gemma4");
  assert.equal(familyKey(model("a", "gemma4_text")), "gemma4");
  assert.equal(familyKey(model("a", "qwen3_5_moe_text")), "qwen3_5-moe");
  // is_moe is authoritative — a dense-named arch with experts is still MoE
  // (gemma4 26B-A4B is exactly this: model_type "gemma4", 128 experts).
  assert.equal(
    familyKey(model("a", "gemma4", { meta: { architecture: "gemma4", is_moe: true } })),
    "gemma4-moe",
  );
  // Ling is the same shape: `bailing_hybrid` carries no _moe suffix, so only
  // is_moe puts it in the MoE bucket — and the MoE bucket is what defaults its
  // MTP and drafter cells OFF.
  assert.equal(
    familyKey(model("a", "bailing_hybrid", { meta: { architecture: "bailing_hybrid", is_moe: true } })),
    "bailing_hybrid-moe",
  );
});

// ── what we probe ──────────────────────────────────────────────────────────

test("isProbeable takes MLX chat models only — no gguf, ds4, encoders or media", () => {
  assert.ok(isProbeable(model("ok", "gemma4")));

  assert.ok(isProbeable(model("ling", "bailing_hybrid")));

  // Engine-served GGUF (llama.cpp / ds4) — out of scope by request.
  assert.ok(!isProbeable(model("g", "gguf")));
  assert.ok(!isProbeable(model("d", "deepseek_v4")));
  // Encoder-only: no chat surface at all.
  assert.ok(
    !isProbeable({ id: "bge", capabilities: ["embeddings"], meta: { architecture: "bert" } }),
  );
  // Media models: capabilities carry the modality, never "chat".
  assert.ok(!isProbeable({ id: "flux", capabilities: ["image"], meta: { architecture: "flux2" } }));
  assert.ok(!isProbeable({ id: "tts", capabilities: ["audio"], meta: { architecture: "qwen3_tts" } }));
  // A stub whose config.json couldn't be read has no capabilities — skip it
  // rather than boot a server to find out.
  assert.ok(!isProbeable({ id: "mystery", meta: {} }));
});

test("selectRepresentatives picks one model per family, skipping the giants", () => {
  const models = [
    model("mlx-community/gemma-4-e4b-it-4bit", "gemma4", { bytes_on_disk: 4e9 }),
    model("mlx-community/gemma-4-31b-it-4bit", "gemma4", { bytes_on_disk: 18e9 }),
    model("mlx-community/gemma-3-12b-it-4bit", "gemma3", { bytes_on_disk: 7e9 }),
    model("ox-ox/Hy3-295B", "hy_v3", { bytes_on_disk: 180e9 }),
    model("x/gguf-thing", "gguf"),
  ];
  const reps = selectRepresentatives(models, { maxBytes: 60e9 });
  const ids = reps.map((r) => r.id);

  assert.deepEqual(ids, [
    "mlx-community/gemma-3-12b-it-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
  ]);
  // Deterministic tie-break: smallest on disk wins, so a re-run probes the
  // same model and the numbers stay comparable.
  assert.equal(reps.find((r) => r.family === "gemma4").id, "mlx-community/gemma-4-e4b-it-4bit");
  // The giant is excluded by size, not by arch — it must be reported as
  // skipped, never silently dropped (no-silent-caps rule).
  const skipped = selectRepresentatives(models, { maxBytes: 60e9, collectSkipped: true }).skipped;
  assert.ok(skipped.some((s) => s.id === "ox-ox/Hy3-295B" && /too large/.test(s.reason)));
});

test("selectRepresentatives prefers a model that ships an MTP head", () => {
  const models = [
    model("org/qwen-plain", "qwen3_5", { bytes_on_disk: 4e9, hasMtp: false }),
    model("org/qwen-mtp", "qwen3_5", { bytes_on_disk: 16e9, hasMtp: true }),
  ];
  const [rep] = selectRepresentatives(models, { maxBytes: 60e9 });
  // Bigger, but it's the only one that can exercise the MTP cell at all.
  assert.equal(rep.id, "org/qwen-mtp");
});

test("selectRepresentatives honours an explicit id list verbatim", () => {
  const models = [model("a/one", "gemma4"), model("b/two", "qwen3_5")];
  const reps = selectRepresentatives(models, { maxBytes: 60e9, only: ["b/two"] });
  assert.deepEqual(reps.map((r) => r.id), ["b/two"]);
});

// ── the cell matrix ────────────────────────────────────────────────────────

test("planCells: baseline is spec-free — PLD is default-ON server-side", () => {
  const cells = planCells(
    { id: "m", family: "gemma4", hasMtp: false, drafterPath: null },
    { kvCheck: false },
  );
  const none = cells.find((c) => c.id === "none");
  // Without BOTH flags the "no speculation" cell would still be running PLD.
  assert.deepEqual(none.flags, ["--no-pld", "--no-mtp"]);
  assert.equal(none.expectSpec, null);
  // ...and it must PROVE it: a cell that disables speculation and still shows
  // engagements means its flags never reached the server.
  assert.equal(none.expectNoSpec, true);

  // The KV cells deliberately run at the model's default spec config, so
  // engagements there are expected — they must NOT carry the assertion.
  const kv = planCells({ id: "m", family: "gemma4" }, { kvCheck: true }).find((c) => c.id === "kv4");
  assert.ok(!kv.expectNoSpec);
});

test("cellVerdict fails a spec-free cell that engaged anyway — the flags-didn't-land guard", () => {
  const ok = { conformance: { passed: 89, total: 89 } };

  // Live bug this pins: an IFS tab-collapse ate the flags, the 'none' cell
  // booted with the server's defaults (PLD on), and every number in the row was
  // secretly a PLD number. Conformance and throughput both looked perfect.
  const leaked = cellVerdict(
    { id: "none", expectSpec: null, expectNoSpec: true },
    ok,
    { mtp: 0, pld: 7, drafter: 0 },
  );
  assert.equal(leaked.ok, false);
  assert.match(leaked.reasons.join(" "), /pld.*7.*--no-pld|flags/i);

  const clean = cellVerdict(
    { id: "none", expectSpec: null, expectNoSpec: true },
    ok,
    { mtp: 0, pld: 0, drafter: 0 },
  );
  assert.equal(clean.ok, true);
});

test("planCells: MTP cell only when a sidecar exists, and it passes --mtp", () => {
  const without = planCells({ id: "m", family: "qwen3_5", hasMtp: false }, {});
  assert.ok(!without.some((c) => c.id === "mtp"));

  const cells = planCells({ id: "m", family: "qwen3_5-moe", hasMtp: true }, {});
  const mtp = cells.find((c) => c.id === "mtp");
  // --mtp is what makes the cell real on a MoE target: without it the server
  // defaults enable_mtp to false and llmprobe (which sends no spec fields)
  // would silently measure plain decode.
  assert.ok(mtp.flags.includes("--mtp"));
  assert.ok(mtp.flags.includes("--no-pld"));
  assert.equal(mtp.expectSpec, "mtp");
});

test("planCells: drafter cell only for a Gemma 4 target with a matching assistant", () => {
  const noDrafter = planCells({ id: "m", family: "gemma4", drafterPath: null }, {});
  assert.ok(!noDrafter.some((c) => c.id === "drafter"));

  const cells = planCells({ id: "m", family: "gemma4", drafterPath: "/d/assistant" }, {});
  const d = cells.find((c) => c.id === "drafter");
  assert.deepEqual(d.flags, ["--no-pld", "--no-mtp", "--drafter", "/d/assistant"]);
});

test("planCells: block-diffusion models never speculate — baseline cell only", () => {
  const cells = planCells(
    { id: "m", family: "diffusion_gemma", hasMtp: true, drafterPath: "/d" },
    { kvCheck: true },
  );
  assert.deepEqual(cells.map((c) => c.id), ["none"]);
});

test("planCells: the KV-quant cells are a crash check on one model, at its real config", () => {
  const plain = planCells({ id: "m", family: "gemma4" }, { kvCheck: false });
  assert.ok(!plain.some((c) => c.id.startsWith("kv")));

  const cells = planCells({ id: "m", family: "gemma4" }, { kvCheck: true });
  const kv4 = cells.find((c) => c.id === "kv4");
  const turbo = cells.find((c) => c.id === "kv-turbo4");
  // No --no-pld/--no-mtp: the point is "does the server survive the config a
  // user actually runs", so speculation stays at its defaults.
  assert.deepEqual(kv4.flags, ["--kv-quant", "4"]);
  assert.deepEqual(turbo.flags, ["--kv-quant", "turbo4"]);
});

// ── engagement: perf alone cannot prove a spec path ran ─────────────────────

test("parseSpecStats counts engagements per mode from the server log", () => {
  const log = [
    "[spec-stats] mode=mtp attempts=12 accepted=31 drafted=42 per_draft=73.8%",
    "some other line",
    "[spec-stats] mode=mtp attempts=8 accepted=19",
    "[spec-stats] mode=pld attempts=3",
    "spec-gate: ngram-score=0.004 (novel) -> pld disabled",
  ].join("\n");
  assert.deepEqual(parseSpecStats(log), { mtp: 2, pld: 1, drafter: 0 });
});

test("cellVerdict fails a spec cell that never engaged — the dispatch-hole guard", () => {
  const ok = { conformance: { passed: 40, total: 40 } };

  // MTP is NOT subject to the n-gram prompt gate, so zero engagements on a
  // cell that asked for it means the request never reached the head.
  const dead = cellVerdict(
    { id: "mtp", expectSpec: "mtp" },
    ok,
    { mtp: 0, pld: 0, drafter: 0 },
  );
  assert.equal(dead.ok, false);
  assert.match(dead.reasons.join(" "), /never engaged/);

  const live = cellVerdict({ id: "mtp", expectSpec: "mtp" }, ok, { mtp: 5, pld: 0, drafter: 0 });
  assert.equal(live.ok, true);

  // PLD and the drafter ARE gated on the prompt's n-gram score, and llmprobe's
  // prompts are mostly novel — zero engagements there is expected behavior,
  // not a bug. Report it, never fail on it.
  const gated = cellVerdict({ id: "pld", expectSpec: "pld" }, ok, { mtp: 0, pld: 0, drafter: 0 });
  assert.equal(gated.ok, true);
  assert.match(gated.reasons.join(" "), /gate/);
});

test("cellVerdict fails a crashed or non-conforming cell", () => {
  const crashed = cellVerdict({ id: "kv4" }, { error: "server exited (rc=139)" }, {});
  assert.equal(crashed.ok, false);
  assert.match(crashed.reasons.join(" "), /rc=139/);

  const broken = cellVerdict({ id: "none" }, { conformance: { passed: 38, total: 40 } }, {});
  assert.equal(broken.ok, false);
  assert.match(broken.reasons.join(" "), /38\/40/);
});

// ── roll-up ────────────────────────────────────────────────────────────────

test("buildSummary flags a cell whose conformance regressed against its own baseline", () => {
  const rows = [
    {
      family: "gemma4",
      cell: "none",
      result: { conformance: { passed: 40, total: 40 }, bench: { decodeTokPerSec: { median: 50 } } },
    },
    {
      family: "gemma4",
      cell: "kv-turbo4",
      result: { conformance: { passed: 37, total: 40 }, bench: { decodeTokPerSec: { median: 55 } } },
    },
  ];
  const s = buildSummary(rows);
  assert.equal(s.regressions.length, 1);
  assert.match(s.regressions[0], /gemma4.*kv-turbo4.*40 → 37/);
});

test("buildSummary names the fastest cell per family — the birds-eye answer", () => {
  const rows = [
    { family: "qwen3_5-moe", cell: "none", result: { bench: { decodeTokPerSec: { median: 60 } } } },
    { family: "qwen3_5-moe", cell: "mtp", result: { bench: { decodeTokPerSec: { median: 79.5 } } } },
    { family: "qwen3_5-moe", cell: "pld", result: { bench: { decodeTokPerSec: { median: 58 } } } },
  ];
  const s = buildSummary(rows);
  const f = s.families.find((x) => x.family === "qwen3_5-moe");
  assert.equal(f.bestCell, "mtp");
  assert.equal(f.bestDecode, 79.5);
  assert.equal(f.speedup, 1.33); // vs the spec-free baseline
});

// ── on-disk companions (drafters + MTP sidecars) ────────────────────────────

test("findDrafters / pickDrafterFor pair a Gemma 4 target with its assistant", () => {
  const root = mkdtempSync(join(tmpdir(), "smoke-"));
  const put = (rel, cfg) => {
    mkdirSync(join(root, rel), { recursive: true });
    writeFileSync(join(root, rel, "config.json"), JSON.stringify(cfg));
  };
  put("mlx-community/gemma-4-E4B-it-assistant-bf16", { model_type: "gemma4_assistant" });
  put("mlx-community/gemma-4-31B-it-assistant-bf16", { model_type: "gemma4_assistant" });
  put("mlx-community/gemma-4-e4b-it-4bit", { model_type: "gemma4" });

  const drafters = findDrafters(root);
  assert.equal(drafters.length, 2);

  // Matched on the size token, case-insensitively (E4B vs e4b) — and the
  // server still validates the pairing, so a wrong guess errors loudly.
  const d = pickDrafterFor("mlx-community/gemma-4-e4b-it-4bit", drafters);
  assert.match(d, /E4B-it-assistant/);
  assert.equal(pickDrafterFor("mlx-community/gemma-3-12b-it-4bit", drafters), null);
  assert.equal(pickDrafterFor("mlx-community/gemma-4-12B-it-4bit", drafters), null);
});

test("hasMtpSidecar accepts every filename the loader accepts, and ignores empty files", () => {
  const root = mkdtempSync(join(tmpdir(), "smoke-"));
  const dir = (rel) => {
    mkdirSync(join(root, rel), { recursive: true });
    return join(root, rel);
  };

  const native = dir("a");
  mkdirSync(join(native, "mtp"));
  writeFileSync(join(native, "mtp", "weights.safetensors"), "x");
  assert.ok(hasMtpSidecar(native));

  const alt = dir("b");
  writeFileSync(join(alt, "model-mtp.safetensors"), "x");
  assert.ok(hasMtpSidecar(alt));

  assert.ok(!hasMtpSidecar(dir("c")));

  // A zero-byte file is a failed download, not a head (mirrors
  // mtp.resolveMtpSidecarInDir's `st.size > 0`).
  const empty = dir("d");
  writeFileSync(join(empty, "mtp.safetensors"), "");
  assert.ok(!hasMtpSidecar(empty));
});
