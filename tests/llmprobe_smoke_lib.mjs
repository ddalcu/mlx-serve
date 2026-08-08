// Pure decision layer for tests/llmprobe_smoke_test.sh.
//
// The shell script owns process lifecycle only. Everything that decides
// something lives here so it can be unit-tested without weights:
// tests/llmprobe_smoke_lib.test.mjs (`node --test`).
//
// It doubles as a small CLI so bash stays dumb:
//   node llmprobe_smoke_lib.mjs plan      --models <json> --dir <root> [...]
//   node llmprobe_smoke_lib.mjs summarize --rows <json>

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

// ── families ───────────────────────────────────────────────────────────────

/** Architectures the Zig MLX path serves (`model_discovery.supported_model_types`
 *  minus the engine-served and non-chat ones). GGUF routes to llama.cpp/ds4 and
 *  is deliberately out of scope. */
const MLX_CHAT_ARCHS = new Set([
  "gemma3", "gemma4", "gemma4_unified", "diffusion_gemma",
  "qwen2", "qwen3", "qwen3_5", "qwen3_5_moe", "qwen3_moe", "qwen3_next",
  "llama", "mistral", "lfm2", "nemotron_h", "hy_v3",
  // Ling-3.0-flash. Its MoE-ness comes from `is_moe` (the arch name carries no
  // `_moe` suffix), so familyKey lands it on `bailing_hybrid-moe` and it
  // inherits the MoE rules: MTP + drafter default OFF. Note the mirror is
  // 68.5 GB against SMOKE_MAX_GB's 60 default, so it needs the cap raised.
  "bailing_hybrid",
]);

/** Block-diffusion models denoise a canvas instead of decoding token-by-token,
 *  so no speculative path can ever apply to them (`modelBatchable` gates on
 *  `isDiffusion` for the same reason). */
const NON_SPECULATIVE_FAMILIES = new Set(["diffusion_gemma"]);

/** Speculative drafters exist for Gemma 4 only (`drafter.zig` cross-attends
 *  into the target's K/V; `error.UnsupportedDrafterArch` otherwise). */
const DRAFTER_FAMILIES = new Set(["gemma4"]);

const baseArch = (a = "") => a.replace(/_text$/, "");

/** Family = base arch + MoE-ness, which is the axis that actually changes
 *  behavior: MoE targets default MTP and the drafter OFF, and the drafter
 *  regresses on them outright. Two independent MoE markers, either one counts —
 *  the arch NAME (`qwen3_5_moe`) and the config's expert count (`is_moe`, which
 *  is how gemma-4-26B-A4B declares its 128 experts under a plain `gemma4`). */
export function familyKey(m) {
  const raw = baseArch(m?.meta?.architecture ?? "");
  const namedMoe = raw.endsWith("_moe");
  const base = namedMoe ? raw.slice(0, -"_moe".length) : raw;
  return namedMoe || m?.meta?.is_moe ? `${base}-moe` : base;
}

/** The family without its MoE suffix — for rules that key on the architecture
 *  itself (block diffusion is `diffusion_gemma`, and it is also an MoE). */
const familyBase = (f = "") => f.replace(/-moe$/, "");

export function isProbeable(m) {
  const caps = m?.capabilities ?? [];
  if (!caps.includes("chat")) return false; // encoders, media, unreadable stubs
  return MLX_CHAT_ARCHS.has(baseArch(m?.meta?.architecture ?? ""));
}

// ── model selection ────────────────────────────────────────────────────────

/**
 * One representative per family. Deterministic so a re-run probes the same
 * model and the numbers stay comparable: prefer a model that ships an MTP head
 * (it's the only one that can exercise the MTP cell at all), then the smallest
 * on disk (fastest to load).
 *
 * Anything dropped is reported, never silently swallowed.
 */
export function selectRepresentatives(models, opts = {}) {
  const { maxBytes = 60e9, only = null, collectSkipped = false } = opts;
  const skipped = [];
  const done = (reps) => (collectSkipped ? { reps, skipped } : reps);

  if (only?.length) {
    const wanted = new Set(only);
    const reps = models
      .filter((m) => wanted.has(m.id))
      .map((m) => ({ ...m, family: familyKey(m) }));
    for (const id of only) {
      if (!reps.some((r) => r.id === id)) skipped.push({ id, reason: "not in /v1/models" });
    }
    return done(reps);
  }

  const byFamily = new Map();
  for (const m of models) {
    if (!isProbeable(m)) {
      skipped.push({ id: m.id, reason: `not an MLX chat model (${m?.meta?.architecture ?? "?"})` });
      continue;
    }
    const bytes = m.bytes_on_disk ?? 0;
    if (maxBytes > 0 && bytes > maxBytes) {
      skipped.push({ id: m.id, reason: `too large (${(bytes / 1e9).toFixed(0)} GB > ${(maxBytes / 1e9).toFixed(0)} GB cap)` });
      continue;
    }
    const family = familyKey(m);
    const cand = { ...m, family };
    const cur = byFamily.get(family);
    if (!cur || betterRepresentative(cand, cur)) byFamily.set(family, cand);
  }

  for (const m of models) {
    if (!isProbeable(m)) continue;
    const chosen = byFamily.get(familyKey(m));
    if (chosen && chosen.id !== m.id && (m.bytes_on_disk ?? 0) <= maxBytes) {
      skipped.push({ id: m.id, reason: `same family as ${chosen.id}` });
    }
  }

  const reps = [...byFamily.values()].sort((a, b) => a.family.localeCompare(b.family));
  return done(reps);
}

function betterRepresentative(a, b) {
  if (!!a.hasMtp !== !!b.hasMtp) return !!a.hasMtp; // an MTP head unlocks a whole cell
  return (a.bytes_on_disk ?? 0) < (b.bytes_on_disk ?? 0);
}

// ── the cell matrix ────────────────────────────────────────────────────────

/**
 * The launch-flag cells for one representative. llmprobe sends no spec/KV body
 * fields, so every cell is a separate server boot.
 *
 * Spec sweep (KV dense): none / pld / mtp / drafter — only the applicable ones.
 * KV sweep: a crash check on ONE fast model, run at its DEFAULT spec config
 * (that's the combination a user actually runs, so a crash there is the one
 * that matters).
 */
export function planCells(rep, opts = {}) {
  const { kvCheck = false } = opts;

  // `expectNoSpec`: this cell turned speculation OFF, so the server must report
  // zero `[spec-stats]` lines. It is the tripwire for "the flags never landed"
  // — a cell running the wrong config still passes conformance and still posts
  // a plausible tok/s, so nothing else in this harness would notice.
  if (NON_SPECULATIVE_FAMILIES.has(familyBase(rep.family))) {
    return [{
      id: "none",
      label: "canvas (no spec path exists)",
      flags: ["--no-pld", "--no-mtp"],
      expectSpec: null,
      expectNoSpec: true,
    }];
  }

  const cells = [
    // PLD is default-ON at the CLI, so a "no speculation" baseline needs BOTH
    // flags — otherwise it isn't a baseline at all.
    { id: "none", label: "no speculation", flags: ["--no-pld", "--no-mtp"], expectSpec: null, expectNoSpec: true },
    { id: "pld", label: "PLD", flags: ["--pld", "--no-mtp"], expectSpec: "pld" },
  ];

  if (rep.hasMtp) {
    cells.push({
      id: "mtp",
      label: "native MTP head",
      // --mtp forces the head on for MoE targets, where the per-request default
      // is OFF; on a dense target it changes nothing.
      flags: ["--no-pld", "--mtp"],
      expectSpec: "mtp",
    });
  }
  if (DRAFTER_FAMILIES.has(rep.family) && rep.drafterPath) {
    cells.push({
      id: "drafter",
      label: "assistant drafter",
      flags: ["--no-pld", "--no-mtp", "--drafter", rep.drafterPath],
      expectSpec: "drafter",
    });
  }
  if (kvCheck) {
    cells.push(
      { id: "kv4", label: "KV 4-bit (crash check)", flags: ["--kv-quant", "4"], expectSpec: null, kv: true },
      { id: "kv-turbo4", label: "KV TurboQuant-4 (crash check)", flags: ["--kv-quant", "turbo4"], expectSpec: null, kv: true },
    );
  }
  return cells;
}

// ── engagement ─────────────────────────────────────────────────────────────

/** Count `[spec-stats] mode=X` lines per mode. Throughput alone cannot prove a
 *  spec path ran — the regular-decode fallback is output-identical and often
 *  similar in speed, which is exactly how two call sites shipped a hardcoded
 *  `use_drafter=false` for a month. */
export function parseSpecStats(log = "") {
  const counts = { mtp: 0, pld: 0, drafter: 0 };
  for (const m of log.matchAll(/\[spec-stats\]\s+mode=(\w+)/g)) {
    if (m[1] in counts) counts[m[1]] += 1;
  }
  return counts;
}

/**
 * Did this cell do what it claimed?
 *
 * MTP is NOT subject to the n-gram prompt gate (the trained head holds its
 * acceptance rate on novel content), so zero engagements means a dispatch hole
 * — hard fail. PLD and the drafter ARE gated, and llmprobe's prompts are mostly
 * novel, so zero engagements there is correct behavior: report, never fail.
 */
export function cellVerdict(cell, result, stats = {}) {
  const reasons = [];
  let ok = true;

  if (result?.error) {
    return { ok: false, reasons: [result.error] };
  }

  const conf = result?.conformance;
  if (conf && conf.total > 0 && conf.passed < conf.total) {
    ok = false;
    reasons.push(`conformance ${conf.passed}/${conf.total}`);
  }

  if (cell.expectNoSpec) {
    const leaked = Object.entries(stats).filter(([, n]) => n > 0);
    if (leaked.length) {
      ok = false;
      reasons.push(
        `speculation ran in a cell that disabled it (${leaked.map(([k, n]) => `${k}×${n}`).join(" ")}) ` +
          `— the launch flags did not reach the server, so every number in this row is for the wrong config`,
      );
    }
  }

  if (cell.expectSpec) {
    const n = stats[cell.expectSpec] ?? 0;
    if (n === 0) {
      if (cell.expectSpec === "mtp") {
        ok = false;
        reasons.push("MTP never engaged (no [spec-stats] mode=mtp) — dispatch hole or the head didn't load");
      } else {
        reasons.push(`${cell.expectSpec} never engaged — expected: the n-gram spec-gate disables it on novel prompts`);
      }
    } else {
      reasons.push(`${cell.expectSpec} engaged ×${n}`);
    }
  }
  return { ok, reasons };
}

// ── roll-up ────────────────────────────────────────────────────────────────

const decodeOf = (r) => r?.result?.bench?.decodeTokPerSec?.median ?? null;

/** Birds-eye: per family, the fastest cell and what it bought over the
 *  speculation-free baseline, plus any cell whose conformance dropped relative
 *  to that family's own baseline (a spec/KV path that changed OUTPUT, not just
 *  speed — the thing this harness exists to catch). */
export function buildSummary(rows) {
  const families = [];
  const regressions = [];

  for (const family of [...new Set(rows.map((r) => r.family))]) {
    const mine = rows.filter((r) => r.family === family);
    const base = mine.find((r) => r.cell === "none");
    const baseConf = base?.result?.conformance;

    for (const r of mine) {
      const c = r.result?.conformance;
      if (!baseConf || !c || r === base) continue;
      if (c.total > 0 && baseConf.total > 0 && c.passed < baseConf.passed) {
        regressions.push(
          `${family} · ${r.cell}: conformance ${baseConf.passed} → ${c.passed} of ${c.total} (vs the 'none' baseline)`,
        );
      }
    }

    const scored = mine.filter((r) => decodeOf(r) !== null);
    const best = scored.sort((a, b) => decodeOf(b) - decodeOf(a))[0];
    const baseDecode = decodeOf(base);
    families.push({
      family,
      model: mine[0]?.model ?? null,
      cells: mine.length,
      bestCell: best?.cell ?? null,
      bestDecode: best ? decodeOf(best) : null,
      baseDecode,
      speedup:
        best && baseDecode ? Math.round((decodeOf(best) / baseDecode) * 100) / 100 : null,
      conformance: baseConf ?? mine.find((r) => r.result?.conformance)?.result?.conformance ?? null,
      capability: base?.result?.capability?.pct ?? null,
      failed: mine.filter((r) => r.verdict && !r.verdict.ok).map((r) => r.cell),
    });
  }
  return { families, regressions };
}

// ── on-disk companions ─────────────────────────────────────────────────────

const MTP_SIDECARS = ["mtp/weights.safetensors", "mtp.safetensors", "model-mtp.safetensors"];

/** Mirrors `mtp.resolveMtpSidecarInDir`: first present, NON-EMPTY file wins (a
 *  zero-byte file is a failed download, not a head). */
export function hasMtpSidecar(dir) {
  return MTP_SIDECARS.some((rel) => {
    try {
      return statSync(join(dir, rel)).size > 0;
    } catch {
      return false;
    }
  });
}

/** `gemma4_assistant` dirs are excluded from /v1/models by design (they can't
 *  decode on their own), so the drafter has to be found on disk. */
export function findDrafters(root) {
  const out = [];
  const cfgType = (dir) => {
    try {
      return JSON.parse(readFileSync(join(dir, "config.json"), "utf8")).model_type ?? "";
    } catch {
      return "";
    }
  };
  const scan = (dir, depth) => {
    if (depth > 2 || !existsSync(dir)) return;
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      if (!e.isDirectory()) continue;
      const child = join(dir, e.name);
      if (cfgType(child).endsWith("_assistant")) out.push(child);
      else scan(child, depth + 1);
    }
  };
  scan(root, 0);
  return out.sort();
}

/** Pair a Gemma 4 target with its assistant on the size token (E2B / E4B / 12B
 *  / 31B / 26B-A4B). A wrong guess is not silent: the server rejects it with
 *  `error.DrafterTargetMismatch` at load. */
export function pickDrafterFor(modelId, drafters) {
  const size = sizeToken(modelId);
  if (!size) return null;
  return drafters.find((d) => sizeToken(d) === size) ?? null;
}

function sizeToken(s) {
  const m = /(?:^|[/\-_])(e2b|e4b|\d{1,3}b(?:-a\d+b)?)(?:[-_]|$)/i.exec(
    s.replace(/-it-assistant.*$/i, "-it").replace(/-it-.*$/i, "-it"),
  );
  return m ? m[1].toLowerCase() : null;
}

// ── CLI ────────────────────────────────────────────────────────────────────

function main(argv) {
  const cmd = argv[0];
  const flag = (name, def = null) => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def;
  };
  const readJson = (p) => JSON.parse(readFileSync(p, "utf8"));

  if (cmd === "plan") {
    const root = flag("dir");
    const models = readJson(flag("models")).data ?? [];
    const only = (flag("only") ?? "").split(",").map((s) => s.trim()).filter(Boolean);
    const maxBytes = Number(flag("max-bytes", 60e9));
    const drafters = findDrafters(root);

    const annotated = models.map((m) => {
      const dir = join(root, m.id);
      return {
        ...m,
        dir,
        hasMtp: hasMtpSidecar(dir),
        drafterPath: pickDrafterFor(m.id, drafters),
      };
    });

    const { reps, skipped } = selectRepresentatives(annotated, {
      maxBytes,
      only,
      collectSkipped: true,
    });
    // The KV crash check runs on ONE model: the fastest to load.
    const kvId = reps.length
      ? [...reps].sort((a, b) => (a.bytes_on_disk ?? 0) - (b.bytes_on_disk ?? 0))[0].id
      : null;

    const plan = reps.map((rep) => ({
      id: rep.id,
      dir: rep.dir,
      family: rep.family,
      bytes: rep.bytes_on_disk ?? 0,
      cells: planCells(rep, { kvCheck: rep.id === kvId }),
    }));
    process.stdout.write(JSON.stringify({ plan, skipped }, null, 2));
    return;
  }

  if (cmd === "summarize") {
    process.stdout.write(JSON.stringify(buildSummary(readJson(flag("rows"))), null, 2));
    return;
  }

  process.stderr.write(`usage: llmprobe_smoke_lib.mjs plan|summarize [...]\n`);
  process.exit(2);
}

if (process.argv[1]?.endsWith("llmprobe_smoke_lib.mjs")) main(process.argv.slice(2));
