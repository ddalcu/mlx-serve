---
name: plugin
description: Create, build, test, install, update and remove an mlx-serve plugin, EXTERNAL (an outside dylib built from the plugin kit, local or shared via CommunityServe/plugins) or INTERNAL (a folder under src/plugins/ in this repo, PR'd to mlx-serve). Use when asked to make a plugin, add or port a model, media backend or quant format, install or try a plugin, or debug why a plugin did not load or did not serve a pack.
---

## Plugins: external or internal?

**FIRST, when this skill is invoked, explain the two kinds to the user in plain words and ask which one they want
(unless they already said).** Do not start building before that is settled.

| | External plugin | Internal plugin |
|---|---|---|
| What | A dylib built OUTSIDE this repo from the plugin kit | A folder in this repo: `src/plugins/<name>/` |
| Where it lives | The author's machine (`~/.mlx-serve/plugins/`), or the `CommunityServe/plugins` repo | The mlx-serve codebase itself |
| Availability | Only where installed; the user turns plugins on (Settings ▸ Server, or `plugins.json`) | Always built in, on for every user, every release |
| Sharing | Stay local, or open a PR to https://github.com/CommunityServe/plugins so others can discover it | PR to https://github.com/ddalcu/mlx-serve |
| Review | Community repo's own bar (its `AGENTS.md`) | Rigorous code review by the maintainers; can be DENIED |
| Bar to meet | Passes `plugin check`; refuses what it cannot serve | Everything in `AGENTS.md` + `src/plugins/AGENTS.md`: TDD, parity oracle, kernel contracts, lint 0, bf16 + quantized packs, app catalog |
| Release cycle | Yours | mlx-serve's |
| Iteration speed | Extremely quick: only the plugin's own small dylib rebuilds (seconds), mlx-serve is never recompiled | Slower: a plugin edit rebuilds its unit group (~30 s with `-Dunits`, ~3.5 min without) |
| Pick it when | Trying a model or format, personal use, niche or experimental, no wish to carry it in core | A mainstream model family or backend that belongs in mlx-serve for everyone, and you accept the review |

A good default: **start external** (the big benefit: extremely quick iteration, and nobody recompiles the whole
mlx-serve codebase; no review either), and promote to internal later if it proves broadly
useful. The code is a similar shape: external plugins may call `kit.blocks` (a subset of `src/sdk/blocks/`, versioned
with the kit), in-tree plugins use all of `src/sdk/`.

- Building an **external** plugin: Part A below (sections 1-5 and the failure table).
- Building an **internal** plugin: Part B, after Part A.

Read first for either: `docs/plugins.md` (seams, rules), the guide for the kind (`docs/plugin-model.md`,
`plugin-media.md`, `plugin-quant.md`) and `docs/plugin-abi.md` (every table and field, generated from `src/abi/`).

---

# Part A: external plugin (outside dylib)

An outside plugin is a dylib built from the plugin kit; the server loads it at startup and it serves the packs it
claims. To share it, open a PR to https://github.com/CommunityServe/plugins (its `AGENTS.md` is the contract
there); otherwise it just stays on the user's machine.

macOS only. The plugin's MLX pin must equal the server's (`mlx.version` in the kit): build against the kit of the
mlx-serve that will load it, or it is refused at load.

### 1. Stage the kit and scaffold

```
./.zig-toolchain/zig build kit -j1        # zig-out/kit/mlx-serve-plugin-kit-<version>/ (+ tarball), seconds
K=zig-out/kit/mlx-serve-plugin-kit-<version>
./zig-out/bin/mlx-serve plugin new <name> [--media|--quant] --kit $K --dir ~/work
```

Restage after any change to `src/abi/` or the SDK files the kit carries. Release users get the same kit as the
`mlx-serve-plugin-kit` tarball (`~/.mlx-serve/kit` or `MLX_SERVE_KIT`). Community plugins live in
`CommunityServe/plugins` (its `scripts/setup.sh`, `new-plugin.sh`, `build.sh`, `check.sh`, `install.sh`; its
`AGENTS.md` is the contract there).

The folder: `config.zig` (claims + parse, MLX-free, no GPU), `plugin.zig` (load, bill, forward, fixture),
`build.zig`. The example is a tiny dense Llama-shaped decoder: a fixture, not a loader for real packs.

### 2. Claim, then write the plugin

- **Claim narrowly.** `.by_marker` beats `.by_type`; a built-in wins a tie. To take some packs of a `model_type` the
  server already serves, claim `.by_marker` on a config key only those packs carry (`sushi_ds41`: `deepseek_v41` with
  `expert_quant.format == "stq"`) and return `.no` for the rest. `.by_type` alone only works for a new `model_type`.
  Naming a plugin like a built-in replaces that built-in for all its packs: never do that by accident.
- **What the kit gives you:** the ABI (`src/abi/`), claim helpers and test fixtures (`kit.sdk`), `kit.generation`,
  `kit.wav`, mlx-c (`@import("mlx_c")`). **`kit.blocks`**: a subset of the server's own layer blocks
  (norms, conv, ssm, schedulers, vae, dit, embedding, linear, qmm, mlp, rope, attention, moe, hc, gdn; Metal only).
  Your own weight layout joins `qmatmul` and the expert matmul through `kit.foreign.register`. They are versioned
  with the kit, so pin it. They take `kit.mlx` handles (`{ ctx }`): convert with `.{ .ctx = x.ctx }`, and a block that
  declines returns null (keep your own ops: see `_example`'s `addNorm`). Otherwise write layers on mlx-c
  (`mlx_fast_rms_norm`,
  `mlx_fast_rope`, `mlx_fast_scaled_dot_product_attention`, `mlx_quantized_matmul`, `mlx_gather_qmm`,
  `mlx_fast_metal_kernel_*`), or vendor code with a `//! copied_from: <path>@<sha>` header and provenance in NOTICE.
- **Hooks beyond a plain decoder** are in `Parsed`/`ModelTable` (`docs/plugin-abi.md`): `SpecHooks.head` (MTP heads),
  `SpecHooks.native_drafter` (a drafter the plugin runs itself), `WeightHooks` (`keep_key`, `resident_bytes`),
  `MediaHooks` (image/audio towers), `SpeechHooks`, `CacheHooks`. `LoadCtx.weights == null` = the plugin reads its
  own shards from `LoadCtx.model_dir`.
- **Rules:** refuse an unsupported config or layout by name (a `Verdict` + `log.err`), never fall back to another
  arch; config JSON is untrusted, bounds-check it; bill every byte you allocate (`bill`, `resident_bytes`, upper
  bounds you measured); no new CLI flags or env levers; weight names and config keys follow mlx-lm.
- **Tests:** `config.zig` claim/parse tests run MLX-free: the packs it must claim, the ones it must leave
  (`.no`), malformed JSON. Red first.

### 3. Build and check

```
cd ~/work/<name>
~/code/mlx-serve/.zig-toolchain/zig build test -j1
~/code/mlx-serve/.zig-toolchain/zig build -Doptimize=ReleaseFast -j1   # zig-out/lib/lib<name>.dylib
cd ~/code/mlx-serve
./zig-out/bin/mlx-serve plugin check ~/work/<name>                     # conformance on the plugin's fixture
./zig-out/bin/mlx-serve plugin check ~/work/<name> --model <pack dir>  # + does it claim and parse that pack
./zig-out/bin/mlx-serve plugin fixture ~/work/<name> <dir>             # its tiny pack, to serve
```

`plugin check --model` prints `<name> claims and parses <dir>: ... (by_marker|by_type)`: confirm the rank you meant.
Passing proves the contract, not the numerics.

### 4. Install and serve under a throwaway HOME (never the user's)

The app owns `~/.mlx-serve/plugins.json` and writes it back, and a real install changes what the user's server
loads. Test like this:

```
H=~/claude-tmp/plug-home; mkdir -p $H/.mlx-serve
HOME=$H ./zig-out/bin/mlx-serve plugin install ~/work/<name>/zig-out/lib/lib<name>.dylib --yes
printf '{"enabled": true, "disabled": []}\n' > $H/.mlx-serve/plugins.json
HOME=$H ./zig-out/bin/mlx-serve --serve --host 127.0.0.1 --port 18082 --model <pack dir>     # plugin on
HOME=$H ./zig-out/bin/mlx-serve --serve --host 127.0.0.1 --port 18083 --model <pack dir> --no-plugins
```

- Pass `--model` as an absolute path (the throwaway HOME has no models). For a real pack: check free RAM, resident
  models and GPU first, one load at a time, never both arms at once; stop each server by its PID.
- Proof it served: `[plugins] loaded <name> (model, sha256 …)` then `[plugin] <name> serves <model_type> (N layers)` in
  the log (`$H/.mlx-serve/logs/mlx-serve-<port>.log`), and `"plugin": "<name>"` on the pack's `/v1/models` entry.
- A/B: greedy (`temperature: 0`) the same prompt on both arms and diff. A pack only the plugin can serve fails the
  `--no-plugins` arm by name (`PluginDeclinedNoLegacy`); a pack the built-in also serves should match it.

### 5. Install for the user (only when asked)

`./zig-out/bin/mlx-serve plugin install <dylib>` (shows hash and signer; `--yes` confirms a quarantined download)
writes `~/.mlx-serve/plugins/<name>.dylib` + `<name>.json` (the receipt). The user turns plugins on in the app
(Settings ▸ Server, a switch per plugin) or, with the app quit, `"enabled": true` in `~/.mlx-serve/plugins.json`.
Plugins load at server start and are never unloaded: restart the server (in-app restart, or `FAST_DEV=1 bash
app/build.sh ko` when rebuilding the app anyway).

- **Update:** rebuild, `plugin install` again (the hash changes; a stale receipt makes the load refuse), restart.
- **Remove:** delete `~/.mlx-serve/plugins/<name>.dylib` and `<name>.json` (ask first), restart.
- **Off for one run:** `--no-plugins`.

### When it does not work

| Symptom | Cause |
|---|---|
| no `[plugins] loaded` line | plugins off (`plugins.json` `enabled` false, or the name in `disabled`), or `--no-plugins` |
| `[plugins] refused <path>: …` | hash ≠ receipt (reinstall), built against another MLX pin (rebuild against this kit), not a plugin dylib |
| loaded, but the built-in serves the pack | the claim is `.by_type` against a built-in's type (tie → built-in): claim `.by_marker` |
| load error `Ambiguous` | two outside plugins claim the pack at the same rank |
| `plugin check` passes, real pack fails | the example is a fixture: real dtype, untied head, quantized weights need real loading code |
| `zig build test` prints `failed command:` | read the exit code; the runner echoes the command even on success |

### Sharing an external plugin

Local only is fine: nothing more to do after section 5. To make it discoverable, PR it to
https://github.com/CommunityServe/plugins using that repo's scripts (`scripts/setup.sh`, `new-plugin.sh`,
`build.sh`, `check.sh`, `install.sh`) and its `AGENTS.md`. No mlx-serve review is involved; that repo's maintainers
decide.

---

# Part B: internal plugin (in-tree, part of mlx-serve)

An internal plugin is a folder under `src/plugins/<name>/` that ships in every mlx-serve build. It is the same
plugin seam (`abi/plugin.zig` model, `abi/media.zig` media, `abi/quant.zig` quant), but it builds from `src/sdk/`
blocks, is registered by the build's scan, and is held to the repo's full rules. **Tell the user up front: this is a
PR to https://github.com/ddalcu/mlx-serve, it gets rigorous review, and it can be rejected.** If they only want it
for themselves, use Part A.

Read, in order: `docs/recipes/add-a-model.md` (a model), `extend-a-model.md` (one block away from an existing
family), `add-a-media-backend.md`, `add-a-kernel.md`, `add-an-engine.md`, `add-an-api-field.md` as they apply;
`src/plugins/AGENTS.md` (what every plugin owes) and the nearest existing plugin's own `AGENTS.md`.

### B1. Start from mlx-lm's layout

Read mlx-lm's model file (merged or open PR) and the MLX packs already on Hugging Face, and match their weight
names, config keys and quant layout so packs load across engines. Invent only what mlx-lm has no answer for.

### B2. Scaffold

Copy `src/plugins/_template/` (or `_template_media`, `_template_quant`) to `src/plugins/<name>/`. The build
registers it by scanning `src/plugins/*/plugin.zig` (`build/plugins.zig`). Files:

- `config.zig`: claim the `model_type`s, parse with the checked readers (config JSON is untrusted), refuse what you
  do not implement BY NAME. MLX-free: the build refuses a `config.zig` that reaches `mlx/`, `engines/`, `host/`,
  `kernels/`.
- `plugin.zig`: the table. Build the forward from `src/sdk/bases/` and `src/sdk/blocks/`; declare `Caps` honestly
  (batching, MoE, exactness). Import only `sdk/`, `abi/`, `mlx/`, `kernels/`, `env.zig` and your folder
  (`zig build unit-closure` checks it).
- `AGENTS.md` beside them: what it owns, its rules, its packs and oracle; plus a row in the root `AGENTS.md`
  architecture table.
- A second plugin needing the same block means promoting it to `src/sdk/` with both plugins' tests, not copying.

### B3. TDD, proof and kernels

- **Failing test FIRST**, for the right reason, then the minimum code to green. Tests sit at the bottom of the source
  file; a live curl is a sanity check, not a test.
- **Parity**: a fixture config in `src/fixtures/model-configs/`, a parity test against the reference (env-gated dump
  script `tests/dump_<name>_fixtures.py`), a bf16 pack AND a quantized one.
- **Every Metal/CUDA kernel registers with the kernel contract** (`src/sdk/testing/kernel_contract/REGISTERING.md`:
  one `contract.zig`, five gates against a float64 reference, planted controls must fail). Kernels used by two or
  more plugins go in `src/kernels/<area>/` with their sha256 in `kernels/digests.txt`.
- **No new CLI flags or env levers** unless two arms are worth comparing; every env var is registered in `src/env.zig`.

### B4. Build and check

```
zig build check                                  # Sema only, ~35 s: the edit loop
zig build test                                   # sharded; 0 failures
zig build unit-closure                           # the plugin's imports stay in sdk/, mlx/, abi/, kernels/
python3 scripts/lint.py --strict                 # zig build lint, stays 0
zig build -Doptimize=ReleaseFast -Dunits=true    # ONLY to run or measure; never bare `zig build`
```

Then run it against a real pack and benchmark (`/bench`): same-methodology numbers, name the engine.

### B5. Before the PR

- Docs: the plugin's `AGENTS.md`, the root table row, a one-sentence user-facing CHANGELOG line (`/release` rules),
  an app catalog entry only where a surface shows it.
- A class-of-bug finding gets a corpus entry/invariant and a story in `docs/gotchas/` (one short story each).
- Commit messages carry no `Co-Authored-By` or generated-by line. UI + Zig changes for one feature go in one PR.
- Tell the user honestly what review will look at: numerics vs the reference, memory bills, kernel contracts, claim
  narrowness, no source-scan tests, small readable diffs.
