# Guest engines: folding community forks back in without merging their code

Status: PROPOSAL (for discussion). Nothing here is implemented yet.

## The problem

Forks like [sushi](https://github.com/beamivalice/sushi) are specialists. They take our engine, cut it down to a
few model types and go deep on them: EXL3 expert quants, SSD expert streaming of a 335 GB bf16 checkpoint, MiMo-V2.
Their `transformer.zig` is 70k lines and has drifted from ours (63k). Merging that back line by line would give us two
diverging copies of the engine core and would put their pins (mlx, kernels, pack formats) in charge of our build.

We want the user to get sushi's models inside mlx-serve (one port, one `/v1/models`, the app, `launch`, LAN, keys)
while the fork keeps its own code, release cadence and MLX pin.

## What we already have

| Seam | Where | Why it isn't the answer alone |
|---|---|---|
| Embedded engines | `src/arch/{ds4,llama}.zig` + `lib/*`, linked in | Every engine grows our binary and build; one MLX/Metal process means one pin and one crash domain |
| Upstream providers | `src/providers.zig`: `<id>@<name>` → curl proxy | Static, user-run servers; no lifecycle, no memory billing, chat/completions only |
| LAN mirroring | `src/lan.zig`: streaming proxy, `<id>@<peer>` | Transport only, remote machine |
| Media modality slots | `gen.zig` unions on `LoadedModel` | In-process; good for in-tree backends, not for forks |

A **guest engine** is the missing fourth seam: a provider that we START, SUPERVISE and BILL, bound to loopback,
routed by `model_type` like a native model.

Sushi already ships the guest half:

- `sushi --guest-manifest` prints `guest.json`: `version`, `commit`, `guest_api` (1), `mlx`/`mlx_c` pins, `min_macos`,
  `model_types` (`["qwen4_exp"]`), and the pack contract it accepts (`expert_quant: {format: exl3, codebooks, windows}`).
- `--parent-pid <pid>`: the guest SIGTERMs itself when the host dies, so a crashed host never strands GPU memory.

## Contract (guest_api 1)

Guest side — a binary that:

1. `--guest-manifest` → JSON as above, plus two fields we should ask for:
   - `routes`: the surfaces it serves (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/completions`, ...).
     A route not listed is a NAMED 400 on our side, same as providers today.
   - `claims`: how to recognise a pack it wants (e.g. `model_type` + `quantization_config.format == "exl3"`), so a
     plain MLX `qwen4_exp` pack stays native and an EXL3 one goes to the guest.
2. Serves OpenAI-compatible HTTP on `--host 127.0.0.1 --port <N>` with `GET /health` (200 when ready) and `/props`
   (at least `memory.active_bytes`).
3. Honours `--parent-pid`, `--model`, `--ctx-size`, and a memory ceiling flag we pass (`--gpu-budget-gib`).
4. Exits cleanly on SIGTERM.

Host side (us):

- **Install**: `~/.mlx-serve/engines/<name>/{bin, guest.json, bin.sha256}`. `mlx-serve engine add <release-url>`
  downloads, checks the sha, runs `--guest-manifest`, refuses an unknown `guest_api`, a `min_macos` above the OS or
  a manifest that disagrees with the shipped `guest.json`. Nothing runs that the user did not install by name.
- **Routing**: `model_discovery` already peeks `config.json`. A pack a guest `claims` gets `engine = <name>`;
  `model-settings.json` `engine` overrides either way. `/v1/models` rows carry an `engine` badge.
- **Lifecycle**: a new registry kind next to `ds4_engine`/`llama_engine`: `guest: ?*GuestProcess`. Load = spawn with
  an ephemeral loopback port + `--parent-pid <ours>`, wait on `/health`; unload = SIGTERM, then SIGKILL after a grace
  period. Load errors surface by name (`GuestExited`, `GuestManifestMismatch`, `GuestNotReady`).
- **Traffic**: the handler proxies the request body as is, reusing the provider/LAN streaming proxy. The guest owns
  templates, tool parsing, sampling, caches. We are TRANSPORT, like LAN: we do not re-parse its output.
- **Memory**: the guest is billed like any resident model. Admission reads its `/props` `active_bytes` into the
  `PromiseLedger`, and we hand it `--gpu-budget-gib` = what the plan leaves it. Two MLX processes share one wired
  limit, so this is the part that must be right before anything ships.
- **Observability**: guest stdout/stderr → `~/.mlx-serve/logs/<engine>-<port>.log`; `/metrics` scrapes are merged with
  an `engine` label.

## What stays native vs guest

- Native: architectures that many users run and that fit our kernels and pins.
- Guest: a fork's specialty (new pack format, SSD expert streaming, a model too big for the common Mac), or anything
  on a different MLX pin.
- A guest is also an INCUBATOR: when a guest feature proves general (e.g. expert streaming for every MoE), it comes
  upstream as an ordinary PR against our tree, and the guest drops it.
- Open question: whether ds4/llama.cpp should later move out as guests too, to shrink our build.

## What we give up

A guest model does not share our prefix cache, batching, spec-decode or tool-call repairs. Its quality is the guest's.
Every guest response is one extra loopback hop (negligible next to decode).

## Phases

1. `src/guest.zig` pure: manifest parse, `guest_api`/`min_macos` gate, `claims` match. Hermetic tests.
2. `GuestProcess` supervisor + registry kind + proxy routing; `tests/test_guest_engine.sh` against a stub guest
   (a tiny script that answers `--guest-manifest`, `/health`, and one chat completion).
3. Memory billing + `--gpu-budget-gib` + `model-settings.json` `engine`.
4. `mlx-serve engine add|list|remove`, app UI (engine badge, install from a release URL).
5. First real guest: sushi, with its authors, pinned by `guest.json`.
