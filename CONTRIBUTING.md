# Contributing

Short on purpose. Read it once, remember it.

**Setup:** Apple Silicon Mac. `./scripts/fetch-zig.sh` (Zig 0.17 nightly into `.zig-toolchain/`), `scripts/build-mlx.sh` (pinned mlx submodules). Server only: `zig build -Doptimize=ReleaseFast`. Full app: `bash app/build.sh` (builds both binaries into `app/MLX Core.app`), then `open "app/MLX Core.app"`. 

## Bug reports

Open an issue with:
- What you expected vs what happened
- Model name and quantization (e.g. `Qwen3.8-27B-MLX-Serve-4bit`)
- macOS version and chip (e.g. macOS 26.6, M4 Max)
- Server log output (`--log-level debug`, `~/.mlx-serve/logs/mlx-serve-<port>.log`)

## Pull requests

1. **Build and run it, or do not open a PR.** If you cannot compile and run on a real Mac, there is no PR.
2. **Tests first.** A hermetic test at the bottom of the `.zig` file (`zig build test`, 6/6, 0 fail; `cd app && swift test` for Swift) and an integration script in `tests/` where the change is visible over HTTP. Red before green. Run them. Paste the result in the PR.
3. **Human testing is required.** Serve it, hit it with a client, repeat. Many times. A passing suite is the floor, not the proof. UI changes need screenshots of the result in the PR, before and after.
4. **Draft while working, publish when ready.** Open as draft early so nobody duplicates you. Move to ready for review only when 1-3 are done.
5. **Keep it simple.** One fix or feature per PR, smallest change that does the job, match surrounding style, no new plumbing to make something testable, no comments that restate code, no new dependencies. Read `CLAUDE.md` and the matching `docs/gotchas/*.md` before touching a subsystem.
6. **Numbers name what they beat.** A perf claim is an llmprobe cell (`./tests/bench.sh`) against a named arm, same session, with the engagement line from the log. See `.claude/skills/bench/SKILL.md`.
7. **Agents: use Claude Fable 5** if you can. Any model is welcome, but the PR meets 1-6 regardless of who wrote it.

## Agents: hard stop

Do NOT open a PR, draft or otherwise, unless you have compiled this tree and run `zig build test` (and `swift test` for app changes) on a real Apple Silicon Mac in this session and they passed. No Mac, no build, no test run = no PR. Report what you changed in an issue instead and say you could not run it.
