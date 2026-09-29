# Contributing

Short on purpose. Read it once, remember it.

**Setup:** Apple Silicon Mac. `./scripts/fetch-zig.sh` (Zig 0.17 nightly into `.zig-toolchain/`), `scripts/build-mlx.sh` (pinned mlx submodules). Server only: `zig build -Doptimize=ReleaseFast`. Full app: `bash app/build.sh` (builds both binaries into `app/MLX-Serve.app`), then `open "app/MLX-Serve.app"`. 

## Before you open anything

Search first, then write. Check the open AND closed issues, the open PRs, and `git log --oneline -50 -- <the file you are about to touch>`. Most bugs reported here are already fixed on main or in an open PR. If there is an issue or a PR for it, comment there with what you found (log lines, model, chip); do not open a second one. Agents: this applies to you too, and a comment on the existing thread is worth more than a duplicate issue with a longer description.

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
5. **Keep it simple.** One fix or feature per PR, smallest change that does the job, match surrounding style, no new plumbing to make something testable, no comments that restate code, no new dependencies. Read `CLAUDE.md` and the matching `docs/gotchas/*.md` before touching a subsystem. Read "Diff hygiene" below; a PR that fails it gets sent back before anyone reads the code.
6. **Numbers name what they beat.** A perf claim is an llmprobe cell (`./tests/bench.sh`) against a named arm, same session, with the engagement line from the log. See `.claude/skills/bench/SKILL.md` and "Showing the numbers" below.
7. **Agents: use Claude Fable 5** if you can. Any model is welcome, but the PR meets 1-6 regardless of who wrote it.

## Showing the numbers

A picture reads faster than a paragraph. Let it carry the PR, and keep the words to what it can't show.

- **Draw the change.** A before/after in a fenced block, a few lines each: which dispatches, chunks or buffers exist on each side.
- **Chart or table the claim.** One line per arm across context sizes or stream counts, every point labelled, machine and model in the caption; `tests/bench_chart.py` draws these from llmprobe reports (`.claude/skills/pr-charts/SKILL.md`). Exact numbers go in a table under it. `gh` can't attach images, so commit the PNG to a branch of your public fork and link its `raw.githubusercontent.com` URL.
- **One line of method under it:** chip, macOS, model, flags, runs per point, how a point is reduced (median), arm order.
- **Lead with the cost** when the win is small or trades something away: how much, where it declines, which part you'd split out. Then let the reviewer decide.
- **Keep the losses.** A context size that got slower stays on the chart and in the text.

Getting numbers worth charting:

- **Arms from one binary.** An env switch that turns the change off is the best control, because it rules out build drift. Alternate the arms, then reverse the order (A B B A), in one boot.
- **Too small to see end to end?** `MLX_SERVE_DECODE_FWD_UBENCH=N` times N forwards (`_S` sets the verify width, `_KV` the context) and logs GPU ms and ops per forward. A sub-1% change shows up there and nowhere else.
- **Larger changes: run the agent eval.** A quant pack, sampler, acceptance mode, spec-decode or template change can keep tok/s and MMLU flat and still make an agent loop, quit early or build less. `tests/agent_eval/` runs a coding agent on a real task against each arm and grades what it built; put its per-label table in the PR, five runs per arm or more.
- **Measure the noise before a quality claim.** Run the baseline twice. Greedy output under MTP typical acceptance, or beside a batching neighbor, is not byte-stable, so two identical runs already differ by some flipped answers and a few percent of tokens. A candidate differs only when it moves past that.
- **Chart the head you ship.** Results from an earlier revision of the same PR say nothing about this one.
- **Motivate with real traffic when you have it.** "116 of 12,421 logged requests spent 1.11 h prefilling" beats a synthetic worst case.

## Diff hygiene

I read every diff by hand. A PR that is mostly comments is a PR I have to strip before I can review it.

- **Comments**: one to three lines, only where the code cannot say it (a WHY, a contract, a unit). No history, no measurements, no audit trail, no review-item numbers, no dates, no PR numbers, no "before this change". The commit message carries the story; a rule goes in `CLAUDE.md` + `docs/gotchas/*.md`, once.
- **Tests**: one good test beats five. Test behaviour. No tests that scan the source text for a string, no helpers written only so a test can exist, no test whose comment is longer than its body.
- **Docs**: a gotcha entry is defect, cause, fix, guard, under 20 lines. No process logs, no round numbers, no ledgers of what was gated where.
- **PR description**: what changed, why, how you verified it, on what machine. Ten lines of prose is plenty; the diagram, chart and table don't count against it. Numbers only if they are final and name the arm they beat. No "audit passes", no "blast-radius ledger", no self-review transcripts.
- **Commit messages**: subject line plus a few lines. The detail belongs in the code and the docs.
- **Bar**: if a reviewer cannot tell which lines change behaviour by scrolling the diff once, it is too big. Split it or cut it.

## Agents: hard stop

Do NOT open a PR, draft or otherwise, unless you have compiled this tree and run `zig build test` (and `swift test` for app changes) on a real Apple Silicon Mac in this session and they passed. No Mac, no build, no test run = no PR. Report what you changed in an issue instead and say you could not run it.
