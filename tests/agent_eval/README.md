# Agent eval: does a change still build good things?

Runs a coding agent against mlx-serve on a real task, then grades what it built with a vision judge
that answers checkable yes/no claims taken from the task prompt. It catches what token-level benches
can't: a pack, sampler or spec-decode change that keeps tok/s and MMLU flat but makes the agent loop,
quit early, or ship a worse app.

```
mlx-serve (your arm) ─▶ run.sh: pi, headless, compacting between turns ─▶ out/<task>/<label>-<n>/project
                    ─▶ capture.sh: npm run build, serve dist/, headless Chrome, frames over time
                    ─▶ judge.py: claude -p reads the frames, answers each rubric claim pass/fail
                    ─▶ per-label table: score ± 95% CI, per-claim pass rate, build, turns, lines
```

The method follows Anthropic's
[Automating eval design and hillclimbing](https://claude.dev/blog/automating-eval-design-and-hillclimbing/):
claims you can check instead of a 1-to-5 scale, the cheapest grader that works (build status, turns
and line counts are counted, not judged), and the eval's own noise measured before a result is read.

## Run it

Needs `pi`, `node`, `agent-browser` (`npm i -g agent-browser`) and the `claude` CLI. `run.sh`
writes pi's config per run with `mlx-serve launch pi` (set `MLX_SERVE` to use a build that isn't on
`PATH`), so pi's context window and compaction match what the server advertises.

```sh
mlx-serve --serve --port 8080 --model <arm A> <flags>        # start the first arm
tests/agent_eval/run.sh voxel-pagoda armA http://127.0.0.1:8080 5
mlx-serve --serve --port 8080 --model <arm B> <flags>        # then the second
tests/agent_eval/run.sh voxel-pagoda armB http://127.0.0.1:8080 5
tests/agent_eval/capture.sh voxel-pagoda
tests/agent_eval/judge.py voxel-pagoda                       # 3 judge reps per run
```

A voxel-pagoda run takes 20 to 90 minutes on an M5 Ultra (`TIMEOUT_MIN` caps it); judging costs
about $0.10 per run-rep on Sonnet 5.5. Everything lands in `tests/agent_eval/out/` (ignored).
Rerunning `capture.sh` or `judge.py` skips what is already done.

## Reading results

The first use, comparing two Flash-Next packs, is written up in #628. Two lessons from it:

- **Runs from a buggy build are not signal.** Most of the first scenes came from builds where prompt
  lookup drafts ran under typical acceptance (fixed in #614). Those runs loop-stopped far more often,
  and including them more than doubled the apparent quality gap between the arms.
- **Read the claims, not only the score.** The first rubric scored 1.00 on most working scenes, and a
  later one passed signs whose text hung off the board. Both were caught by checking the judge's
  per-claim reasons against the frames; a pilot on a handful of runs before the full set is cheap.

Scene score alone can hide a difference: the counted columns (turns, source lines, how runs ended)
separated those packs when the score didn't.

## Use it again

Run it on larger changes that can move what the model writes over hundreds of turns: a quant pack, a
sampler or acceptance mode, speculative decoding, the chat template, KV quantization. Five runs per
arm is the floor; a clean A/B needs both arms on the same build and flags except for the change.

A task is two files in `tasks/<name>/`:

- `prompt.txt`: what the agent is asked to build. It must produce a web app with `npm run build`
  writing `dist/`.
- `rubric.json`: `capture` (how many frames, how far apart) and `claims`. Each claim is one checkable
  sentence tied to something the prompt asks for, with a `weight`. Leave out anything a counter can
  measure, and anything the frames can't show.

Pilot a new rubric on 3 to 5 runs and read every reason before the full set. If most working runs
score 1.00, the claims are too easy to separate anything.
