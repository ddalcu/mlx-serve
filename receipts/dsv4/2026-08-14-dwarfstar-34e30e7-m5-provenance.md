# DwarfStar `34e30e7` M5 target-only smoke provenance

Date: 2026-08-14

This is one serialized local compatibility smoke. It is not a DSpark result,
a paid or network benchmark, a representative-speed measurement, or an
approval of output quality/parity.

## Exact inputs

- Wrapper: `/Users/pjb/git/mlx-serve-worktrees/dsv4-dwarfstar-merge-ready/zig-out/bin/mlx-serve`
  - SHA-256: `361a21f31e5c0df8559e488ad932f414b992b62e59f593b3371623d48ad48e91`
  - Size: 11,527,672 bytes
  - Version: `mlx-serve 26.8.6`; `mlx 0.32.0`; `mlx-c fba4470b8907`;
    `llama.cpp b10034`; `ds4 34e30e7a6635`; NAX enabled.
- Target: `/Users/pjb/git/ds4/gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf`
  - SHA-256: `659e22fbd01c9e13ea37a57c8d9c41e0a8819dffa3473d3c5286ee44b2d3398f`
  - Size: 97,591,747,456 bytes
- Launch environment (complete):
  `HOME=/Users/pjb`, `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, and
  `DYLD_LIBRARY_PATH=/Users/pjb/git/mlx-serve-worktrees/dsv4-dwarfstar-merge-ready/lib/mlx/lib:/Users/pjb/git/mlx-serve-worktrees/dsv4-dwarfstar-merge-ready/lib/llama/lib`.
- Exact argv (after `env -i` with the environment above):

  ```text
  /Users/pjb/git/mlx-serve-worktrees/dsv4-dwarfstar-merge-ready/zig-out/bin/mlx-serve --model /Users/pjb/git/ds4/gguf/DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf --serve --host 127.0.0.1 --port 18075 --ssd-streaming --no-ds4-mtp --ctx-size 8192 --temp 0 --no-pld
  ```

## Exclusive preflight and outcome

The final preflight found no competing model, GPU backend, build, or test
process, and no listener on port 18075. An unrelated orphaned MTPLX test
fixture remained as a 2.5 MiB shell looping over `sleep 1`; it had no listener
and no model/backend child. System-wide free memory was 93% before launch.

The server admitted the target-only SSD route with `has_mtp=false`, the wrapper
safe-default 8.00 GiB expert-cache budget, and a 9.47 GiB planned DS4 memory
ledger. Full model residency and warmup were explicitly skipped. `/health`
passed, then one local OpenAI-compatible request was made:

```json
{"model":"deepseek-v4-flash-0731","messages":[{"role":"user","content":"Reply with exactly the integer that is one more than 10. No punctuation."}],"max_tokens":1,"temperature":0,"stream":false}
```

It returned exactly `11` with HTTP 200, 20 prompt tokens and one completion
token. Client wall time was 1.421339 seconds; server timing recorded 1,206.352
ms prompt work and 0.000 ms decode for the one-token limit. Those numbers are
not representative performance evidence. At the response observation the
server RSS was 4,949,616 KiB and system-wide free memory was 89%; after clean
shutdown it returned to 93%. The server log records `Shutting down gracefully…`;
the server PID and port 18075 were absent after teardown.

## Raw evidence

- `2026-08-14-dwarfstar-34e30e7-m5-smoke-live.log`
  (`dffa03366db0353b15e358be07ee6c45da85b044874ef13333b8c100a0870e4d`)
- `2026-08-14-dwarfstar-34e30e7-m5-response.json`
  (`4fe95d4ad2faa753a6d32df01befc749f664c800d42ac914c4d668184599e86b`)
- `2026-08-14-dwarfstar-34e30e7-m5-response.headers`
  (`32fba9463f03676b413b472ecfcf329d1e1ea94d95c4e049743109fc482c67e6`)
- `2026-08-14-dwarfstar-34e30e7-m5-client.txt`
  (`bff846892ca51d4ca7c26497c237ca3757c4f347801f598e5ce8e5f35a522946`)

Two distinct earlier launch-lifecycle artifacts are also retained without
replacement: `2026-08-14-dwarfstar-34e30e7-m5-smoke.log` is a zero-byte log
from a session-reaped pre-initialization launch, and
`2026-08-14-dwarfstar-34e30e7-m5-smoke-detached.log` reached server readiness
but was reaped by the execution host before a client could connect. Neither
artifact is a request or performance result.
