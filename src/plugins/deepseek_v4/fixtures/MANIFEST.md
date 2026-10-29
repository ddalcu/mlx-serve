# deepseek_v4 fixtures

Proof tapes (`engines/mlx/plugin/tape.zig`): the legacy forward's values, recorded on `applegpu_g17d`
(the tape names its GPU architecture) before the legacy forward was deleted. Played back bit-equal on the
recording's architecture; elsewhere only dtype, shape and finiteness are checked. The packs are
`../deepseek_v4/fixture.zig`'s tiny random packs (V4 seed 11, V4.1 seed 7; `v4DsparkFiles` / `v41DsparkFiles` add
two DSpark stages and a peaked head).

Regenerate only from a tree that still has the legacy forward (6f2266e7, the commit that recorded them):
`MLX_SERVE_PROOF_RECORD=$PWD zig build test -Dtest-filter=deepseek_v4`.

- `steps.tape`: `proof_test.zig`: the table's prefill / decode / verify steps (verify = the module's all-rows extend).
- `device-decode.tape`: `proof_test.zig`: decode tokens handed as device arrays.
- `served.tape`: `served_test.zig`: the driver's shell, chunked prefill, host and pipelined greedy decodes, a new request.
- `dspark-greedy.tape`: `served_test.zig`: 16 DSpark rounds through the generator, greedy (tokens, accepted, next token, position per round).
- `dspark-sampled.tape`: the same, seeded sampling (temperature 0.8, top_p 0.95).
