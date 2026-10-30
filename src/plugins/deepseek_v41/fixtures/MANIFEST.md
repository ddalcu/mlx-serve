# deepseek_v41 fixtures

Proof tapes (`engines/mlx/plugin/tape.zig`): the legacy forward's values, recorded on `applegpu_g17d`
(the tape names its GPU architecture) before the legacy forward was deleted. Played back bit-equal on the
recording's architecture; elsewhere only dtype, shape and finiteness are checked. The packs are
`../deepseek_v4/fixture.zig`'s tiny random packs (V4 seed 11, V4.1 seed 7; `v4DsparkFiles` / `v41DsparkFiles` add
two DSpark stages and a peaked head).

Regenerate only from a tree that still has the legacy forward (6f2266e7, the commit that recorded them):
`MLX_SERVE_PROOF_RECORD=$PWD zig build test -Dtest-filter=deepseek_v4`.

- `steps.tape`: `proof_test.zig`: the table's steps (verify = `extendAllResumable`).
- `proof-resume.tape`: `proof_test.zig`: `resumePrompt` lengths and the resumed tail's logits.
- `device-decode.tape`: `proof_test.zig`: decode tokens handed as device arrays.
- `served.tape`: `../deepseek_v4/served_test.zig`: the driver's shell, as V4's.
- `resume.tape`: `../deepseek_v4/served_test.zig`: `driver.resumePrompt` lengths and the tails' logits.
- `image.tape`: `../deepseek_v4/served_test.zig`: five image rows spliced across a chunk boundary.
- `dspark-greedy.tape`: `../deepseek_v4/served_test.zig`: 16 DSpark rounds, greedy.
- `dspark-sampled.tape`: the same, seeded sampling.
