#!/usr/bin/env python3
"""Pure-layer test for the headline chart's title (`python3 tests/plot_title_test.py`).

The title used to be a hardcoded "MLX-serve vs LM Studio · oMLX · MTPLX". The
bars have always dropped engines the CSV has no rows for, so a run on a box
without LM Studio installed rendered a correct chart under a headline claiming a
comparison that never happened — and that chart is a public artifact.

Same rule as "never quote a win without naming the engine it is over": do not
name an engine you did not measure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plot_vs_lmstudio_omlx import FAMILIES, comparison_engine_label  # noqa: E402

V = FAMILIES["all"]["variants"]

CASES = [
    # (engines present in the CSV, expected title fragment)
    ({"mlx-serve", "omlx", "mtplx"}, "oMLX · MTPLX"),
    ({"mlx-serve", "omlx"}, "oMLX"),
    ({"mlx-serve", "mtplx"}, "MTPLX"),
    # Both LM Studio bars collapse to one prose name.
    ({"mlx-serve", "lmstudio-alt", "lmstudio-baseline", "omlx", "mtplx"},
     "LM Studio · oMLX · MTPLX"),
    ({"mlx-serve", "lmstudio-alt"}, "LM Studio"),
    ({"mlx-serve", "lmstudio-baseline"}, "LM Studio"),
    # Ordered hardest-to-beat last, matching the bar ramp — not set order.
    ({"mlx-serve", "mtplx", "omlx"}, "oMLX · MTPLX"),
    # The perf-gate run (step 3 of /release) is mlx-serve only: it must not
    # claim a comparison, and must not render an empty "vs  —".
    ({"mlx-serve"}, "nothing"),
]


def main() -> int:
    failed = 0
    for engines, want in CASES:
        got = comparison_engine_label(V, engines)
        ok = got == want
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {sorted(engines)} -> {got!r}"
              + ("" if ok else f"  (want {want!r})"))

    # Ours is never listed as a competitor, whatever else ran.
    for engines, _ in CASES:
        if "MLX-serve" in comparison_engine_label(V, engines):
            print("  FAIL  mlx-serve listed as its own competitor")
            failed += 1
            break

    # The template must still carry the placeholder, or render() silently goes
    # back to a fixed headline and every case above becomes decorative.
    if "{engines}" not in FAMILIES["all"]["title"]:
        print("  FAIL  family title no longer contains {engines}")
        failed += 1

    print(f"\n{len(CASES) + 2 - failed}/{len(CASES) + 2} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
