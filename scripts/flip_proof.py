#!/usr/bin/env python3
"""The flip proof: which tests depend on a lever's default without naming the arm they need.

  python3 scripts/flip_proof.py run LEVER [zig build test args...]   the tests that fail only with LEVER flipped
  python3 scripts/flip_proof.py changed BASE                        the flags whose registered default differs from BASE
  python3 scripts/flip_proof.py ci BASE [zig build test args...]    `run` for each of those (a PR that flips a default)

`run` runs `zig build test` twice, as given and with `-Dtest-flip=LEVER` (every test starts with LEVER at its other
arm, src/sdk/testing/world.zig), and prints the tests that fail only flipped. A test that pins the lever, or reads its
default through `env.override(.X, null)`, does not change; one that leans on the default does. Exit 1 when any does.
After a default flip, `ci` runs the tests at the old arm, so it lists the tests that hold only at the new default.
"""
import re, subprocess, sys

FAILED = re.compile(r"^error: '(.+?)' failed", re.M)
FLAG = re.compile(r'pub const (MLX_SERVE_\w+): Var = \.\{ \.kind = \.kill, \.type = \.flag, \.default = "(on|off)')


def zig_test(args, flip=None):
    cmd = ["zig", "build", "test", "--summary", "all", *args] + ([f"-Dtest-flip={flip}"] if flip else [])
    print("$ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = p.stdout + p.stderr
    for l in out.splitlines():
        if "tests passed" in l or "Build Summary" in l:
            print("  " + l.strip())
    return p.returncode, set(FAILED.findall(out))


def run(lever, args):
    rc_a, fail_a = zig_test(args)
    rc_b, fail_b = zig_test(args, lever)
    dep = sorted(fail_b - fail_a)
    print(f"flip {lever}: {len(dep)} test(s) fail only flipped; {len(fail_a)} fail either way or unflipped")
    for t in dep:
        print(f"  depends on the {lever} default: {t}")
    for t in sorted(fail_a - fail_b):
        print(f"  fails only unflipped: {t}")
    if rc_b != 0 and rc_a == 0 and not dep:
        print("  the flipped run failed without a test failure line: read its log")
        return 1
    return 1 if dep else 0


def defaults(text):
    return {m.group(1): m.group(2) for m in FLAG.finditer(text)}


def changed(base):
    old = defaults(subprocess.run(["git", "show", f"{base}:src/env.zig"], capture_output=True, text=True, check=True).stdout)
    new = defaults(open("src/env.zig").read())
    return sorted(n for n in new if n in old and old[n] != new[n])


def main(argv):
    if len(argv) < 3 or argv[1] not in ("run", "changed", "ci"):
        print(__doc__)
        return 2
    if argv[1] == "run":
        return run(argv[2], argv[3:])
    levers = changed(argv[2])
    if argv[1] == "changed":
        print("\n".join(levers))
        return 0
    if not levers:
        print(f"flip proof: no registered default differs from {argv[2]}")
        return 0
    return max(run(l, argv[3:]) for l in levers)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
