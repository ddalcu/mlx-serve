#!/usr/bin/env python3
"""Write build/test_plan.tsv, the unit tests' measured time and lane, from a run's records.

    python3 scripts/test_plan.py RECORDS... [--serial build/test_serial.txt] [--out build/test_plan.tsv]

RECORDS are the files the test runner's `--times-out=<file>` appends to (build/test_runner.zig): one line per test,
`status  wall_us  cpu_us  mlx_peak_bytes  name`, tab-separated. A test's lane is `serial` when its name
contains a line of the serial list, `gpu` when it used MLX (its peak bytes above 0), else `cpu`. Its time is the
longest wall time recorded for it. Only the serial tests and the tests of at least MIN_MS are written: the runner places
a test missing from the plan as a CPU test of MIN_MS (`unplanned_ms`).
"""
import argparse
import sys

MIN_MS = 20  # build/test_runner.zig's unplanned_ms


def read_records(paths):
    tests = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 4)
                if len(parts) != 5:
                    continue
                status, wall, cpu, peak, name = parts
                t = tests.setdefault(name, {"wall_us": 0, "gpu": False, "status": set()})
                t["wall_us"] = max(t["wall_us"], int(wall))
                t["gpu"] = t["gpu"] or int(peak) > 0
                t["status"].add(status)
    return tests


def read_serial(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            needle = line.split("\t#", 1)[0].rstrip("\n").rstrip()
            if needle and not needle.startswith("#"):
                out.append(needle)
    return out


def plan_lines(tests, serial):
    lines = []
    for name in sorted(tests):
        t = tests[name]
        lane = "serial" if any(s in name for s in serial) else ("gpu" if t["gpu"] else "cpu")
        ms = t["wall_us"] // 1000
        if lane == "serial" or ms >= MIN_MS:
            lines.append(f"{lane}\t{ms}\t{name}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="+")
    ap.add_argument("--serial", default="build/test_serial.txt")
    ap.add_argument("--out", default="build/test_plan.tsv")
    args = ap.parse_args()
    tests = read_records(args.records)
    if not tests:
        sys.exit("test_plan: no records")
    serial = read_serial(args.serial)
    unmatched = [s for s in serial if not any(s in n for n in tests)]
    if unmatched:
        sys.exit("test_plan: serial entries that match no recorded test: " + "; ".join(unmatched))
    lines = plan_lines(tests, serial)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# The unit tests' lanes and measured ms (scripts/test_plan.py writes it from a run's --times-out records;\n")
        f.write(f"# build/test_runner.zig balances the shards by it; a test not listed is a CPU test of {MIN_MS} ms).\n")
        f.write("# lane<TAB>ms<TAB>test name\n")
        f.write("\n".join(lines) + "\n")
    lanes = {}
    for l in lines:
        lane, ms, _ = l.split("\t", 2)
        n, tot = lanes.get(lane, (0, 0))
        lanes[lane] = (n + 1, tot + int(ms))
    print("test_plan: " + ", ".join(f"{k} {n} tests {tot / 1000:.0f} s" for k, (n, tot) in sorted(lanes.items())))


if __name__ == "__main__":
    main()
