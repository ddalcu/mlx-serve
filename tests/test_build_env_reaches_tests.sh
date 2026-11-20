#!/bin/bash
# A shell variable set for `zig build test` reaches the tests on every run, not only on the run that configured the
# build (Zig caches the configuration, and a Run step's env map with it). CPU only: the acestep tokenization oracle
# skips without its env and fails on a missing fixture path.
set -u
cd "$(dirname "$0")/.."
Z=./.zig-toolchain/zig
F="acestep oracle: prompt tokenization matches reference ids"
"$Z" build test -Dtest-filter="$F" >/dev/null 2>&1 || { echo "FAIL: the skipped run did not pass"; exit 1; }
if ACESTEP_TEST_MODEL=/nonexistent ACESTEP_TEXT_IDS=/nonexistent "$Z" build test -Dtest-filter="$F" >/dev/null 2>&1; then
    echo "FAIL: the env set on the second run did not reach the test (it skipped)"; exit 1
fi
echo "PASS: env reaches tests on a cached configuration"
