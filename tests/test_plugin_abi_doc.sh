#!/bin/bash
# docs/plugin-abi.md is what scripts/plugin_abi_doc.py makes from src/abi/*.zig, and names every public ABI type.
set -u
cd "$(dirname "$0")/.." || exit 1
fail=0
python3 scripts/plugin_abi_doc.py --check || fail=1
for t in $(grep -h -o -E '^pub const [A-Za-z0-9_]+ = (extern struct|extern union|packed struct|enum)' src/abi/{plugin,media,quant,host,engine,common,dylib}.zig | awk '{print $3}'); do
    grep -q "^### \`$t\` (" docs/plugin-abi.md 2>/dev/null || { echo "FAIL: docs/plugin-abi.md has no section for $t"; fail=1; }
done
[ $fail = 0 ] && echo "PASS: plugin ABI reference is current and complete"
exit $fail
