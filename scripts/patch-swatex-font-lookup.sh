#!/usr/bin/env bash
# Point SwaTexRender's KaTeX font lookup at a location a signed .app can hold.
#
# SwiftPM generates `Bundle.module` as
#   Bundle(path: Bundle.main.bundleURL + "SwaTex_SwaTexRender.bundle")
# with the CI build directory as the only fallback, and `fatalError`s when
# neither exists. For an app bundle `bundleURL` is the .app itself, and
# codesign refuses to seal anything in a bundle's root ("unsealed contents
# present in the bundle root"), so the fonts can only ship in
# Contents/Resources — where that accessor never looks. Result: every
# hand-assembled build (DMG and MAS alike, both `swift build`) traps on the
# first equation (issue #233). Xcode's own accessor searches Contents/Resources
# and is why an Xcode-built app is fine.
#
# The one call site is replaced with a candidate search. Idempotent; run after
# `swift package resolve` and before `swift build`. Drop this once upstream
# takes the equivalent fix.
set -euo pipefail

CHECKOUT="${1:?usage: patch-swatex-font-lookup.sh <SwaTex checkout dir>}"
FILE="$CHECKOUT/Sources/SwaTexRender/KaTeXFontProvider.swift"
MARKER="mlxServeKaTeXFontURL"

[ -f "$FILE" ] || { echo "ERROR: $FILE not found" >&2; exit 1; }

if grep -q "$MARKER" "$FILE"; then
    echo "  SwaTex font lookup already patched"
    exit 0
fi

# SwiftPM leaves checkouts read-only.
chmod u+w "$FILE"

python3 - "$FILE" <<'PY'
import sys

path = sys.argv[1]
source = open(path).read()

original = """            let url = Bundle.module.url(
                forResource: name, withExtension: "ttf", subdirectory: "Fonts"),"""
if original not in source:
    sys.exit(
        "ERROR: SwaTexRender's font lookup does not look the way this patch expects.\n"
        "       Re-check KaTeXFontProvider.swift against scripts/patch-swatex-font-lookup.sh."
    )

source = source.replace(original, "            let url = mlxServeKaTeXFontURL(name),", 1)
source += '''
// Added by mlx-serve: scripts/patch-swatex-font-lookup.sh (issue #233).
// `Bundle.module` resolves to a path no signed .app can contain, and traps
// when it misses, so the bundle is searched for where a bundle can actually
// ship it. Nothing found returns nil, which `makeUnitFont` already answers
// with a system font.
private final class MlxServeKaTeXBundleFinder {}

func mlxServeKaTeXFontURL(_ name: String) -> URL? {
    let bundleName = "SwaTex_SwaTexRender.bundle"
    let own = Bundle(for: MlxServeKaTeXBundleFinder.self)
    let bases = [
        Bundle.main.resourceURL, Bundle.main.bundleURL,
        own.resourceURL, own.bundleURL,
        // `swift test`: the reading bundle is the .xctest, the resource
        // bundle is its sibling in the build directory.
        own.bundleURL.deletingLastPathComponent(),
    ]
    for base in bases.compactMap({ $0 }) {
        if let bundle = Bundle(url: base.appendingPathComponent(bundleName)),
            let url = bundle.url(forResource: name, withExtension: "ttf", subdirectory: "Fonts")
        {
            return url
        }
    }
    return nil
}
'''
open(path, "w").write(source)
PY

grep -q "$MARKER" "$FILE" || { echo "ERROR: patch reported success but did not apply" >&2; exit 1; }
echo "  Patched SwaTex font lookup for bundled apps"
