#!/bin/bash
# Static guard for .github/workflows/release.yml event gating.
#
# The release workflow triples as (1) the tag/dispatch RELEASE pipeline,
# (2) the dry-run packaging check, and (3) the PR packaging build that
# signs + notarizes a DMG artifact WITHOUT releasing. The class of bug this
# pins: someone edits a step's `if:` and a PR suddenly creates a tag, a
# GitHub release, or a Homebrew formula push — or the opposite, PR builds
# silently stop notarizing and the artifact regresses to unsigned.
#
# Hermetic — parses the YAML, no network, no runners.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'EOF'
import re, sys, yaml

FAIL = 0
def check(cond, msg):
    global FAIL
    if cond:
        print(f"PASS {msg}")
    else:
        print(f"FAIL {msg}")
        FAIL = 1

wf = yaml.safe_load(open(".github/workflows/release.yml"))

# YAML 1.1 parses the bare key `on` as boolean True.
triggers = wf.get("on", wf.get(True, {}))
check("pull_request" in triggers, "pull_request trigger present")
check("push" in triggers and "workflow_dispatch" in triggers,
      "tag-push + workflow_dispatch triggers still present")

job = wf["jobs"]["build"]

# Fork PRs have no secrets — the job must skip itself, not fail at cert import.
job_if = str(job.get("if", ""))
check("github.event.pull_request.head.repo.full_name == github.repository" in job_if,
      "job-level fork-PR guard present")

steps = {s.get("name", ""): s for s in job["steps"]}

def step_if(name):
    check(name in steps, f"step exists: {name}")
    return str(steps.get(name, {}).get("if", ""))

# Release-only steps must be OFF for PRs.
rel_if = step_if("Create Release")
check("pull_request" in rel_if and "!=" in rel_if,
      "Create Release gated off for pull_request")
# The formula push lives in homebrew.yml, triggered by the draft -> published
# transition — pushing it from the build job advertised a version whose assets
# were still locked behind the unpublished draft (checked below).
check(not any("Update Homebrew formulas" in n for n in steps),
      "release.yml no longer pushes Homebrew formulas")
check("Formula/mlx-serve.rb" not in open(".github/workflows/release.yml").read(),
      "release.yml never touches the formula files")
check("workflow_dispatch" in step_if("Create tag (manual dispatch)"),
      "tag creation restricted to workflow_dispatch")

# Notarization must RUN on PRs — its gate may exclude dry_run but never PRs.
for n in ("Notarize CLI", "Notarize app bundle"):
    check("pull_request" not in step_if(n),
          f"{n} not excluded on pull_request")

# The NAX static guard must run in the RELEASE pipeline itself — ci.yml
# checking the same cache key doesn't cover a cache-miss rebuild on the
# release runner, and that stage is what actually ships in the DMG.
nax_steps = [s for s in job["steps"]
             if "test_mlx_staged_nax.sh" in str(s.get("run", ""))]
check(len(nax_steps) == 1, "NAX metallib static guard step present")
check(nax_steps and "if" not in nax_steps[0],
      "NAX guard unconditional (runs on every event incl. PRs)")

# The PR build's output must be uploaded as an artifact.
upload = [s for s in job["steps"]
          if s.get("uses", "").startswith("actions/upload-artifact")]
check(any("pull_request" in str(s.get("if", "")) for s in upload),
      "artifact upload covers pull_request")

# ── Version consistency: the app bundle can never advertise a version other
# than the release it came from. ───────────────────────────────────────────
# The .app's version came from the COMMITTED app/Info.plist while the tag and
# the CLI binary came from the CI-computed CalVer, and nothing kept the two
# equal. v26.8.1 shipped a bundle reporting 26.7.12, so UpdateChecker saw a
# permanently "newer" release and installing it never helped — the new DMG
# reported 26.7.12 too. Self-perpetuating, and invisible to every other guard.
pkg_run = str(steps.get("Package app bundle", {}).get("run", ""))
check("cp app/Info.plist" in pkg_run, "app bundle ships app/Info.plist")

def logical_lines(script):
    """Shell lines with backslash-continuations folded — a guard must pin what
    a command DOES, not how it happens to be wrapped."""
    out, buf = [], ""
    for raw in script.splitlines():
        buf += raw[:-1] + " " if raw.rstrip().endswith("\\") else raw
        if not raw.rstrip().endswith("\\"):
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out

lines = logical_lines(pkg_run)
stamp = [l for l in lines if "PlistBuddy" in l]
check(any("CFBundleShortVersionString" in l for l in stamp)
      and any("CFBundleVersion" in l for l in stamp),
      "app bundle Info.plist is version-stamped")
check(stamp and all("steps.version.outputs.version" in l for l in stamp),
      "Info.plist stamp uses the CI-computed version")

# Stamp the bundle COPY, never the repo file — the committed plist stays clean.
check(stamp and all('"$CONTENTS/Info.plist"' in l for l in stamp),
      "stamp targets the bundle copy, not the repo's app/Info.plist")

# ...and it has to land before the bundle is sealed, or the signature breaks.
sign_at = next((i for i, l in enumerate(lines)
                if "codesign" in l and '"$APP"' in l), None)
check(sign_at is not None, "app bundle is codesigned")
check(stamp and sign_at is not None
      and max(i for i, l in enumerate(lines) if "PlistBuddy" in l) < sign_at,
      "Info.plist stamped before the app bundle is signed")

# ── CalVer timezone: the release ran at 01:26 UTC on Aug 1 while it was still
# Jul 31 locally, so CI minted 26.8.1 against a CHANGELOG, Info.plist and perf
# artifacts that all said 26.7.12. CI and app/build.sh must resolve YY.M from
# the SAME clock or they disagree for a few hours around every month boundary.
wf_text = open(".github/workflows/release.yml").read()
build_sh = open("app/build.sh").read()

def tz_pin(text):
    m = re.search(r"TZ[=:]\s*[\"']?([A-Za-z_]+/[A-Za-z_]+)", text)
    return m.group(1) if m else None

check("date -u +%y" not in wf_text, "CalVer month is not computed in UTC")
check(tz_pin(wf_text) is not None, "release.yml pins the CalVer timezone")
check(tz_pin(wf_text) == tz_pin(build_sh),
      "release.yml and app/build.sh compute CalVer in the SAME timezone")

# ── Third-party attribution must travel WITH the binaries. The shipped binary
# links Apache-2.0 code (MTPLX/dflash/oMLX Metal kernels, jinja.cpp), and
# section 4 conditions redistribution on the recipient getting the license text
# and the NOTICE attributions. Nothing pinned that those files leave the repo,
# and for months they did not: every packaging path shipped the binary alone.
for name, text in (("release.yml CLI tarball + app bundle", wf_text),
                   ("app/build.sh", build_sh)):
    for f in ("LICENSE-APACHE-2.0", "NOTICE"):
        check(f in text, f"{name} packages {f}")
# Two copies in release.yml: one per artifact (tarball and .app).
check(wf_text.count("LICENSE-APACHE-2.0") >= 2,
      "release.yml packages the licenses into BOTH the tarball and the .app")

# ── app/build.sh FAST_DEV: an iteration lever must not become a shipping mode.
# Every fast-dev shortcut (incremental Swift, in-place bundle, reused
# frameworks, no notarization) is gated on ONE variable, and an UNSET variable
# has to be the release path — the version of this that got written first
# reindented the whole script and left `# ... (rest of the block is identical)`
# where the notarization and the App Store .pkg used to be, which is a release
# script that silently ships an unnotarized app.
b_lines = build_sh.splitlines()
# Comments blanked, line numbers kept: these checks are about what the script
# DOES, and every rule below is also explained in a comment right beside it.
b_code = ["" if l.lstrip().startswith("#") else l for l in b_lines]

def b_first(pat):
    return next((i for i, l in enumerate(b_code) if re.search(pat, l)), None)

def b_count(text):
    return sum(l.count(text) for l in b_code)

def in_fast_dev_else(idx):
    """True when line `idx` sits in the ELSE arm of the nearest enclosing FAST_DEV if."""
    els = max((i for i in range(idx) if b_lines[i].strip() == "else"), default=None)
    if els is None:
        return False
    ifs = max((i for i in range(els) if b_lines[i].lstrip().startswith("if ")), default=None)
    return ifs is not None and "$FAST_DEV" in b_lines[ifs]

check("FAST_DEV:-0" in build_sh,
      "FAST_DEV defaults to 0 — an unset env is the release path")

# The engine defaults to ReleaseFast in BOTH modes: a Debug mlx-serve decodes
# 2-4x slower, so an app built that way lies about every latency. ZIG_DEBUG is
# the deliberate opt-in, and it is FAST_DEV-only — FAST_DEV is what stops before
# notarization and the DMG, so that gate is the whole reason a Debug engine
# cannot reach a shipping artifact.
check("ZIG_DEBUG:-0" in build_sh,
      "ZIG_DEBUG defaults to 0 — the engine is ReleaseFast unless asked otherwise")
check(b_count("-Doptimize=ReleaseFast") == 1
      and re.search(r"^ZIG_OPT=\(-Doptimize=ReleaseFast\)", build_sh, re.M),
      "the zig optimize mode defaults to ReleaseFast, in exactly one place")
dbg_opt = b_first(r"-Doptimize=Debug")
check(b_count("-Doptimize=Debug") == 1 and dbg_opt is not None
      and 'if [ "$ZIG_DEBUG" = "1" ]; then' in b_lines[dbg_opt - 1],
      "the Debug optimize mode sits directly under the ZIG_DEBUG gate")
check(re.search(r'\[ "\$ZIG_DEBUG" = "1" \] && \[ "\$FAST_DEV" != "1" \]', build_sh)
      and re.search(r'ZIG_DEBUG=1 is a FAST_DEV-only lever', build_sh),
      "ZIG_DEBUG without FAST_DEV is refused BY NAME, never quietly ignored")
refuse = b_first(r"ZIG_DEBUG=1 is a FAST_DEV-only lever")
check(refuse is not None
      and any(l.strip() == "exit 1" for l in b_lines[refuse:refuse + 4]),
      "the ZIG_DEBUG refusal actually exits")
check(refuse is not None and b_first(r'"\$ZIG" build') is not None
      and refuse < b_first(r'"\$ZIG" build'),
      "the refusal fires before any build work is done")

# The debug Swift configuration is reachable only from the FAST_DEV gate.
dbg = b_first(r"^\s*SWIFT_CONFIG=debug")
rel = b_first(r"^\s*SWIFT_CONFIG=release")
check(dbg is not None and 'if [ "$FAST_DEV" = "1" ]; then' in b_lines[dbg - 1],
      "the debug swift configuration sits directly under the FAST_DEV gate")
check(rel is not None and in_fast_dev_else(rel),
      "the release path still compiles -c release")

# The clean-bundle wipe belongs to the release path — only `rm -rf` guarantees
# a retired resource stops shipping.
rmapp = b_first(r'^\s*rm -rf "\$APP"$')
check(rmapp is not None and in_fast_dev_else(rmapp),
      "the release build still starts from an EMPTY bundle")

# FAST_DEV stops at a signed .app: everything past that point produces
# something to hand someone.
fast_exit = next((i for i, l in enumerate(b_lines)
                  if l.strip() == "exit 0"
                  and any('"$FAST_DEV" = "1"' in x for x in b_lines[max(0, i - 8):i])), None)
sign_app = b_first(r'codesign "\$\{APP_SIGN_OPTS\[@\]\}" "\$APP"')
check(fast_exit is not None, "FAST_DEV exits before anything shippable is produced")
check(sign_app is not None and fast_exit is not None and sign_app < fast_exit,
      "a fast build is still a SIGNED bundle")
for shipping in (r"notarytool submit", r"productbuild", r"create-dmg\.sh"):
    at = b_first(shipping)
    check(at is not None and fast_exit is not None and fast_exit < at,
          f"the FAST_DEV exit precedes {shipping.replace(chr(92), '')}")

# Reusing the staged frameworks means skipping their re-sign too: the fixups
# that would follow rewrite the files and invalidate the signatures.
skip_stage = b_first(r"^\s*STAGE_FRAMEWORKS=0")
check(skip_stage is not None
      and any('"$FAST_DEV" = "1"' in l for l in b_lines[max(0, skip_stage - 8):skip_stage]),
      "frameworks are only reused under FAST_DEV")
for pat, what in ((r'install_name_tool -add_rpath @loader_path "\$CONTENTS/Frameworks/libmlx\.dylib"',
                   "libmlx rpath fixup"),
                  (r'codesign "\$\{SIGN_OPTS\[@\]\}" "\$fw"', "framework signing")):
    for i, l in enumerate(b_lines):
        if re.search(pat, l):
            guarded = any('STAGE_FRAMEWORKS' in x for x in b_lines[max(0, i - 12):i])
            check(guarded, f"{what} runs only when the frameworks were restaged")

# ── Homebrew push timing: the release is created as a DRAFT, so its assets
# are not downloadable until it is published. The formula push must fire on
# the publish transition ONLY — any other trigger reintroduces the window
# where `brew upgrade` advertises a version nobody can download.
bwf = yaml.safe_load(open(".github/workflows/homebrew.yml"))
btriggers = bwf.get("on", bwf.get(True, {}))
check(list(btriggers) == ["release"],
      "homebrew.yml triggers on the release event only")
check(btriggers.get("release", {}).get("types") == ["published"],
      "homebrew.yml fires only when a release is PUBLISHED")

bjob = bwf["jobs"]["update-formulas"]
check("prerelease" in str(bjob.get("if", "")),
      "pre-releases never reach brew")

bsteps = bjob["steps"]
checkout = next((s for s in bsteps
                 if s.get("uses", "").startswith("actions/checkout")), {})
check(checkout.get("with", {}).get("ref") == "main",
      "formula commit lands on main, not the release tag checkout")

bruns = " ".join(str(s.get("run", "")) for s in bsteps)
for f in ("Formula/mlx-serve.rb", "Casks/mlx-core.rb"):
    check(f in bruns, f"homebrew.yml updates {f}")

sys.exit(FAIL)
EOF
