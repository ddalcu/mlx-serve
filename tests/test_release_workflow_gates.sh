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
