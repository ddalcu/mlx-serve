#!/bin/bash
# test_release_workflow_gates.sh — static guard for release packaging. release.yml event gating: the PR trigger and
# fork-PR guard, Create Release + Homebrew push OFF for pull_request, tag creation dispatch-only, notarize steps never
# excluded on PRs, artifact upload covers PRs (the class "a PR build releases" / "PR DMGs stop being notarized"), and
# the shipped mlx-serve signed with the entitlement that lets it load a plugin compiled on the user's Mac.
# app/build.sh: the CalVer timezone, license packaging, every Contents/Resources entry staged outside the App
# Store-only block is staged by release.yml too, and FAST_DEV (default 0, Debug Swift only under its gate, one
# ReleaseFast default, `-Doptimize=Debug` only under the ZIG_DEBUG gate, which is FAST_DEV-only and refused by name,
# signing skipped only with the staging that invalidates it, the FAST_DEV exit after signing and before notarizing).
# ci.yml, linux.yml, windows.yml and the two one-shot build scripts: only a pull request builds below ReleaseFast or
# strips the test binary, and only main saves Actions caches (the MLX and ds4 stages aside). linux.yml's libjinja
# cache key hashes the inputs of build-linux.sh's libjinja stamp.
#
#   ./tests/test_release_workflow_gates.sh   # hermetic: PyYAML parse, no runners (ci.yml's `gates` job runs it)

set -euo pipefail
cd "$(dirname "$0")/.."
# PyYAML: the repo venv has it, a stock macOS python3 does not.
PY=python3
[ -x venv/bin/python3 ] && PY=venv/bin/python3

"$PY" - <<'EOF'
import glob, re, sys, yaml

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

# ── The pre_release checkbox exists so a build can be cut WITHOUT consuming the
# version number: it tags v<YY.M.N>-pre-release.<n>, which the `^vYY.M.[0-9]+$`
# match that picks N never sees, so the plain vYY.M.N is still there to cut
# later. Two halves, both load-bearing:
#   - the suffix must be applied where the tag is MINTED (the version step), not
#     at the release step, or the tag consumes the number anyway;
#   - the release must still be created as a DRAFT and must NOT set `prerelease`
#     itself — flipping that flag is a deliberate manual step, and setting it
#     here would also be what keeps the formula push away (homebrew.yml skips
#     pre-releases), turning a checkbox into a silent brew decision.
inputs = triggers.get("workflow_dispatch", {}).get("inputs", {})
check("pre_release" in inputs, "workflow_dispatch offers a pre_release checkbox")
check(inputs.get("pre_release", {}).get("default") in (False, "false"),
      "pre_release defaults to OFF")
version_step = next((s for s in job["steps"] if s.get("id") == "version"), {})
check("inputs.pre_release" in str(version_step.get("run", "")),
      "the pre-release suffix is applied where the tag is minted")
check("-pre-release" in str(version_step.get("run", "")),
      "the pre-release tag suffix is spelled in the version step")
rel_with = steps.get("Create Release", {}).get("with", {})
check(rel_with.get("draft") in (True, "true"),
      "the release is still created as a draft")
check("prerelease" not in rel_with,
      "the workflow never sets the prerelease flag itself (manual, by design)")

# The Linux tarballs are linux.yml's build of the same commit, attached to the
# draft: never rebuilt here, never on a PR or a dry run.
linux = wf["jobs"].get("linux-assets", {})
linux_text = str(linux)
check(linux.get("needs") == "build", "linux-assets runs after the draft exists")
check("pull_request" in str(linux.get("if", "")) and "dry_run" in str(linux.get("if", "")),
      "linux-assets gated off for pull_request and dry_run")
check("gh run download" in linux_text and "gh release upload" in linux_text,
      "linux-assets downloads linux.yml's artifacts and uploads them")
check("build-linux.sh" not in linux_text, "linux-assets never rebuilds")

# MLX JIT-compiles its CUDA kernels against the headers beside libmlx (lib/mlx/include,
# CCCL included): a tarball without them fails every custom kernel on a user's machine.
lwf = yaml.safe_load(open(".github/workflows/linux.yml"))
pkg = [st.get("run", "") for j in lwf["jobs"].values() for st in j.get("steps", []) if st.get("name") == "Package"]
check(len(pkg) == 1 and "lib/mlx/include" in pkg[0], "the Linux tarball carries lib/mlx/include")

# The Windows zip is windows.yml's build of the same commit, attached the same way.
win = wf["jobs"].get("windows-assets", {})
win_text = str(win)
check(win.get("needs") == "build", "windows-assets runs after the draft exists")
check("pull_request" in str(win.get("if", "")) and "dry_run" in str(win.get("if", "")),
      "windows-assets gated off for pull_request and dry_run")
check("gh run download" in win_text and "gh release upload" in win_text,
      "windows-assets downloads windows.yml's artifact and uploads it")
check("windows.yml" in win_text and "build-windows" not in win_text,
      "windows-assets reads windows.yml's run and never rebuilds")
check("steps.version.outputs.tag" in str(job.get("outputs", {}).get("tag", "")),
      "build job exports the release tag")

# Notarization must RUN on PRs — its gate may exclude dry_run but never PRs.
for n in ("Notarize CLI", "Notarize app bundle"):
    check("pull_request" not in step_if(n),
          f"{n} not excluded on pull_request")

mlx_serve_builds = [s for s in job["steps"]
                    if s.get("name") == "Build mlx-serve (Zig)"]
check(len(mlx_serve_builds) == 1, "exactly one mlx-serve release-artifact build")
check(any("-Dgit-sha=${{ github.sha }}" in str(s.get("run", ""))
          for s in mlx_serve_builds),
      "release mlx-serve build passes -Dgit-sha=${{ github.sha }}")

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

# ── The version is stamped into the BUNDLE, never into the tracked plist.
# Two failures meet here, one on each side of the same line. Stamping the repo
# file dirties the working tree — release.yml's Homebrew step cannot `git
# rebase` on a dirty tree, and a contributor who merely BUILDS ends up with a
# modified app/Info.plist they never touched. Not stamping at all is worse and
# has already shipped: v26.8.1's DMG contained an app reporting 26.7.12, so
# UpdateChecker offered an update that installing could never satisfy, forever.
# The one answer that is neither: stamp $CONTENTS/Info.plist after the cp that
# creates it and before the app is signed, which is what release.yml does.
# The number being stamped has to be RIGHT, and `gh release list` is the only
# thing that knows it. Swallowing its failure into a default (`|| echo "0"`)
# yields v<YY.M>.1 — behind every release already published that month — and
# that is the same UpdateChecker dead end as an unstamped bundle, reached
# through a different door. Observed live: a working copy carrying 26.8.1 with
# 26.8.9 committed and v26.8.8 published, written by a build whose `gh` was not
# authed. A legitimately empty month must still yield 0, so the jq keeps
# `// 0`; what must not be defaulted is the COMMAND failing.
gh_line = b_first(r"gh release list")
check(gh_line is not None, "app/build.sh computes CalVer from the published releases")
if gh_line is not None:
    check("|| echo" not in b_code[gh_line],
          "a `gh` failure is not swallowed into a default CalVer number")
    check(any("exit 1" in l for l in b_code[gh_line:gh_line + 12]),
          "a `gh` failure stops the build instead of stamping a version behind the published ones")

    # A pre-release tag (v26.9.5-pre-release.1) is not a release number: the
    # filter must skip it like release.yml does, not choke on it and stop the build.
    m = re.search(r'--jq "(.*?)" 2>', b_code[gh_line])
    check(m is not None, "the CalVer jq filter is readable")
    if m:
        import json, subprocess
        # Undo double-quote escaping the way the shell does, then fill in YM.
        prog = re.sub(r'\\([\\"$`])', r'\1', m.group(1)).replace("${YM}", "26.9")
        tags = ["v26.9.5-pre-release.1", "v26.9.4", "v26.9.3", "v26.8.11"]
        out = subprocess.run(["jq", prog], input=json.dumps([{"tagName": t} for t in tags]),
                             capture_output=True, text=True)
        check(out.returncode == 0 and out.stdout.strip() == "4",
              f"the CalVer filter skips pre-release tags (got {out.stdout.strip() or out.stderr.strip()})")

    # ...but only a build that could BECOME a release may require `gh` at all.
    # The GitHub CLI is release tooling, not a build dependency: a contributor
    # who clones and runs app/build.sh signs ad-hoc, cuts nothing, and must not
    # be stopped by a missing login. Nothing in CI reaches this code (release.yml
    # mirrors the script and runs its own `gh release list`), so the identity is
    # the whole discriminator — and it is already the flag deciding hardened
    # runtime, notarization and productbuild further down.
    guard = "\n".join(b_code[max(0, gh_line - 14):gh_line])
    check('"$IDENTITY" = "-"' in guard,
          "the `gh` round-trip is skipped for an ad-hoc build — no signing identity, no release")
    check("$FAST_DEV" in guard,
          "the `gh` round-trip is skipped under FAST_DEV too")
    # The skip has to LAND somewhere: an ad-hoc build still needs a CALVER.
    adhoc_ver = b_first(r'CALVER="\$\(bash "\$PROJECT_ROOT/scripts/dev-version.sh"')
    check(adhoc_ver is not None and adhoc_ver < gh_line,
          "the no-`gh` arm takes the source's version from scripts/dev-version.sh (git tags, then the plist)")

stamp_targets = [l for l in b_code if "PlistBuddy" in l and "Set :CFBundle" in l]
check(stamp_targets and all("$CONTENTS/Info.plist" in l for l in stamp_targets),
      "app/build.sh stamps the BUNDLE's Info.plist, never the tracked source")
check(not any("$INFO_PLIST_SRC" in l for l in stamp_targets),
      "no build mode writes the tracked app/Info.plist — a build leaves the tree clean")
for key in ("CFBundleVersion", "CFBundleShortVersionString"):
    check(any(f"Set :{key} " in l for l in stamp_targets),
          f"the bundle's {key} is stamped from the version being built")
first_stamp = b_first(r'PlistBuddy -c "Set :CFBundle')
cp_plist = b_first(r'^\s*cp "\$INFO_PLIST_SRC" "\$CONTENTS/Info\.plist"')
check(cp_plist is not None and first_stamp is not None and cp_plist < first_stamp,
      "the stamp runs AFTER the plist is copied into the bundle (or the cp overwrites it)")
check(first_stamp is not None and sign_app is not None and first_stamp < sign_app,
      "the stamp runs BEFORE the bundle is sealed (a later write breaks the signature)")

# ── Outside plugins are compiled on the user's Mac, so they are ad-hoc signed: a hardened-runtime server without
# disable-library-validation refuses to load them. Every codesign of the shipped mlx-serve carries the entitlement.
signs = re.findall(r'codesign [^\n]*--sign [^\n]*"\$(?:STAGING|CONTENTS/MacOS)/mlx-serve"', wf_text)
check(len(signs) == 2 and all("--entitlements app/mlx-serve.entitlements" in c for c in signs),
      "both shipped mlx-serve binaries are signed with app/mlx-serve.entitlements")
check("com.apple.security.cs.disable-library-validation" in open("app/mlx-serve.entitlements").read(),
      "app/mlx-serve.entitlements lets the server load a locally built plugin")

# ── release.yml assembles its own .app: every resource app/build.sh stages
# (its App Store-only block aside) must be staged there too, or the DMG ships
# without it and the app's reader silently falls back to nothing.
mas_only = set()
for start in (i for i, l in enumerate(b_code) if re.match(r'if \[ "\$MAS" = "1" \]', l)):
    end = next(k for k in range(start, len(b_lines)) if b_lines[k] == "fi")
    mas_only.update(range(start, end + 1))
res_name = r'\$CONTENTS/Resources/([\w.-]+)'
built_res = {n for i, l in enumerate(b_code) if i not in mas_only for n in re.findall(res_name, l)}
release_res = {n for l in lines if not l.lstrip().startswith("#") for n in re.findall(res_name, l)}
check("agent-skills" in built_res, "app/build.sh's staged resources are readable")
for n in sorted(built_res):
    check(n in release_res, f"the release bundle stages Resources/{n} like app/build.sh")

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
for f in ("Formula/mlx-serve.rb", "Casks/mlx-serve.rb"):
    check(f in bruns, f"homebrew.yml updates {f}")

# ── The formula moves lib/* into libexec/, so every library the binary links
# through its rpaths must be repointed there; the server links libmlx directly.
formula = open("Formula/mlx-serve.rb").read()
check(re.search(r'"@rpath/libmlx\.dylib",\s*"#\{libexec\}/libmlx\.dylib",\s*"#\{bin\}/mlx-serve"', formula),
      "formula repoints the binary's @rpath/libmlx.dylib into libexec")

# ── Only a pull request may build below ReleaseFast: linux-assets and windows-assets attach what the push and
# dispatch runs built.
DEBUG_ON_PR = "${{ github.event_name == 'pull_request' && 'Debug' || 'ReleaseFast' }}"
STRIP_ON_PR = "${{ github.event_name == 'pull_request' && '-Dstrip' || '' }}"

def code_of(path):
    return "\n".join(l for l in open(path).read().splitlines() if not l.lstrip().startswith("#"))

workflows = {n: yaml.safe_load(open(f".github/workflows/{n}")) for n in ("ci.yml", "linux.yml", "windows.yml")}

release_modes = re.findall(r"-Doptimize=(\w+)", code_of(".github/workflows/release.yml"))
check(release_modes and set(release_modes) == {"ReleaseFast"}, "every release.yml build is ReleaseFast")

linux = workflows["linux.yml"]
# `-Dunits` and `-Dstrip` too: push and dispatch build the shipped binary as one unit, with debug info.
ARGS_ON_PR = r"\$\{\{ github\.event_name == 'pull_request' && '[^']*' \|\| '' \}\}"
for name in ("linux.yml", "windows.yml"):
    cwf, cuda = workflows[name], workflows[name]["jobs"]["cuda"]
    check(cuda["env"].get("ZIG_OPTIMIZE") == DEBUG_ON_PR,
          f"{name} builds Debug on pull requests and ReleaseFast otherwise")
    args = str(cuda["env"].get("ZIG_BUILD_ARGS", ""))
    check(args == "" or re.fullmatch(ARGS_ON_PR, args),
          f"{name} passes extra zig build arguments on pull requests only")
    elsewhere = str(cwf.get("env", {})) + str(cuda["steps"])
    for var in ("ZIG_OPTIMIZE", "ZIG_BUILD_ARGS"):
        check(var not in elsewhere, f"no other {name} scope or step overrides {var}")
    check("-Doptimize" not in args, f"{name} sets no optimize mode through ZIG_BUILD_ARGS")

check(re.findall(r'-Doptimize=("[^"]*"|\S+)', code_of("scripts/build-linux.sh"))
      == ['"${ZIG_OPTIMIZE:-ReleaseFast}"'],
      "build-linux.sh defaults to ReleaseFast, in exactly one place")
ps1 = code_of("scripts/build-windows.ps1")
check(re.findall(r"^\$optimize = .*$", ps1, re.M)
      == ['$optimize = if ($env:ZIG_OPTIMIZE) { $env:ZIG_OPTIMIZE } else { "ReleaseFast" }']
      and re.findall(r"-Doptimize=[^\s\"]+", ps1) == ["-Doptimize=$optimize"],
      "build-windows.ps1 defaults to ReleaseFast, in exactly one place")

ci_jobs = workflows["ci.yml"]["jobs"]
server = [s for s in ci_jobs["zig"]["steps"]
          if re.search(r"^\s*zig build(\s+-|\s*$)", str(s.get("run", "")), re.M)]

def builds_release_off_pr(step):
    if "-Doptimize=ReleaseFast" in step["run"]:
        return True
    return '-Doptimize="$OPTIMIZE"' in step["run"] and step.get("env", {}).get("OPTIMIZE") == DEBUG_ON_PR

check(len(server) == 1 and builds_release_off_pr(server[0]),
      "ci.yml's zig job builds ReleaseFast outside pull requests")
check(len(server) == 1 and "-Dstrip" in server[0]["run"].split(),
      "ci.yml's zig job build stays stripped")
test_runs = [str(s.get("run", "")) for s in ci_jobs["zig-test"]["steps"]
             if "zig build test" in str(s.get("run", ""))]
check(len(test_runs) == 1 and STRIP_ON_PR in test_runs[0]
      and "-Dstrip" not in test_runs[0].replace(STRIP_ON_PR, ""),
      "ci.yml strips the test binary on pull requests only")

# ── libjinja cache: build-linux.sh rebuilds on a stamp mismatch, and an exact key hit is never saved again, so a
# file the stamp hashes and the key does not would rebuild on every run.
JINJA_FILES = {"lib/jinja_cpp/libjinja-linux.a", "lib/jinja_cpp/.linux-version"}
linux_steps = linux["jobs"]["cuda"]["steps"]

def step_at(uses, name):
    return next((i for i, s in enumerate(linux_steps)
                 if s.get("name") == name and str(s.get("uses", "")).startswith(uses)), None)

restore_at = step_at("actions/cache/restore@", "Restore libjinja")
save_at = step_at("actions/cache/save@", "Save libjinja")
build_at = next((i for i, s in enumerate(linux_steps) if s.get("run") == "./scripts/build-linux.sh"), None)
check(None not in (restore_at, build_at, save_at) and restore_at < build_at < save_at,
      "linux.yml restores libjinja before the build and saves it after")
restore = linux_steps[restore_at] if restore_at is not None else {}
save = linux_steps[save_at] if save_at is not None else {}
for what, s in (("restore", restore), ("save", save)):
    check(set(str(s.get("with", {}).get("path", "")).split()) == JINJA_FILES,
          f"the libjinja {what} carries the archive and its stamp")
key = str(restore.get("with", {}).get("key", ""))
check(restore.get("id") == "jinja"
      and save.get("with", {}).get("key") == "${{ steps.jinja.outputs.cache-primary-key }}",
      "the libjinja save writes the key the restore missed")
check("steps.jinja.outputs.cache-hit != 'true'" in str(save.get("if", ""))
      and "hashFiles('lib/jinja_cpp/.linux-version') != ''" in str(save.get("if", "")),
      "libjinja is saved only after a miss that left a stamp")
for part in ("matrix.name", "env.ZIG_CPU"):
    check(part in key, f"the libjinja cache key names {part}")
hashed = set(re.findall(r"'([^']+)'", " ".join(re.findall(r"hashFiles\(([^)]*)\)", key))))
stamp_inputs = re.search(r"^JINJA_INPUTS=\((.*)\)$", open("scripts/build-linux.sh").read(), re.M)
stamp_inputs = set(stamp_inputs.group(1).split()) if stamp_inputs else set()
drift = sorted(hashed ^ (stamp_inputs | {"scripts/fetch-zig.sh"}))
check(stamp_inputs and not drift,
      f"the libjinja cache key hashes the stamp's inputs and the pinned Zig, no more (differ: {drift})")
stamped = {p for pattern in stamp_inputs for p in glob.glob(pattern)}
sources = {p for ext in ("cpp", "h", "hpp") for p in glob.glob(f"lib/jinja_cpp/**/*.{ext}", recursive=True)}
check(sources and sources <= stamped, f"the libjinja stamp covers every source and header (missing: {sorted(sources - stamped)})")

# ── The 10 GB Actions cache cap: only main saves. The MLX and ds4 stages take an hour to rebuild, so a pull request
# that changes their inputs keeps its own copy.
PR_MAY_SAVE = {"Cache MLX stage", "Cache ds4 stage", "Save MLX stage"}
MAIN_ONLY = "github.ref == 'refs/heads/main'"
for name, cwf in workflows.items():
    cache_steps = [s for j in cwf["jobs"].values() for s in j["steps"]
                   if str(s.get("uses", "")).startswith(("actions/cache@", "actions/cache/save@"))
                   and s.get("name") not in PR_MAY_SAVE]
    check(all(s["uses"].startswith("actions/cache/save@") for s in cache_steps),
          f"{name} saves caches only through actions/cache/save steps, MLX and ds4 stages aside")
    for s in cache_steps:
        cond = str(s.get("if", ""))
        check(MAIN_ONLY in cond and "||" not in cond, f"{name}: '{s.get('name')}' saves from main only")

sys.exit(FAIL)
EOF
