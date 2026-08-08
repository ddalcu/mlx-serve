#!/bin/bash
# test_website_pages.sh — static checks for the website/ marketing site (GitHub Pages).
#
# Verifies, without a browser:
#   1. every feature deep-dive page exists at website/<slug>/index.html
#   2. each page carries the SEO contract: <title>, meta description, canonical
#      URL matching its slug, Open Graph tags, JSON-LD, and the shared stylesheet
#   3. every relative href/src across all site pages resolves to a real file
#      (screenshots/ refs are reported as PENDING, not failures — they are the
#      screenshot shopping list)
#   4. sitemap.xml lists the homepage + every deep-dive URL; robots.txt points at it
#   5. index.html links to every deep-dive page (internal-link SEO)
#
# Usage: ./tests/test_website_pages.sh

set -u
cd "$(dirname "$0")/.." || exit 1
DOCS="website"
# custom domain (repo Settings → Pages) — the old ddalcu.github.io/mlx-serve
# URLs 301 here, but nothing in the site may reference them anymore
BASE_URL="https://mlxserve.com"

SLUGS=(
  claude-code-local
  lm-studio-alternative
  ollama-alternative
  image-generation
  video-generation
  voice-cloning
  tool-calling
  agent-sandbox
  speculative-decoding
  local-ai-assistant
  llm-tier-list
)

PASS=0; FAIL=0; PEND=0
pass() { PASS=$((PASS+1)); }
fail() { FAIL=$((FAIL+1)); echo "FAIL: $1"; }
pend() { PEND=$((PEND+1)); echo "PENDING: $1"; }

check() { # check <file> <grep-pattern> <label>
  if grep -q -- "$2" "$1"; then pass; else fail "$3 (pattern not found: $2)"; fi
}

# ── 1+2: per-page existence + SEO contract ─────────────────────────────────
for slug in "${SLUGS[@]}"; do
  f="$DOCS/$slug/index.html"
  if [ ! -f "$f" ]; then
    fail "$slug: page missing ($f)"
    continue
  fi
  pass
  check "$f" "<title>"                                        "$slug: <title>"
  check "$f" 'name="description"'                             "$slug: meta description"
  check "$f" "rel=\"canonical\" href=\"$BASE_URL/$slug/\""    "$slug: canonical URL"
  check "$f" 'property="og:title"'                            "$slug: og:title"
  check "$f" 'property="og:image"'                            "$slug: og:image"
  check "$f" 'application/ld+json'                            "$slug: JSON-LD"
  check "$f" 'assets/feature.css'                             "$slug: shared stylesheet"
  check "$f" 'assets/feature.js'                              "$slug: shared script"
done

# ── shared assets ───────────────────────────────────────────────────────────
for a in "$DOCS/assets/feature.css" "$DOCS/assets/feature.js"; do
  if [ -f "$a" ]; then pass; else fail "shared asset missing: $a"; fi
done

# ── 3: relative link/image resolution across all docs html ─────────────────
html_files=("$DOCS/index.html")
for slug in "${SLUGS[@]}"; do
  [ -f "$DOCS/$slug/index.html" ] && html_files+=("$DOCS/$slug/index.html")
done

for f in "${html_files[@]}"; do
  dir="$(dirname "$f")"
  refs=$(grep -oE '(href|src)="[^"]+"' "$f" | sed -E 's/^(href|src)="//; s/"$//' | sort -u)
  while IFS= read -r ref; do
    [ -z "$ref" ] && continue
    case "$ref" in
      http://*|https://*|mailto:*|data:*|\#*|javascript:*) continue ;;
    esac
    target="${ref%%#*}"          # strip fragment
    target="${target%%\?*}"     # strip query
    [ -z "$target" ] && continue
    resolved="$dir/$target"
    # a trailing slash means directory-style URL -> index.html
    case "$resolved" in */) resolved="${resolved}index.html" ;; esac
    if [ -e "$resolved" ]; then
      pass
    elif [ -d "${dir}/${target}" ] && [ -f "${dir}/${target}/index.html" ]; then
      pass
    else
      case "$target" in
        *screenshots/*) pend "screenshot not yet added: $target (referenced by $f)" ;;
        *) fail "broken relative ref in $f: $ref" ;;
      esac
    fi
  done <<< "$refs"
done

# ── 4: sitemap + robots ─────────────────────────────────────────────────────
if [ -f "$DOCS/sitemap.xml" ]; then
  pass
  check "$DOCS/sitemap.xml" "$BASE_URL/</loc>" "sitemap: homepage entry"
  for slug in "${SLUGS[@]}"; do
    check "$DOCS/sitemap.xml" "$BASE_URL/$slug/</loc>" "sitemap: $slug entry"
  done
else
  fail "$DOCS/sitemap.xml missing"
fi

if [ -f "$DOCS/robots.txt" ]; then
  pass
  check "$DOCS/robots.txt" "Sitemap: $BASE_URL/sitemap.xml" "robots.txt: sitemap reference"
else
  fail "$DOCS/robots.txt missing"
fi

# ── 5: index.html links to every deep-dive page ─────────────────────────────
for slug in "${SLUGS[@]}"; do
  check "$DOCS/index.html" "href=\"$slug/\"" "index.html links to $slug/"
done

# ── 6: download CTAs point at the direct latest-DMG URL (version-free) ──────
# GitHub redirects releases/latest/download/<asset> to the newest release's
# asset, so this never needs a version bump. The asset name MLXCore.dmg is
# stable across releases (produced by app/build.sh).
DMG_URL="https://github.com/ddalcu/mlx-serve/releases/latest/download/MLXCore.dmg"
for f in "${html_files[@]}"; do
  check "$f" "$DMG_URL" "$f: direct DMG download CTA"
done

# ── 7: llm-tier-list interactive contract ──────────────────────────────────
# The tier-list page is the one interactive page: an S–D tier board, a
# unified-memory hardware filter, and Google-account voting via Firebase
# (Auth + Firestore). Votes degrade to localStorage until the Firebase web
# config is pasted in, so the page must always ship both paths.
TIER="$DOCS/llm-tier-list/index.html"
if [ -f "$TIER" ]; then
  pass
  check "$TIER" 'id="tier-board"'          "tier-list: tier board container"
  for t in S A B C D; do
    check "$TIER" "data-tier=\"$t\""       "tier-list: tier row $t"
  done
  check "$TIER" 'id="unranked-table"'      "tier-list: unranked models table"
  check "$TIER" 'id="unranked-search"'     "tier-list: unranked text filter"
  check "$TIER" 'promote-pop'              "tier-list: tier-promotion effect"
  if grep -q 'class="hero-f"' "$TIER"; then
    fail "tier-list: hero section must stay removed (board above the fold)"
  else
    pass
  fi
  check "$TIER" 'data-ram='                "tier-list: hardware RAM filter buttons"
  check "$TIER" 'const SEED_MODELS'        "tier-list: pinned seed/fallback model list"
  check "$TIER" 'huggingface.co/api/models' "tier-list: live HF popularity query"
  check "$TIER" 'base_model:quantized'     "tier-list: quant roll-up lineage rule"
  check "$TIER" 'const firebaseConfig'     "tier-list: firebase web config block"
  check "$TIER" 'gstatic.com/firebasejs'   "tier-list: firebase SDK import"
  check "$TIER" 'GoogleAuthProvider'       "tier-list: google sign-in"
  check "$TIER" 'wilsonLower'              "tier-list: wilson-score tier ranking"
  check "$TIER" 'localStorage'             "tier-list: unconfigured localStorage fallback"
  # unit-test the page's embedded logic (tiering, vote sanitization, model
  # data integrity) — evals the page's own script, so no drift possible
  if command -v node >/dev/null 2>&1; then
    if node tests/website_tier_list_logic.mjs; then pass; else fail "tier-list: logic assertions (tests/website_tier_list_logic.mjs)"; fi
  else
    pend "tier-list logic assertions skipped (node not installed)"
  fi
else
  fail "llm-tier-list: interactive page missing ($TIER)"
fi

# ── 8: ONE shared header on every page ──────────────────────────────────────
# Every page carries the same nav: an EMPTY version pill, a Tier list link, and
# the Download CTA.
#
# The pill is empty in the markup on purpose. It used to be hardcoded and
# checked against the CHANGELOG's newest entry here, which meant every release
# had to rewrite 18 files and the guard sat red in between (it did, through
# 26.8.1/2/3). assets/version.js now fills it from the latest GitHub release —
# the same source the version-free Download CTA points at — so drift is
# impossible by construction and the only thing left to guard is that nobody
# pastes a literal version back in.
for f in "$DOCS"/index.html "$DOCS"/*/index.html; do
  page=$(basename "$(dirname "$f")")
  check "$f" 'class="nav-ver"></span>' "$page: nav version pill (empty, filled at runtime)"
  check "$f" 'assets/version.js'       "$page: nav version resolver script"
  check "$f" 'class="nav-dl '          "$page: nav Download CTA"
  check "$f" 'nav-dl-desktop"'         "$page: nav Download CTA (desktop variant)"
  check "$f" 'class="nav-appstore"'    "$page: nav App Store CTA (mobile variant)"
  check "$f" 'appstore.svg'            "$page: nav App Store badge asset ref"
  check "$f" '>Tier list<'             "$page: nav Tier list link"
  if grep -qE 'nav-ver">v?[0-9]' "$f"; then
    fail "$page: version hardcoded in the nav pill — assets/version.js resolves it"
  else
    pass
  fi
done

# The resolver itself: reads the repo's latest release, hides on failure rather
# than showing a stale or wrong version, and caches so a visit doesn't burn
# GitHub's 60/hour unauthenticated budget on every page view.
VJS="$DOCS/assets/version.js"
if [ -f "$VJS" ]; then
  pass
  check "$VJS" 'releases/latest'   "version.js: reads the latest release"
  check "$VJS" 'tag_name'          "version.js: takes the release tag"
  check "$VJS" 'localStorage'      "version.js: caches the lookup"
  check "$DOCS/assets/feature.css" '.nav-ver:empty' "feature.css: unresolved pill renders as nothing"
  check "$DOCS/index.html"         '.nav-ver:empty' "index: unresolved pill renders as nothing"
else
  fail "version resolver missing ($VJS)"
fi

# ── 9: anchor links must land clear of the fixed 52px nav ───────────────────
check "$DOCS/assets/feature.css" 'scroll-padding-top' "feature.css: scroll offset for fixed nav"
check "$DOCS/index.html"         'scroll-padding-top' "index: scroll offset for fixed nav"

# ── 10: tier page has NO local FAQ — its nav points at the homepage FAQ ─────
if grep -q 'class="faq-wrap"' "$TIER"; then
  fail "tier-list: local FAQ must stay removed (FAQ lives on the homepage)"
else
  pass
fi
check "$TIER" 'href="../#faq"' "tier-list: FAQ nav link points at homepage FAQ"

echo ""
echo "website pages: $PASS passed, $FAIL failed, $PEND pending screenshots"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
