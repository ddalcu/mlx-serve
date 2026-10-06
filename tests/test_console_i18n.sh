#!/bin/bash
# Hermetic completeness, resolver, placeholders and pre-paint contract guard.
# No model, server binary, npm package or network required.
set -euo pipefail
cd "$(dirname "$0")/.."
exec node --test tests/console_i18n_test.mjs
