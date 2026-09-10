#!/usr/bin/env bash
# Thin wrapper around `python -m spoken_syntax_probe.treekernelprobe`.
#
# The wrapper changes to the repository root first so the cwd-relative
# `regress-data/` and `embeddings/` paths match the original behaviour.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python -m spoken_syntax_probe.treekernelprobe "$@"
