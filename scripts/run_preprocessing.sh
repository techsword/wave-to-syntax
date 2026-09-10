#!/usr/bin/env bash
# Thin wrapper around `python -m spoken_syntax_probe.preprocessing`.
#
# Run it from anywhere: the wrapper changes to the repository root first, so
# the cwd-relative output files (dataset csvs, stanza tree dumps, bow model)
# land in exactly the same place as the original `python preprocessing.py`.
# The corpus roots/splits are passed as options:
#   --spokencoco_root, --spokencoco_split, --librispeech_root, --libri_split.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Use the project virtualenv when present, otherwise the active interpreter.
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python -m spoken_syntax_probe.preprocessing "$@"
