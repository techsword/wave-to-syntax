#!/usr/bin/env bash
# Thin wrapper around `python -m spoken_syntax_probe.forced_alignment`.
#
# Machine-local corpus roots are passed as options (see README "Datasets"):
#   scripts/run_forced_alignment.sh \
#       --libri-root /path/to/librispeech/train-clean-100 \
#       --scc-root   /path/to/SpokenCOCO \
#       --tg-root    /path/to/aligned_val
# The wrapper changes to the repository root first so cwd-relative outputs
# (spokencoco_val.csv, split transcription files) match the original behaviour.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python -m spoken_syntax_probe.forced_alignment "$@"
