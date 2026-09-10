#!/usr/bin/env bash
# Thin wrapper around `python -m spoken_syntax_probe.finetune`.
#
# NOTE: the module reads ./vocab.json and calls load_metric("wer") at import
# time. The wrapper changes to the repository root first, so ./vocab.json
# resolves exactly as it did when the module lived at the root.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python -m spoken_syntax_probe.finetune "$@"
