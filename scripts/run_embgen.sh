#!/usr/bin/env bash
# Thin wrapper around `python -m spoken_syntax_probe.embedding_generation`.
#
# The models/datasets lists live in the module argparse defaults. The wrapper
# changes to the repository root first so extracted embeddings land under the
# cwd-relative `embeddings/` directory, exactly as before.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python -m spoken_syntax_probe.embedding_generation "$@"
