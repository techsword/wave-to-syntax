#!/usr/bin/env bash
# Thin wrapper around `python -m spoken_syntax_probe.extract_segmented_embeddings`.
#
# Machine-local paths are passed as options (see README "Datasets"):
#   scripts/run_extract_segmented.sh \
#       --model_path   facebook/wav2vec2-base \
#       --model_tag    wav2vec_small \
#       --dataset      scc \
#       --root         /path/to/SpokenCOCO \
#       --aligned_path /path/to/SpokenCOCO/aligned_val \
#       --save_dir     segmented_embeddings
# --model_path is a Hugging Face model id or a local HF checkpoint directory;
# --model_tag reproduces the published output-file tag (defaults to the basename
# of --model_path with a trailing .pt stripped).
# The wrapper changes to the repository root first so cwd-relative outputs
# match the original behaviour.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python -m spoken_syntax_probe.extract_segmented_embeddings "$@"
