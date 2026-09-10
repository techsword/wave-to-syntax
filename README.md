# Wave to Syntax: Probing spoken language models for syntax

This repo hosts the code for 

```
@misc{shen2023wave,
      title={Wave to Syntax: Probing spoken language models for syntax}, 
      author={Gaofei Shen and Afra Alishahi and Arianna Bisazza and Grzegorz Chrupała},
      year={2023},
      eprint={2305.18957},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Installation

Python 3.10, 3.11 or 3.12.

Dependencies are declared in `pyproject.toml`; `requirements.txt` is only a
pointer to it. Install with [uv](https://docs.astral.sh/uv/), selecting one
accelerator extra:

```
uv sync --python 3.10 --extra cpu
# or, for a CUDA 12.1 build:
uv sync --python 3.10 --extra cu121
```

This creates `.venv`, installs the dependencies, and installs the
`spoken_syntax_probe` package from `src/` in editable mode. Run commands with
the environment active (`source .venv/bin/activate`) or via `uv run`.

A pip-based install is also possible: `pip install .`, then install
`torch==2.5.1` and `torchaudio==2.5.1` from the matching PyTorch index (for CPU:
`--index-url https://download.pytorch.org/whl/cpu`).

The previous `environment.yml` and `spec-file.txt` are removed.

## Repository layout

The reusable probe code is packaged under `src/`:

```
src/spoken_syntax_probe/          # importable package (side-effect-free __init__)
    preprocessing.py
    forced_alignment.py
    embedding_generation.py
    extract_segmented_embeddings.py
    treedepthprobe.py
    treekernel_prep.py
    treekernelprobe.py
    finetune.py
    utils/
        custom_classes.py
        custom_functions.py
scripts/                          # shell wrappers for the entry points
```

Run a module from the repository root with `python -m`, for example
`python -m spoken_syntax_probe.preprocessing`. The wrappers in `scripts/` do the
same and change to the repository root first, so all cwd-relative inputs and
outputs keep their original locations. The wrappers activate `.venv` when it
exists.

## Pipeline

1. `spoken_syntax_probe.preprocessing` builds the dataset csv files and the bag-of-words model.
2. `spoken_syntax_probe.forced_alignment` prepares per-utterance text and converts word-aligned TextGrid files to csv.
3. `spoken_syntax_probe.embedding_generation` extracts utterance-level layerwise embeddings.
4. `spoken_syntax_probe.extract_segmented_embeddings` extracts word-segmented embeddings. This is
   a standalone export: no script in this repo consumes its output. The probe
   scripts read the utterance-level embeddings from step 3.
5. `spoken_syntax_probe.treedepthprobe`, `spoken_syntax_probe.treekernel_prep` and
   `spoken_syntax_probe.treekernelprobe` run the probes.
6. `spoken_syntax_probe.finetune` fine-tunes wav2vec2-base on the combined corpus. This needs
   `accelerate` (in `pyproject.toml`).

## Datasets

Download [SpokenCOCO](https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz)
and [LibriSpeech](https://www.openslr.org/12), then extract them. The dataset
paths are not hardcoded; pass them on the command line.

Build the dataset csv files and the bag-of-words model:

```
python -m spoken_syntax_probe.preprocessing \
    --spokencoco_root /path/to/SpokenCOCO \
    --spokencoco_split val \
    --librispeech_root /path/to/LibriSpeech \
    --libri_split train-clean-100
```

(or `scripts/run_preprocessing.sh` with the same options.)

This writes `spokencoco_val.csv`, `librispeech_train-clean-100.csv`, the
`dataset_*.csv` files used by the extraction scripts, and `bow_model.pt`.

## Forced alignment

`spoken_syntax_probe.forced_alignment` splits the transcripts into per-utterance text files, and
converts word-aligned TextGrid files to csv. Run an external forced aligner
(for example [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/))
between the two steps.

```
# 1. write per-utterance transcripts
python -m spoken_syntax_probe.forced_alignment --libri-root /path/to/train-clean-100 \
                           --scc-root /path/to/SpokenCOCO
# 2. align with an external tool, then convert TextGrid files to csv
python -m spoken_syntax_probe.forced_alignment --tg-root /path/to/TextGrids
```

(or `scripts/run_forced_alignment.sh` with the same options.)

Options: `--libri-root`, `--scc-root`, `--scc-csv` (default
`spokencoco_val.csv`), `--tg-root`.

## Models

Text and speech models are loaded from the Hugging Face Hub by `select_model()`
in `spoken_syntax_probe.embedding_generation`:

| name | checkpoint |
|---|---|
| `wav2vec2_small` | `facebook/wav2vec2-base` |
| `wav2vec2_small_ft` | `techsword/wav2vec2-small-libri-scc-ft-ckp-10000` |
| `wav2vec2_large` | `facebook/wav2vec2-large` |
| `wav2vec2_large_ft` | `jonatasgrosman/wav2vec2-large-english` |
| `hubert_base_ls960` | `facebook/hubert-base-ls960` |
| `bert`, `bert-large` | `bert-base-uncased`, `bert-large-uncased` |
| `wav2vec2_random` | random weights from `Wav2Vec2Config` |
| `BOW` | `bow_model.pt`, built by `spoken_syntax_probe.preprocessing` |

The local fairseq checkpoint `wav2vec_small.pt` used in the paper is the
pretrained base model, which is `facebook/wav2vec2-base` in the table above. The
fine-tuned model is `techsword/wav2vec2-small-libri-scc-ft-ckp-10000`, the
paper's `checkpoint-10000`. The code loads models with `transformers`; fairseq
is not required.

### FaST-VGS

The FaST-VGS models are not on the Hugging Face Hub. Download them manually from
[jasonppy/FaST-VGS-Family](https://github.com/jasonppy/FaST-VGS-Family) and put
the checkpoints in this layout:

```
fast_vgs_family/model_path/fast-vgs-coco/
fast_vgs_family/model_path/fast-vgs-plus-coco/
```

Install the FaST-VGS package so that `fast_vgs_family` is importable (see that
repo). If you use another location, pass it with `--fast_vgs_root`.

## Extracting embeddings

Utterance-level layerwise embeddings:

```
python -m spoken_syntax_probe.embedding_generation --models wav2vec2_small \
                               --datasets dataset_spokencoco_val.csv
```

(or `scripts/run_embgen.sh` with the same options.)

Options: `--models`, `--datasets`, `--save_dir` (default `embeddings`),
`--fast_vgs_root` (default `fast_vgs_family/model_path`), `--rewrite`,
`--no_cls`.

Word-segmented embeddings:

```
python -m spoken_syntax_probe.extract_segmented_embeddings --model_path facebook/wav2vec2-base \
    --model_tag wav2vec_small \
    --dataset scc --root /path/to/SpokenCOCO --aligned_path /path/to/aligned
```

(or `scripts/run_extract_segmented.sh` with the same options.)

Options: `--model_path` (Hugging Face id or local Hugging Face checkpoint),
`--model_tag` (output-file tag; defaults to the basename of
`--model_path` with a trailing `.pt` stripped. Pass `wav2vec_small` or
`checkpoint-10000` to reproduce the published file names),
`--dataset` (`scc` or `libri`), `--csv`, `--root`, `--aligned_path`,
`--save_dir` (default `segmented_embeddings`).

## Running the probes

Tree-depth probe:

```
python -m spoken_syntax_probe.treedepthprobe >> treedepth.out
```

(or `scripts/run_treedepth.sh`.)

Tree-kernel probe:

```
python -m spoken_syntax_probe.treekernel_prep
python -m spoken_syntax_probe.treekernelprobe >> treekernel.out
```

(or `scripts/run_treekernel_prep.sh` and `scripts/run_treekernel.sh`.)

Prerequisites:

- `spoken_syntax_probe.treekernel_prep` reads every `*_generated_trees.pt` file
  in the current directory (written by `spoken_syntax_probe.preprocessing`) and
  writes the kernels to `regress-data/`.
- `spoken_syntax_probe.treekernelprobe` reads the kernels from `regress-data/`.
- Both probe scripts read embeddings from `embeddings/` in the repo root.

## Fine-tuning

`spoken_syntax_probe.finetune` fine-tunes `facebook/wav2vec2-base` on a
`combined_libri_scc_DS` dataset saved with `datasets`. It also needs a
`vocab.json` CTC vocabulary in the repo root. This file and the combined
dataset are not shipped with the repo.
