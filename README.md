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

Python 3.10 or 3.11.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CPU-only install, install torch and torchaudio from the PyTorch CPU
index first:

```
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`requirements.txt` is the single dependency file. The previous `environment.yml`
and `spec-file.txt` are removed.

## Pipeline

1. `preprocessing.py` builds the dataset csv files and the bag-of-words model.
2. `forced_alignment.py` prepares per-utterance text and converts word-aligned TextGrid files to csv.
3. `embedding_generation.py` extracts utterance-level layerwise embeddings.
4. `extract_segmented_embeddings.py` extracts word-segmented embeddings.
5. `treedepthprobe.py`, `treekernel_prep.py` and `treekernelprobe.py` run the probes.
6. `finetune.py` fine-tunes wav2vec2-base on the combined corpus.

## Datasets

Download [SpokenCOCO](https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz)
and [LibriSpeech](https://www.openslr.org/12), then extract them. The dataset
paths are not hardcoded; pass them on the command line.

Build the dataset csv files and the bag-of-words model:

```
python preprocessing.py \
    --spokencoco_root /path/to/SpokenCOCO \
    --spokencoco_split val \
    --librispeech_root /path/to/LibriSpeech \
    --libri_split train-clean-100
```

This writes `spokencoco_val.csv`, `librispeech_train-clean-100.csv`, the
`dataset_*.csv` files used by the extraction scripts, and `bow_model.pt`.

## Forced alignment

`forced_alignment.py` splits the transcripts into per-utterance text files, and
converts word-aligned TextGrid files to csv. Run an external forced aligner
(for example [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/))
between the two steps.

```
# 1. write per-utterance transcripts
python forced_alignment.py --libri-root /path/to/train-clean-100 \
                           --scc-root /path/to/SpokenCOCO
# 2. align with an external tool, then convert TextGrid files to csv
python forced_alignment.py --tg-root /path/to/TextGrids
```

Options: `--libri-root`, `--scc-root`, `--scc-csv` (default
`spokencoco_val.csv`), `--tg-root`.

## Models

Text and speech models are loaded from the Hugging Face Hub by `select_model()`
in `embedding_generation.py`:

| name | checkpoint |
|---|---|
| `wav2vec2_small` | `facebook/wav2vec2-base` |
| `wav2vec2_small_ft` | `techsword/wav2vec2-small-libri-scc-ft-ckp-10000` |
| `wav2vec2_large` | `facebook/wav2vec2-large` |
| `wav2vec2_large_ft` | `jonatasgrosman/wav2vec2-large-english` |
| `hubert_base_ls960` | `facebook/hubert-base-ls960` |
| `bert`, `bert-large` | `bert-base-uncased`, `bert-large-uncased` |
| `wav2vec2_random` | random weights from `Wav2Vec2Config` |
| `BOW` | `bow_model.pt`, built by `preprocessing.py` |

The fine-tuned checkpoint `techsword/wav2vec2-small-libri-scc-ft-ckp-10000` is
the Hugging Face equivalent of the local fairseq checkpoint `wav2vec_small.pt`
used in the paper. The code loads models with `transformers`; fairseq is not
required.

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
python embedding_generation.py --models wav2vec2_small \
                               --datasets dataset_spokencoco_val.csv
```

Options: `--models`, `--datasets`, `--save_dir` (default `embeddings`),
`--fast_vgs_root` (default `fast_vgs_family/model_path`), `--rewrite`,
`--no_cls`.

Word-segmented embeddings:

```
python extract_segmented_embeddings.py --model_path facebook/wav2vec2-base \
    --dataset scc --root /path/to/SpokenCOCO --aligned_path /path/to/aligned
```

Options: `--model_path` (Hugging Face id or local Hugging Face checkpoint),
`--dataset` (`scc` or `libri`), `--csv`, `--root`, `--aligned_path`,
`--save_dir` (default `segmented_embeddings`).

## Running the probes

Tree-depth probe:

```
python treedepthprobe.py >> treedepth.out
```

Tree-kernel probe:

```
python treekernel_prep.py
python treekernelprobe.py >> treekernel.out
```

Both probe scripts read embeddings from `embeddings/` in the repo root.

## Fine-tuning

`finetune.py` fine-tunes `facebook/wav2vec2-base` on a `combined_libri_scc_DS`
dataset saved with `datasets`. It also needs a `vocab.json` CTC vocabulary in
the repo root. This file and the combined dataset are not shipped with the repo.
