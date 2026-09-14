"""Regression test for the tree-kernel corpus-name mapping.

Kernel files are named ``spokencoco_val_42_200anchors_regress_kernel.pt`` and
``librispeech_train_42_200anchors_regress_kernel.pt``. The old mapping only
matched the legacy ``scc`` token, so SpokenCOCO kernels silently selected
librispeech embedding files (no pairing). These tests pin the corrected
mapping. No corpora required.
"""

from spoken_syntax_probe.treekernelprobe import kernel_dataset_name

SPOKENCOCO_KERNEL = "regress-data/spokencoco_val_42_200anchors_regress_kernel.pt"
LIBRISPEECH_KERNEL = "regress-data/librispeech_train_42_200anchors_regress_kernel.pt"


def test_spokencoco_kernel_maps_to_spokencoco():
    assert kernel_dataset_name(SPOKENCOCO_KERNEL) == "spokencoco"


def test_librispeech_kernel_maps_to_librispeech():
    assert kernel_dataset_name(LIBRISPEECH_KERNEL) == "librispeech"


def test_legacy_scc_token_still_maps_to_spokencoco():
    assert kernel_dataset_name("regress-data/scc_val_42_200anchors_regress_kernel.pt") == "spokencoco"


def test_mapping_selects_the_matching_embedding_file():
    datasetname = kernel_dataset_name(SPOKENCOCO_KERNEL)
    embedding_files = [
        "embeddings/wav2vec_small_spokencoco_extracted.pt",
        "embeddings/wav2vec_small_librispeech_extracted.pt",
    ]

    assert [x for x in embedding_files if datasetname in x] == [embedding_files[0]]
