"""Regression tests for the ``rsa.py`` scoring fixes.

Covers the known scoring bugs: the ``'scc'``-only kernel mapping (and the
unbound ``list_of_files`` it caused for ``spokencoco_val`` kernels), the broken
Pearson branch (result-object handling and the one-layer-per-file format), the
filename-derived layer index, and the nested output directory that an
underscored embedding directory produces. Corpus-free and CPU-only.
"""

import numpy as np
import pytest
import torch
from scipy.stats import pearsonr

from spoken_syntax_probe.structural import rsa


def test_pairwise_files_for_kernel_maps_spokencoco_and_librispeech():
    files = [
        "pairwise_distances/wav2vec2-base_flat/wav2vec2-base_spokencoco_val_0_pd.pt",
        "pairwise_distances/hubert-base-ls960_flat/hubert-base-ls960_spokencoco_val_0_pd.pt",
        "pairwise_distances/wav2vec2-base_flat/wav2vec2-base_librispeech_train_0_pd.pt",
    ]
    spokencoco = rsa.pairwise_files_for_kernel(
        "regress-data/spokencoco_val_42_200anchors_regress_kernel.pt", files
    )
    librispeech = rsa.pairwise_files_for_kernel(
        "regress-data/librispeech_train_42_200anchors_regress_kernel.pt", files
    )

    # Both models' spokencoco_val files must be selected.
    assert spokencoco == [files[0], files[1]]
    assert librispeech == [files[2]]


def test_pairwise_files_for_kernel_handles_legacy_scc_names():
    # Legacy 'scc' kernels were always paired with 'spokencoco' pairwise files.
    files = ["pd/spokencoco_val_0_pd.pt", "pd/librispeech_train_0_pd.pt"]
    assert rsa.pairwise_files_for_kernel("tree_kernel/scc_delexed_0.5_kernel.pt", files) == [files[0]]


def test_pairwise_files_for_kernel_never_unbound_for_unknown_names():
    # The old 'scc'/'libri' test left list_of_files unbound when the kernel
    # matched neither; the helper always returns a list.
    assert rsa.pairwise_files_for_kernel("tree_kernel/mystery_kernel.pt", ["a.pt"]) == []


@pytest.mark.parametrize(
    "name,expected",
    [
        ("pairwise_distances/wav2vec2-base_flat/wav2vec2-base_spokencoco_val_6_pd.pt",
         ("wav2vec2-base", "spokencoco_val", 6)),
        ("pairwise_distances/hubert-base-ls960_flat/hubert-base-ls960_spokencoco_val_12_pd.pt",
         ("hubert-base-ls960", "spokencoco_val", 12)),
        ("pairwise_distances/x/wav2vec2_small_librispeech_train_3_pd.pt",
         ("wav2vec2_small", "librispeech_train", 3)),
    ],
)
def test_parse_pd_filename(name, expected):
    assert rsa.parse_pd_filename(name) == expected


def test_kernel_layer_pearson_matches_hand_computed_value():
    distance = np.array([
        [1.0, 0.9, 0.2],
        [0.9, 1.0, 0.4],
        [0.2, 0.4, 1.0],
    ])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    kernel = np.array([0.8, 0.1, 0.35])

    expected = pearsonr(kernel, np.array([0.9, 0.2, 0.4])).statistic
    got = rsa.kernel_layer_pearson(kernel, distance, pairs)

    assert got == pytest.approx(expected, abs=1e-12)


def test_kernel_layer_pearson_uses_the_pair_indices():
    # Asymmetric matrix: only the (row, col) = (test, ref) entries are correct.
    distance = np.array([
        [5.0, 1.0, 9.0],
        [9.0, 5.0, 3.0],
        [9.0, 9.0, 5.0],
    ])
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    kernel = np.array([1.0, 3.0, 9.0])  # selected distances: 1.0, 3.0, 9.0

    assert rsa.kernel_layer_pearson(kernel, distance, pairs) == pytest.approx(1.0, abs=1e-9)


def test_main_scores_a_spokencoco_kernel_without_unbound_names(tmp_path, monkeypatch):
    # Full scoring path on the real per-layer (N, N) format.
    kernel_dir = tmp_path / "tree_kernel"
    pd_dir = tmp_path / "pairwise_distances"
    kernel_dir.mkdir()
    pd_dir.mkdir()

    selected = np.array([0.9, 0.7, 0.5, 0.3])
    pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    kernel_values = np.array([0.8, 0.6, 0.4, 0.2])
    kernel_rows = np.column_stack([kernel_values, pairs])
    torch.save(kernel_rows, kernel_dir / "spokencoco_val_42_200anchors_delexed_0.5_kernel.pt")

    distance = np.eye(4)
    distance[0, 1] = distance[1, 0] = 0.9
    distance[1, 2] = distance[2, 1] = 0.7
    distance[2, 3] = distance[3, 2] = 0.5
    distance[0, 3] = distance[3, 0] = 0.3
    torch.save(distance, pd_dir / "wav2vec2-base_spokencoco_val_0_pd.pt")

    monkeypatch.chdir(tmp_path)
    rsa.main(alpha=0.5, seed=42, delexed=True,
             tree_kernel_path=str(kernel_dir), pairwise_distance_path=str(pd_dir))

    expected = pearsonr(kernel_values, selected).statistic
    got = rsa.kernel_layer_pearson(kernel_values, distance, pairs)
    assert got == pytest.approx(expected, abs=1e-12)


def test_kernel_matches_run_discovers_regress_and_legacy_names():
    regress = "regress-data/spokencoco_val_42_200anchors_regress_kernel.pt"
    assert rsa.kernel_matches_run(regress, "42", True, "0.5")
    assert not rsa.kernel_matches_run(
        "regress-data/spokencoco_val_7_200anchors_regress_kernel.pt", "42", True, "0.5")

    legacy = "tree_kernel/spokencoco_val_42_200anchors_delexed_0.5_kernel.pt"
    assert rsa.kernel_matches_run(legacy, "42", True, "0.5")
    assert not rsa.kernel_matches_run(
        "tree_kernel/spokencoco_val_42_200anchors_0.5_kernel.pt", "42", True, "0.5")
    assert rsa.kernel_matches_run(
        "tree_kernel/spokencoco_val_42_200anchors_0.5_kernel.pt", "42", False, "0.5")
    assert not rsa.kernel_matches_run(
        "tree_kernel/spokencoco_val_42_200anchors_delexed_0.6_kernel.pt", "42", True, "0.5")


def test_main_scores_a_regress_kernel_end_to_end(tmp_path, monkeypatch):
    kernel_dir = tmp_path / "regress-data"
    pd_dir = tmp_path / "pairwise_distances"
    kernel_dir.mkdir()
    pd_dir.mkdir()

    # treekernel_prep regress format: a list of per-test-point (n_anchors, 3).
    rows = np.array([
        [0.8, 0, 0],
        [0.9, 1, 0],
        [0.7, 0, 1],
        [0.6, 1, 1],
    ])
    torch.save([rows], kernel_dir / "spokencoco_val_42_2anchors_regress_kernel.pt")

    distance = np.eye(2)
    distance[0, 1] = distance[1, 0] = 0.5
    torch.save(distance, pd_dir / "wav2vec2-base_spokencoco_val_0_pd.pt")

    calls = []
    original = rsa.kernel_layer_pearson

    def spy(kernel_values, distance_matrix, kernel_pairs):
        calls.append((np.array(kernel_values), np.array(kernel_pairs)))
        return original(kernel_values, distance_matrix, kernel_pairs)

    monkeypatch.setattr(rsa, "kernel_layer_pearson", spy)
    monkeypatch.chdir(tmp_path)
    rsa.main(alpha=0.5, seed=42, delexed=True,
             tree_kernel_path=str(kernel_dir), pairwise_distance_path=str(pd_dir))

    assert len(calls) == 1
    kernel_values, pairs = calls[0]
    assert np.allclose(kernel_values, [0.8, 0.9, 0.7, 0.6])
    assert np.array_equal(pairs, np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))


def test_pairwise_distances_create_nested_dir_for_underscored_embedding_path(tmp_path, monkeypatch):
    """Regression: an underscore in the embedding *directory* nests the output.

    ``datasetname`` is ``embedding_file.split('_', 1)[1]``, so a directory name
    such as ``rsa_flat`` leaks a path separator into the save name. A fresh run
    directory must not raise ``FileNotFoundError``, and the nested filename
    convention must stay unchanged. Paths stay relative, as in the CLI run.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rsa_flat").mkdir()
    embeddings = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    torch.save((embeddings, None, ["a b", "c d"], None, None, None),
               tmp_path / "rsa_flat" / "wav2vec2-base_spokencoco_val_extracted.pt")

    rsa.compute_pairwise_dist_for_embs(
        embedding_path="rsa_flat", pd_save_path="pairwise_distances", device="cpu")

    nested = tmp_path / "pairwise_distances" / "wav2vec2-base_flat"
    layer_0 = nested / "wav2vec2-base_spokencoco_val_0_pd.pt"
    assert layer_0.is_file()
    assert (nested / "wav2vec2-base_spokencoco_val_1_pd.pt").is_file()
    assert torch.load(layer_0, weights_only=False).shape == (2, 2)


def test_pairwise_distance_calc_keeps_flat_convention_without_underscore(tmp_path, monkeypatch):
    """The fix only creates directories; the flat path convention is unchanged."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "embeddings").mkdir()
    embeddings = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    torch.save((embeddings, None, ["a b", "c d"], None, None, None),
               tmp_path / "embeddings" / "wav2vec2-base_spokencoco_val_extracted.pt")

    rsa.pairwise_distance_calc(
        "embeddings/wav2vec2-base_spokencoco_val_extracted.pt",
        "pairwise_distances", device="cpu")

    assert (tmp_path / "pairwise_distances" /
            "wav2vec2-base_spokencoco_val_0_pd.pt").is_file()
