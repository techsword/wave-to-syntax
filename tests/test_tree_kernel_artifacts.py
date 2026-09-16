"""Fixtures (b) and (c): tree-kernel filenames/splits and EWT regression.

Fixture (b) builds six synthetic trees (one longer than the word limit) and
checks the tree-kernel output filename, the anchor/test split, and the
filename-driven word limit.

Fixture (c) slices the tracked ``ewt.json`` and checks the tree-kernel values
and the RSA Pearson score. Both fixtures are deterministic, corpus-free, and
CPU-only.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from nltk.tree import Tree

from spoken_syntax_probe import treekernel_prep, treekernelprobe
from spoken_syntax_probe.structural import rsa

REPO_ROOT = Path(__file__).resolve().parents[1]

_SHORT_TREES = [
    ("(S (NP (DT The) (NN cat)) (VP (VBZ sleeps)))", "The cat sleeps"),
    ("(S (NP (DT A) (NN dog)) (VP (VBZ runs) (ADVP (RB fast))))", "A dog runs fast"),
    ("(S (NP (PRP We)) (VP (VBP like) (NP (NN pizza))))", "We like pizza"),
    ("(S (NP (NNP Anna)) (VP (VBZ reads) (NP (DT a) (NN book))))", "Anna reads a book"),
    ("(S (NP (DT The) (NN baby)) (VP (VBZ cries)))", "The baby cries"),
]

_LONG_WORDS = 25


def _six_synthetic_trees():
    short = [(Tree.fromstring(text), sentence) for text, sentence in _SHORT_TREES]
    tokens = " ".join("(NN w%d)" % i for i in range(_LONG_WORDS - 1))
    long_tree = Tree.fromstring("(S (NP (DT the) %s) (VP (VBZ is)))" % tokens)
    long_sentence = " ".join("w%d" % i for i in range(_LONG_WORDS))
    return short + [(long_tree, long_sentence)]


@pytest.mark.parametrize(
    "tree_stem,kernel_stem,expected_test_count",
    [
        ("wav2vec2_small_spokencoco", "wav2vec2_small", 2),  # limit 20 drops the long tree
        ("wav2vec2_librispeech", "wav2vec2_librispeech", 3),  # limit 52 keeps all six trees
    ],
)
def test_kernel_filename_and_anchor_split(tmp_path, monkeypatch, tree_stem, kernel_stem, expected_test_count):
    monkeypatch.chdir(tmp_path)
    tree_name = "%s_generated_trees.pt" % tree_stem
    tree_file = tmp_path / tree_name
    torch.save(_six_synthetic_trees(), tree_file)

    # Real usage calls this with bare filenames from the generation directory,
    # so keep the argument cwd-relative.
    treekernel_prep.generate_kernel_regress(
        [tree_name],
        seed=42,
        alpha=0.5,
        num_anchors=2,
        save_path=str(tmp_path),
        parallel=False,
        rewrite=True,
    )

    # The kernel stem is the first two underscore-separated tokens of the input.
    kernel_file = tmp_path / ("%s_42_2anchors_regress_kernel.pt" % kernel_stem)
    assert kernel_file.is_file()

    container = torch.load(kernel_file)
    assert len(container) == expected_test_count
    assert np.array(container[0]).shape == (2, 3)


@pytest.mark.parametrize(
    "name,expected_limit",
    [
        ("wav2vec2_librispeech_42_2anchors_regress_kernel.pt", 52),
        ("wav2vec2_spokencoco_42_2anchors_regress_kernel.pt", 20),
    ],
)
def test_load_tree_kernel_filename_word_limit(tmp_path, name, expected_limit):
    rows = [
        np.array([0.5, 0, 0]),
        np.array([0.7, 1, 0]),
        np.array([0.2, 0, 1]),
        np.array([0.9, 1, 1]),
    ]
    torch.save(rows, tmp_path / name)

    tk, ref_idx, test_idx, limit = treekernelprobe.load_tree_kernel(str(tmp_path / name))

    assert limit == expected_limit
    assert tk.shape == (2, 2)
    assert ref_idx.tolist() == [0, 1]
    assert test_idx.tolist() == [0, 1]


def _ewt_slice(count=3):
    data = json.loads((REPO_ROOT / "ewt.json").read_text())
    return data["test"][:count]


def _normalized_kernel(a, b):
    from ursa.kernel import Kernel, delex

    kernel = Kernel(alpha=0.5)
    delexed_a, delexed_b = delex(a), delex(b)
    denominator = (kernel(delexed_a, delexed_a) * kernel(delexed_b, delexed_b)) ** 0.5
    return kernel(delexed_a, delexed_b) / denominator


def test_ewt_slice_tree_kernel_regression():
    trees = [Tree.fromstring(item["tree"]) for item in _ewt_slice()]
    kernel_vector = np.array(
        [
            _normalized_kernel(trees[0], trees[1]),
            _normalized_kernel(trees[0], trees[2]),
            _normalized_kernel(trees[1], trees[2]),
        ]
    )
    np.testing.assert_allclose(
        kernel_vector,
        [0.0992966161862378, 0.11985340431738455, 0.14494041391118112],
        rtol=1e-9,
        atol=1e-12,
    )


def test_ewt_slice_rsa_regression():
    from torchmetrics.functional import pairwise_cosine_similarity

    items = _ewt_slice()
    trees = [Tree.fromstring(item["tree"]) for item in items]
    kernel_vector = np.array(
        [
            _normalized_kernel(trees[0], trees[1]),
            _normalized_kernel(trees[0], trees[2]),
            _normalized_kernel(trees[1], trees[2]),
        ]
    )

    def features(item):
        tokens = item["sent"].split()
        punctuation = sum(1 for token in tokens if any(c in token for c in ",.!?"))
        longest = max(len(token) for token in tokens)
        mean_length = sum(len(token) for token in tokens) / len(tokens)
        return [len(tokens), punctuation, longest, mean_length]

    embeddings = torch.tensor([features(item) for item in items], dtype=torch.float)
    similarity = pairwise_cosine_similarity(embeddings).numpy()
    similarity_vector = np.array([similarity[0, 1], similarity[0, 2], similarity[1, 2]])

    assert rsa.pearson_r_score(kernel_vector, similarity_vector) == pytest.approx(
        -0.27524623274417354, abs=1e-9
    )
