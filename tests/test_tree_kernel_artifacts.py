"""Tree-kernel fixtures applicable to the public module set.

Ported from the private suite. The private-only structural assertions are
intentionally dropped: the public repo does not ship ``rsa.py`` or ``ewt.json``,
so the EWT slice regression and the RSA Pearson tests are omitted here.

The two tests below are deterministic, corpus-free, and CPU-only.
"""

import numpy as np
import pytest
import torch
from nltk.tree import Tree

from spoken_syntax_probe import treekernel_prep, treekernelprobe

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
