"""Bit-identical equivalence for the tree-kernel precompute + batching change.

The performance change memoizes anchor delexicalisation and anchor
self-kernels, and batches the joblib dispatches. It must not change a single
kernel value. These tests compare the new implementation against an inline
naive reference (today's semantics: delex + self-kernel recomputed for every
test point) for ``normalize=True``/``False`` and for the sequential and
parallel paths. Corpus-free and CPU-only.
"""

import random

import numpy as np
import pytest
import torch
from nltk.tree import Tree
from ursa.kernel import Kernel, delex

from spoken_syntax_probe import treekernel_prep

TREE_COUNT = 20
NUM_ANCHORS = 4
WORD_LIMIT = 20  # spokencoco default in generate_kernel_regress


def _synthetic_tree(i):
    n = 3 + (i % 4)
    leaves = " ".join("(NN w%d)" % j for j in range(n))
    if i % 2:
        frame = ("(S (NP (DT the) %s) (VP (VBZ is) "
                 "(PP (IN in) (NP (DT a) (NN box)))))" % leaves)
        words = ["w%d" % j for j in range(n)] + ["in", "a", "box"]
    else:
        frame = "(S (NP (DT the) %s) (VP (VBZ is)))" % leaves
        words = ["w%d" % j for j in range(n)]
    return Tree.fromstring(frame), " ".join(words)


def _tree_list(count=TREE_COUNT):
    return [_synthetic_tree(i) for i in range(count)]


def _reference_container(tree_list, seed, alpha, num_anchors, normalization):
    """Inline copy of the pre-change semantics (recompute per test point)."""
    random.seed(seed)
    kernel = Kernel(alpha=alpha)
    filtered = [x for x in tree_list if len(str.split(x[1])) < WORD_LIMIT]
    trees_filtered = [[item, i] for i, item in enumerate(filtered)]
    random.shuffle(trees_filtered)

    ref_pts = trees_filtered[:num_anchors]
    test_pts = trees_filtered[num_anchors + 1:]
    return [
        treekernel_prep.compute_kernel(kernel, test_pt, ref_pts, normalization)
        for test_pt in test_pts
    ]


@pytest.mark.parametrize("normalize", [True, False])
def test_compute_kernel_precomputed_matches_legacy(normalize):
    kernel = Kernel(alpha=0.5)
    trees = _tree_list(6)
    anchors = [[trees[i], i] for i in range(4)]
    test_pt = [trees[4], 4]

    anchor_trees = [delex(anchor[0][0]) for anchor in anchors]
    anchor_self = [kernel(tree, tree) for tree in anchor_trees]

    legacy = treekernel_prep.compute_kernel(kernel, test_pt, anchors, normalize)
    new = treekernel_prep.compute_kernel(
        kernel, test_pt, anchors, normalize, anchor_trees, anchor_self
    )

    assert np.array_equal(legacy, new)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("parallel,n_jobs", [(False, treekernel_prep.default_n_jobs()), (True, 2)])
def test_generate_kernel_regress_is_bit_identical(
    tmp_path, monkeypatch, normalize, parallel, n_jobs
):
    monkeypatch.chdir(tmp_path)
    tree_name = "wav2vec2_small_spokencoco_generated_trees.pt"
    torch.save(_tree_list(), tmp_path / tree_name)

    seed, alpha = 42, 0.5
    reference = _reference_container(_tree_list(), seed, alpha, NUM_ANCHORS, normalize)

    treekernel_prep.generate_kernel_regress(
        [tree_name],
        seed=seed,
        alpha=alpha,
        num_anchors=NUM_ANCHORS,
        save_path=str(tmp_path),
        normalization=normalize,
        parallel=parallel,
        rewrite=True,
        n_jobs=n_jobs,
    )

    kernel_file = tmp_path / ("wav2vec2_small_%d_%danchors_regress_kernel.pt"
                              % (seed, NUM_ANCHORS))
    container = torch.load(kernel_file, weights_only=False)

    assert len(container) == len(reference)
    for got, want in zip(container, reference):
        assert np.array_equal(got, want)


def test_n_jobs_is_capped_and_never_all_cores():
    assert treekernel_prep.effective_n_jobs(-1) == treekernel_prep.default_n_jobs()
    assert treekernel_prep.effective_n_jobs(0) == treekernel_prep.default_n_jobs()
    assert treekernel_prep.effective_n_jobs(2) == 2
    assert treekernel_prep.effective_n_jobs(64) == 64


def test_default_n_jobs_follows_environment(monkeypatch):
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    expected_affinity = max(1, len(treekernel_prep.os.sched_getaffinity(0)))
    assert treekernel_prep.default_n_jobs() == expected_affinity
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '7')
    assert treekernel_prep.default_n_jobs() == 7
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', 'not-a-number')
    assert treekernel_prep.default_n_jobs() == expected_affinity
