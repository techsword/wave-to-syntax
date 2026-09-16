"""Equivalence tests for the ``sp_run`` structural-probe performance changes.

Covers three changes that must not alter any value: the row-wise dependency
lengths in ``get_dep_distance_matrix`` (one BFS per token instead of one per
pair), the annotation-only ``gen_labels`` pass, and the redundancy-free
``LoadFromDisk_.__getitem__``. CPU-only; uses the packaged spaCy model.
"""

import networkx as nx
import numpy as np
import pytest
import torch

from spoken_syntax_probe.structural import sp_run

SENTENCES = [
    "The cat sat on the mat.",
    "A dog barks loudly at the mailman today.",
    "The cat sat on the mat and the cat slept.",
    "I saw the man with the telescope in the park yesterday.",
    "The quick brown fox jumps over the lazy dog.",
    "He said that she would come.",
]


def _reference_dep_distance_matrix(sent):
    """Pre-change implementation: one shortest_path_length per (i, j) pair."""
    doc = sp_run.nlp(sent)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                          '{0}'.format(child.lower_)))
    graph = nx.Graph(edges)
    M = torch.zeros((len(doc), len(doc)))
    N = torch.zeros((len(doc)))
    for i, d1 in enumerate(doc):
        N[i] = nx.shortest_path_length(
            graph, source=d1.text.lower(), target=d1.sent.root.text.lower())
        for j, d2 in enumerate(doc):
            if i > j:
                M[i, j] = M[j, i]
            else:
                M[i, j] = nx.shortest_path_length(
                    graph, source=str(d1).lower(), target=str(d2).lower())
    return M, N


@pytest.mark.parametrize("sent", SENTENCES)
def test_get_dep_distance_matrix_matches_per_pair_reference(sent):
    want_M, want_N = _reference_dep_distance_matrix(sent)
    got_M, got_N = sp_run.get_dep_distance_matrix(sent)

    assert torch.equal(got_M, want_M)
    assert torch.equal(got_N, want_N)


class _AnnotationOnly:
    """Dataset stand-in whose ``__getitem__`` records any indexing."""

    def __init__(self, annot):
        self.annot = annot
        self.indexed = False

    def __getitem__(self, idx):
        self.indexed = True
        raise AssertionError("gen_labels must read annotations, not index")


def test_gen_labels_reads_annotations_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sp_run, "get_dep_distance_matrix",
        lambda sent: ("twd:" + sent, "wd:" + sent))

    dataset = _AnnotationOnly(["a b", "c d"])
    container = sp_run.gen_labels(dataset, save_file=str(tmp_path / "labels.pt"))

    assert not dataset.indexed
    assert container == {
        0: {"sent": "a b", "twd": "twd:a b", "wd": "wd:a b"},
        1: {"sent": "c d", "twd": "twd:c d", "wd": "wd:c d"},
    }


def test_load_from_disk_getitem_matches_direct_stack():
    raw = [
        [np.full((2, 3), 1.0), np.full((2, 3), 2.0)],
        [np.full((2, 3), 3.0), np.full((2, 3), 4.0)],
    ]
    dataset = sp_run.LoadFromDisk_(
        list(zip(raw, ["a b", "c d e"], [1.5, 2.5])))

    embedding, text, audio_len, text_len = dataset[0]

    expected = torch.stack([torch.tensor(layer) for layer in raw[0]])
    assert torch.equal(embedding, expected)
    assert (text, audio_len, text_len) == ("a b", 1.5, 2)
