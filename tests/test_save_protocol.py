"""torch.save protocol-5 round-trip for the large structural artefacts.

Both patched call sites save many-small-array objects (per-layer N*N cosine
similarity matrices; per-utterance feature lists). Verify that the protocol-5
legacy settings used at those call sites round-trip an identical object, and
that the call sites keep using those settings. No corpora are needed.
"""

from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

KWARGS = {"pickle_protocol": 5, "_use_new_zipfile_serialization": False}

PATCHED_FILES = [
    REPO_ROOT / "src" / "spoken_syntax_probe" / "structural" / "rsa.py",
    REPO_ROOT / "src" / "spoken_syntax_probe" / "extract_segmented_embeddings.py",
    REPO_ROOT / "src" / "spoken_syntax_probe" / "structural" / "ewt_test.py",
    REPO_ROOT / "src" / "spoken_syntax_probe" / "treekernel_prep.py",
    REPO_ROOT / "src" / "spoken_syntax_probe" / "embedding_generation.py",
    REPO_ROOT / "src" / "spoken_syntax_probe" / "preprocessing.py",
]


def test_protocol5_round_trip_preserves_numpy_array(tmp_path):
    rng = np.random.default_rng(0)
    matrix = rng.random((6, 6))
    obj = (matrix + matrix.T) / 2.0

    path = tmp_path / "layer_pd.pt"
    torch.save(obj, path, **KWARGS)
    loaded = torch.load(path, weights_only=False)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == obj.shape
    np.testing.assert_array_equal(loaded, obj)


def test_protocol5_round_trip_preserves_many_small_arrays(tmp_path):
    rng = np.random.default_rng(1)
    # Many small per-utterance/layer arrays: the shape that triggered the
    # pickle-protocol-2 memory blowup.
    obj = [
        {
            "emb": [rng.random((3, 4)), rng.random((3, 4))],
            "text": "a b c",
            "audio_len": 1.0,
        }
        for _ in range(20)
    ]

    path = tmp_path / "extracted.pt"
    torch.save(obj, path, **KWARGS)
    loaded = torch.load(path, weights_only=False)

    assert len(loaded) == len(obj)
    for got, want in zip(loaded, obj):
        assert got["text"] == want["text"]
        assert got["audio_len"] == want["audio_len"]
        for got_array, want_array in zip(got["emb"], want["emb"]):
            np.testing.assert_array_equal(got_array, want_array)


def test_patched_call_sites_use_protocol5_legacy():
    for path in PATCHED_FILES:
        source = path.read_text()
        assert "pickle_protocol=5" in source, path
        assert "_use_new_zipfile_serialization=False" in source, path
