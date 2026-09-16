"""Equivalence tests for the treedepthprobe caching + per-layer parallelism.

The changes must not alter the fitted scores: the embedding-file cache only
removes repeated ``torch.load`` calls, and the per-layer joblib dispatch must
reproduce the serial fit. CPU-only; synthetic embeddings.
"""

import os

import numpy as np
import pytest
import torch

from spoken_syntax_probe import treedepthprobe as tdp


def _write_extracted(path, n=32, layers=3, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = [rng.random((layers, dim)).astype("float32") for _ in range(n)]
    labels = [float(rng.integers(0, 4)) for _ in range(n)]
    wordcount = [int(rng.integers(1, 8)) for _ in range(n)]
    audiolen = [float(rng.random() * 4 + 0.5) for _ in range(n)]
    torch.save([embeddings, labels, None, None, wordcount, audiolen], path)
    return path


def test_load_embedding_file_reads_each_file_once(tmp_path, monkeypatch):
    path = _write_extracted(tmp_path / "wav2vec2-base_x_extracted.pt")
    tdp._EMBEDDING_CACHE.clear()

    calls = {"n": 0}
    real_load = torch.load

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(tdp.torch, "load", counting_load)

    first = tdp.load_embedding_file(str(path))
    second = tdp.load_embedding_file(str(path))

    assert calls["n"] == 1
    assert first is second
    tdp._EMBEDDING_CACHE.clear()


def test_effective_n_jobs_honours_env_override(monkeypatch):
    monkeypatch.delenv("TREEDEPTHPROBE_N_JOBS", raising=False)
    assert tdp.effective_n_jobs(1) == 1
    assert tdp.effective_n_jobs(3) == 3
    assert tdp.effective_n_jobs(None) >= 1

    monkeypatch.setenv("TREEDEPTHPROBE_N_JOBS", "1")
    assert tdp.effective_n_jobs(None) == 1
    monkeypatch.setenv("TREEDEPTHPROBE_N_JOBS", "4")
    assert tdp.effective_n_jobs(None) == 4
    monkeypatch.setenv("TREEDEPTHPROBE_N_JOBS", "not-a-number")
    assert tdp.effective_n_jobs(None) >= 1


def _reference_run_model(embedding_file, reg_model_name='ridge', combi=False):
    """Pre-change loop: one reused estimator, one dict printed per layer."""
    reg_model = tdp.load_regression_model(reg_model_name)
    embeddings, labels, _, _, wordcount, audiolen = \
        tdp.load_embedding_file(embedding_file)
    embeddings = np.array(embeddings)
    labels = torch.tensor(labels).numpy()

    lookup_table = None
    if combi:
        wordcount = torch.tensor(wordcount).numpy()
        audiolen = torch.tensor(audiolen).numpy()
        lookup_table = {'wordcount': wordcount,
                        'audiolen': audiolen,
                        'wordcount+audiolen': np.column_stack((wordcount, audiolen))}

    modelname = os.path.basename(embedding_file).split('_', 1)[0]
    datasetname = embedding_file.split('_', 1)[1].replace('_extracted.pt', '')
    out = []
    for layer in range(embeddings.shape[1]):
        layer_embs = embeddings[:, layer, :]
        if not combi:
            results = tdp.model_fitting(X=layer_embs, y=labels, model=reg_model)
            out.append({'modelname': modelname, 'datasetname': datasetname,
                        'layer': layer, 'feature': 'EMB'} | results)
        else:
            for feat_name in lookup_table:
                results = tdp.model_fitting(
                    X=np.column_stack((layer_embs, lookup_table[feat_name])),
                    y=labels, model=reg_model)
                out.append({'modelname': modelname, 'datasetname': datasetname,
                            'layer': layer, 'feature': 'EMB+' + feat_name}
                           | results)
    return out


@pytest.mark.parametrize("combi", [False, True])
def test_run_model_serial_matches_pre_change_loop(tmp_path, combi):
    path = str(_write_extracted(tmp_path / "wav2vec2-base_x_extracted.pt"))

    tdp._EMBEDDING_CACHE.clear()
    reference = _reference_run_model(path, combi=combi)
    tdp._EMBEDDING_CACHE.clear()
    got = tdp.run_model([path], combi=combi, n_jobs=1)
    tdp._EMBEDDING_CACHE.clear()

    assert len(got) == len(reference)
    for entry, want in zip(got, reference):
        assert entry["layer"] == want["layer"]
        assert entry["feature"] == want["feature"]
        # Pre-change loop shares one estimator across layers; the refactor
        # builds a fresh one per layer. fit() resets it, so hold to 1e-12.
        assert entry["r2score"] == pytest.approx(want["r2score"], abs=1e-12)
        assert entry["mse"] == pytest.approx(want["mse"], abs=1e-12)
        assert entry["model_alpha"] == want["model_alpha"]


@pytest.mark.parametrize("combi", [False, True])
def test_run_model_serial_and_parallel_agree(tmp_path, combi):
    path = str(_write_extracted(tmp_path / "wav2vec2-base_x_extracted.pt"))

    tdp._EMBEDDING_CACHE.clear()
    serial = tdp.run_model([path], combi=combi, n_jobs=1)
    tdp._EMBEDDING_CACHE.clear()
    parallel = tdp.run_model([path], combi=combi, n_jobs=2)
    tdp._EMBEDDING_CACHE.clear()

    assert len(serial) == len(parallel)
    assert [entry["layer"] for entry in serial] == \
        [entry["layer"] for entry in parallel]
    for got, want in zip(parallel, serial):
        assert got["feature"] == want["feature"]
        assert got["r2score"] == pytest.approx(want["r2score"], abs=1e-6)
        assert got["mse"] == pytest.approx(want["mse"], abs=1e-6)
        assert got["model_alpha"] == want["model_alpha"]
