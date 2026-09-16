"""Structural-probe surface contract (public tree).

Adapted from the private suite. The private version also asserted that the
private-only legacy names (the root compatibility shims, the ``struct_probes``
namespace, the ``utils.struct_probe_utils`` facade) are gone. Those names never
existed in the public repository, so those assertions are dropped here. What
remains pins the public structural surface:

1. ``flat_probe_utils`` is the canonical flat-schema module;
2. the runner imports the canonical flat probes;
3. the relocated ``model``/``regimen`` modules live in the package and back the
   runner.
"""

import importlib

import spoken_syntax_probe.structural.flat_probe_utils as flat

CANONICAL_SYMBOLS = [
    "Probe",
    "TwoWordPSDProbe",
    "OneWordPSDProbe",
    "Reporter",
    "WordPairReporter",
    "WordReporter",
    "UnionFind",
    "prims_matrix_to_edges",
    "get_nopunct_argmin",
    "L1DistanceLoss",
    "L1DepthLoss",
    "seed_everything",
]


def test_flat_probe_utils_has_the_canonical_symbols():
    for name in CANONICAL_SYMBOLS:
        assert hasattr(flat, name), name


def test_canonical_runner_uses_flat_probe_utils():
    runner = importlib.import_module("spoken_syntax_probe.structural.sp_run")
    for name in [
        "OneWordPSDProbe",
        "TwoWordPSDProbe",
        "L1DistanceLoss",
        "L1DepthLoss",
        "WordPairReporter",
        "WordReporter",
        "seed_everything",
    ]:
        assert getattr(runner, name) is getattr(flat, name)


def test_relocated_models_live_in_the_package():
    model = importlib.import_module("spoken_syntax_probe.structural.model")
    regimen = importlib.import_module("spoken_syntax_probe.structural.regimen")
    assert hasattr(model, "DiskModel")
    assert hasattr(regimen, "ProbeRegimen")


def test_canonical_runner_uses_the_relocated_models():
    runner = importlib.import_module("spoken_syntax_probe.structural.sp_run")
    model = importlib.import_module("spoken_syntax_probe.structural.model")
    regimen = importlib.import_module("spoken_syntax_probe.structural.regimen")
    assert runner.DiskModel is model.DiskModel
    assert runner.ProbeRegimen is regimen.ProbeRegimen
