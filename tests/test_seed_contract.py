"""Seed contract: the default is a no-op; an explicit seed is reproducible.

Phase 0 found that the runner trains probes without setting a seed. Phase 4
adds an opt-in ``--seed``. These tests pin the two required properties:

1. ``seed_everything(None)`` does not touch any RNG state, so runs without the
   flag keep the original RNG stream exactly;
2. an explicit seed makes probe initialisation and a tiny training run
   reproducible.
"""

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from spoken_syntax_probe.structural.flat_probe_utils import (
    L1DistanceLoss,
    TwoWordPSDProbe,
    seed_everything,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rng_state():
    return (
        random.getstate(),
        np.random.get_state()[1].copy(),
        torch.random.get_rng_state().clone(),
    )


def _assert_rng_state_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert torch.equal(left[2], right[2])


def test_seed_none_is_a_noop_for_all_rngs():
    before = _rng_state()
    seed_everything(None)
    _assert_rng_state_equal(before, _rng_state())


def test_explicit_seed_reproduces_probe_initialisation():
    args = {"device": "cpu", "hidden_dim": 8}
    seed_everything(1234)
    first = TwoWordPSDProbe(args).proj.detach().clone()
    seed_everything(1234)
    second = TwoWordPSDProbe(args).proj.detach().clone()
    assert torch.equal(first, second)


def test_different_seeds_change_probe_initialisation():
    args = {"device": "cpu", "hidden_dim": 8}
    seed_everything(1)
    first = TwoWordPSDProbe(args).proj.detach().clone()
    seed_everything(2)
    second = TwoWordPSDProbe(args).proj.detach().clone()
    assert not torch.equal(first, second)


def test_fixed_seed_reproduces_a_tiny_training_run():
    args = {"device": "cpu", "hidden_dim": 4}

    def run(seed):
        seed_everything(seed)
        probe = TwoWordPSDProbe(args)
        loss_fn = L1DistanceLoss(args)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
        batch = torch.randn(2, 3, 4)
        length = torch.tensor([3, 3])
        labels = torch.rand(2, 3, 3)
        for _ in range(2):
            optimizer.zero_grad()
            loss, _ = loss_fn(probe(batch), labels, length)
            loss.backward()
            optimizer.step()
        return probe.proj.detach().clone()

    assert torch.equal(run(7), run(7))


def test_seed_everything_is_exposed_by_the_canonical_module():
    import spoken_syntax_probe.structural.flat_probe_utils as canonical

    assert canonical.seed_everything is seed_everything


def test_runner_help_exposes_seed():
    result = subprocess.run(
        [sys.executable, "-m", "spoken_syntax_probe.structural.sp_run", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--seed" in result.stdout
