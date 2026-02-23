"""Tests for FEM readout mitigation."""

from __future__ import annotations

import numpy as np
import pytest

from error_mitigation.fem_readout import FEMReadoutMitigator


def test_characterize_builds_correct_matrices():
    mitigator = FEMReadoutMitigator(n_qubits=2)
    # Perfect calibration: preparing |0> always reads |0>, etc.
    protocol_results = {
        "0": {"0": 1000},
        "1": {"1": 1000},
    }
    mitigator.characterize(protocol_results, groups=[[0], [1]])

    for M in mitigator.cal_matrices:
        np.testing.assert_allclose(M, np.eye(2), atol=1e-12)


def test_mitigate_identity_noise_returns_original():
    mitigator = FEMReadoutMitigator(n_qubits=2)
    # Identity calibration matrices (no noise)
    protocol_results = {
        "0": {"0": 1000},
        "1": {"1": 1000},
    }
    mitigator.characterize(protocol_results, groups=[[0], [1]])

    counts = {"00": 500, "11": 500}
    result = mitigator.mitigate(counts)

    assert result["00"] == pytest.approx(0.5)
    assert result["11"] == pytest.approx(0.5)
    assert result.get("01", 0.0) == pytest.approx(0.0)
    assert result.get("10", 0.0) == pytest.approx(0.0)
