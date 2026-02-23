"""Tests for the 8-channel identity decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from quantum_circuit_cutting.channel_decomposition import (
    ChannelDecomposition,
    decompose_identity_channel,
)


def test_eight_channels_returned():
    channels = decompose_identity_channel()
    assert len(channels) == 8


def test_coefficients_sum_to_kappa():
    channels = decompose_identity_channel()
    kappa = sum(abs(ch.coefficient) for ch in channels)
    assert kappa == pytest.approx(4.0)


def test_channels_reconstruct_identity():
    """sum_i c_i * |rho_i><rho_i| * Tr(. * O_i) = I for all single-qubit rho."""
    channels = decompose_identity_channel()

    pauli = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    # For arbitrary input rho, the reconstructed output should equal rho.
    rng = np.random.default_rng(0)
    for _ in range(5):
        psi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        psi /= np.linalg.norm(psi)
        rho_in = np.outer(psi, psi.conj())

        rho_out = np.zeros((2, 2), dtype=complex)
        for ch in channels:
            O_i = pauli[ch.observable.value]
            rho_i = np.outer(ch.prep_vector, ch.prep_vector.conj())
            rho_out += ch.coefficient * np.trace(rho_in @ O_i) * rho_i

        np.testing.assert_allclose(rho_out, rho_in, atol=1e-12)


def test_enumerate_configurations_small_k():
    decomp = ChannelDecomposition(n_cuts=2)
    configs = decomp.enumerate_configurations()
    assert len(configs) == 8**2

    # Each config is a 2-tuple of channel indices in [0,7]
    for indices, coeff in configs:
        assert len(indices) == 2
        assert all(0 <= i < 8 for i in indices)
