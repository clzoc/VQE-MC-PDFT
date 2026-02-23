"""Tests for reconstruction."""

from __future__ import annotations

import pytest

from quantum_circuit_cutting.reconstruction import _parity_from_counts


def test_parity_from_counts_known():
    # All |00> -> parity on qubit 0 is +1
    counts = {"00": 100}
    assert _parity_from_counts(counts, [0]) == pytest.approx(1.0)

    # All |10> -> qubit 0 is '1', parity = -1
    counts = {"10": 100}
    assert _parity_from_counts(counts, [0]) == pytest.approx(-1.0)

    # 50/50 mix -> parity = 0
    counts = {"00": 50, "10": 50}
    assert _parity_from_counts(counts, [0]) == pytest.approx(0.0)


def test_parity_from_counts_two_qubit():
    # |11> -> parity on both qubits = (-1)^2 = +1
    counts = {"11": 100}
    assert _parity_from_counts(counts, [0, 1]) == pytest.approx(1.0)

    # |10> -> parity on both qubits = (-1)^1 = -1
    counts = {"10": 100}
    assert _parity_from_counts(counts, [0, 1]) == pytest.approx(-1.0)


def test_reconstruct_expectation_single_cluster():
    """Trivial case: 1 cluster, 0 cuts -> direct expectation value."""
    from quantum_circuit_cutting.channel_decomposition import ChannelDecomposition
    from quantum_circuit_cutting.partition import ClusterPartition
    from quantum_circuit_cutting.reconstruction import CuttingReconstructor
    import networkx as nx

    partition = ClusterPartition(
        clusters=[[0, 1]],
        inter_cluster_edges=[],
        cluster_graph=nx.Graph(),
        n_cuts=0,
    )
    decomp = ChannelDecomposition(n_cuts=0)
    recon = CuttingReconstructor(partition, decomp)

    # All |00> -> <ZI> = +1
    fragment_results = {(0, 0): {"00": 1000}}
    val = recon.reconstruct_expectation(fragment_results, "ZI")
    assert val == pytest.approx(1.0)
