"""End-to-end integration test for the circuit cutting round-trip.

Verifies that partitioning a 4-qubit circuit into two 2-qubit fragments,
generating fragment circuits, executing them (simulated), and reconstructing
the expectation value produces a numerically consistent result.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from quantum_circuit_cutting.channel_decomposition import ChannelDecomposition
from quantum_circuit_cutting.fragment_circuits import FragmentCircuitGenerator
from quantum_circuit_cutting.partition import ClusterPartition
from quantum_circuit_cutting.reconstruction import CuttingReconstructor


def test_round_trip_4qubit_cutting():
    """Partition 4 qubits into [0,1] and [2,3], cut 1 edge, reconstruct."""
    partition = ClusterPartition(
        clusters=[[0, 1], [2, 3]],
        inter_cluster_edges=[(1, 2)],
        cluster_graph=nx.Graph([(0, 1)]),
        n_cuts=1,
    )

    decomposition = ChannelDecomposition(n_cuts=1)
    assert decomposition.n_configurations == 8

    # Use small fixed params
    params = np.array([0.1, 0.2, 0.3, 0.4])

    generator = FragmentCircuitGenerator(
        partition=partition,
        decomposition=decomposition,
        ansatz_params=params,
        n_electrons=2,
        n_layers=2,
    )

    all_fragments = generator.generate_all(use_sampling=False)
    assert len(all_fragments) == 8  # 8^1 configurations

    # Each configuration should produce 2 fragments (one per cluster)
    for fragments in all_fragments:
        assert len(fragments) == 2
        for frag in fragments:
            assert frag.n_qubits == 2

    # Flatten and verify metadata
    circuits, metadata = generator.flatten_circuits(all_fragments)
    assert len(circuits) == 16  # 8 configs * 2 clusters
    assert all(0 <= ci <= 1 for _, ci in metadata)

    # Simulate execution with deterministic fake counts
    rng = np.random.default_rng(42)
    fragment_results = {}
    for config_idx, cluster_idx in metadata:
        counts = {format(i, "02b"): int(rng.integers(10, 100)) for i in range(4)}
        fragment_results[(config_idx, cluster_idx)] = counts

    # Reconstruct a simple Pauli string
    reconstructor = CuttingReconstructor(partition, decomposition)
    configs = decomposition.enumerate_configurations()

    # <ZIII> should be a finite real number
    val_zi = reconstructor.reconstruct_expectation(fragment_results, "ZIII", configs)
    assert np.isfinite(val_zi)

    # <IIZI> should also be finite
    val_iz = reconstructor.reconstruct_expectation(fragment_results, "IIZI", configs)
    assert np.isfinite(val_iz)

    # Reconstruct a trivial Hamiltonian
    ham = {"ZIII": 1.0, "IIZI": -0.5, "IIII": 3.0}
    energy = reconstructor.reconstruct_hamiltonian_energy(
        fragment_results, ham, configs
    )
    assert np.isfinite(energy)
    # Identity term contributes exactly 3.0
    assert abs(energy - (3.0 + val_zi - 0.5 * val_iz)) < 1e-12


def test_single_cluster_no_cuts():
    """Trivial case: 1 cluster, 0 cuts should give direct expectation."""
    partition = ClusterPartition(
        clusters=[[0, 1]],
        inter_cluster_edges=[],
        cluster_graph=nx.Graph(),
        n_cuts=0,
    )
    decomposition = ChannelDecomposition(n_cuts=0)
    reconstructor = CuttingReconstructor(partition, decomposition)

    # All |00> -> <ZI> = +1, <IZ> = +1
    fragment_results = {(0, 0): {"00": 1000}}
    configs = decomposition.enumerate_configurations()

    assert reconstructor.reconstruct_expectation(fragment_results, "ZI", configs) == 1.0
    assert reconstructor.reconstruct_expectation(fragment_results, "IZ", configs) == 1.0

    # All |11> -> <ZI> = -1, <IZ> = -1, <ZZ> = +1
    fragment_results = {(0, 0): {"11": 1000}}
    assert reconstructor.reconstruct_expectation(fragment_results, "ZZ", configs) == 1.0
