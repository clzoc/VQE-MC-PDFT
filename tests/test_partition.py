"""Tests for circuit partitioning."""

from __future__ import annotations

from quantum_circuit_cutting.partition import CircuitPartitioner


def test_greedy_partition_respects_max_cluster_size():
    n_qubits = 10
    max_size = 4
    partitioner = CircuitPartitioner(n_qubits, max_cluster_size=max_size)
    edges = [(i, i + 1) for i in range(n_qubits - 1)]
    result = partitioner.partition(edges)

    for cluster in result.clusters:
        assert len(cluster) <= max_size


def test_symmetry_partition_groups_correctly():
    n_qubits = 8
    partitioner = CircuitPartitioner(n_qubits, max_cluster_size=4)
    symmetries = {
        0: "sigma_g",
        1: "sigma_g",
        2: "sigma_g",
        3: "sigma_g",
        4: "pi_u",
        5: "pi_u",
        6: "pi_u",
        7: "pi_u",
    }
    edges = [(0, 4), (1, 5)]
    result = partitioner.partition(edges, orbital_symmetries=symmetries)

    # All qubits sharing a symmetry label should be in the same cluster
    q2c = result.qubit_to_cluster()
    assert q2c[0] == q2c[1] == q2c[2] == q2c[3]
    assert q2c[4] == q2c[5] == q2c[6] == q2c[7]


def test_qubit_to_cluster_mapping_is_complete():
    n_qubits = 6
    partitioner = CircuitPartitioner(n_qubits, max_cluster_size=3)
    edges = [(0, 3), (1, 4), (2, 5)]
    result = partitioner.partition(edges)

    q2c = result.qubit_to_cluster()
    assert set(q2c.keys()) == set(range(n_qubits))
