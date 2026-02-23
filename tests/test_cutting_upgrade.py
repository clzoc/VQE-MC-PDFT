"""Tests for spectral bisection, KL refinement, and QWC grouping."""

from __future__ import annotations

import networkx as nx

from quantum_circuit_cutting.partition import (
    CircuitPartitioner,
    ClusterPartition,
    _kl_refine,
    _build_interaction_graph,
)
from quantum_circuit_cutting.reconstruction import group_commuting_paulis, _qwc_compatible


# --- Spectral bisection tests ---

def test_spectral_partition_16_qubits():
    """C2 CAS(8e,8o) -> 16 qubits, max 13 -> 2 clusters."""
    partitioner = CircuitPartitioner(16, max_cluster_size=13)
    edges = [(i, i + 1) for i in range(15)]
    result = partitioner.partition(edges)

    assert result.n_clusters == 2
    assert result.max_cluster_size <= 13
    assert sum(len(c) for c in result.clusters) == 16
    assert result.n_cuts >= 1


def test_spectral_partition_24_qubits():
    """Cr2 CAS(12e,12o) -> 24 qubits, max 13 -> 2 clusters."""
    partitioner = CircuitPartitioner(24, max_cluster_size=13)
    edges = [(i, i + 1) for i in range(23)]
    result = partitioner.partition(edges)

    assert result.n_clusters >= 2
    assert result.max_cluster_size <= 13
    assert sum(len(c) for c in result.clusters) == 24


def test_spectral_partition_44_qubits():
    """Cr2 CAS(12e,22o) -> 44 qubits, max 13 -> 4+ clusters."""
    partitioner = CircuitPartitioner(44, max_cluster_size=13)
    edges = [(i, i + 1) for i in range(43)]
    result = partitioner.partition(edges)

    assert result.n_clusters >= 4
    assert result.max_cluster_size <= 13
    assert sum(len(c) for c in result.clusters) == 44


def test_spectral_partition_84_qubits():
    """Cr2 CAS(48e,42o) -> 84 qubits, max 13 -> 7+ clusters."""
    partitioner = CircuitPartitioner(84, max_cluster_size=13)
    edges = [(i, i + 1) for i in range(83)]
    result = partitioner.partition(edges)

    assert result.n_clusters >= 7
    assert result.max_cluster_size <= 13
    assert sum(len(c) for c in result.clusters) == 84


def test_no_cutting_needed():
    """Benzene CAS(6e,6o) -> 12 qubits, max 13 -> 1 cluster, 0 cuts."""
    partitioner = CircuitPartitioner(12, max_cluster_size=13)
    edges = [(i, i + 1) for i in range(11)]
    result = partitioner.partition(edges)

    assert result.n_clusters == 1
    assert result.n_cuts == 0
    assert result.max_cluster_size == 12


# --- KL refinement tests ---

def test_kl_reduces_cuts():
    """KL refinement should not increase the cut count."""
    graph = _build_interaction_graph(10, [(i, i + 1) for i in range(9)])
    # Start with a bad partition: [0,1,5,6] and [2,3,4,7,8,9]
    initial = [[0, 1, 5, 6], [2, 3, 4, 7, 8, 9]]
    refined = _kl_refine(initial, graph, max_size=6)

    # Count cuts before and after
    def count_cuts(clusters):
        q2c = {}
        for ci, qs in enumerate(clusters):
            for q in qs:
                q2c[q] = ci
        return sum(1 for u, v in graph.edges() if q2c.get(u) != q2c.get(v))

    assert count_cuts(refined) <= count_cuts(initial)


def test_kl_respects_max_size():
    """KL refinement must not exceed max cluster size."""
    graph = _build_interaction_graph(8, [(i, i + 1) for i in range(7)])
    initial = [[0, 1, 2, 3], [4, 5, 6, 7]]
    refined = _kl_refine(initial, graph, max_size=4)
    for cluster in refined:
        assert len(cluster) <= 4


# --- QWC grouping tests ---

def test_qwc_compatible_basic():
    assert _qwc_compatible("IXYZ", "IXYZ") is True
    assert _qwc_compatible("IXYZ", "IIII") is True
    assert _qwc_compatible("XIII", "YIII") is False
    assert _qwc_compatible("XIZI", "XIZI") is True
    assert _qwc_compatible("XIZI", "YIZI") is False


def test_group_commuting_paulis_trivial():
    groups = group_commuting_paulis(["ZIII", "IZII", "IIZI"])
    # All Z-only strings are QWC with each other
    assert len(groups) == 1


def test_group_commuting_paulis_incompatible():
    groups = group_commuting_paulis(["XIII", "YIII", "ZIII"])
    # X, Y, Z on same qubit -> 3 separate groups
    assert len(groups) == 3


def test_group_commuting_paulis_mixed():
    strings = ["ZIII", "IZII", "XIII", "IXII"]
    groups = group_commuting_paulis(strings)
    # ZIII and IZII are compatible; XIII and IXII are compatible
    # But ZIII and XIII are not (Z vs X on qubit 0)
    assert len(groups) == 2


# --- Overhead property test ---

def test_overhead_kappa_property():
    partition = ClusterPartition(
        clusters=[[0, 1], [2, 3]],
        inter_cluster_edges=[(1, 2)],
        cluster_graph=nx.Graph([(0, 1)]),
        n_cuts=1,
    )
    assert partition.overhead_kappa == 4.0 ** 1

    partition2 = ClusterPartition(
        clusters=[[0, 1], [2, 3], [4, 5]],
        inter_cluster_edges=[(1, 2), (3, 4)],
        cluster_graph=nx.Graph([(0, 1), (1, 2)]),
        n_cuts=2,
    )
    assert partition2.overhead_kappa == 4.0 ** 2
