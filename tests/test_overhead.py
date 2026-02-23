"""Tests for the overhead budget calculator and cut strategy selector."""

from __future__ import annotations

from quantum_circuit_cutting.overhead import (
    GAMMA_WIRE_CUT,
    compute_overhead_budget,
    estimate_sampling_cost,
    select_cut_strategy,
)


def test_overhead_budget_basic():
    budget = compute_overhead_budget(n_qubits=16, max_cluster_size=13)
    assert budget.max_cuts >= 1
    assert budget.gamma_per_cut == GAMMA_WIRE_CUT  # Only wire cuts supported currently
    assert budget.kappa_max >= budget.gamma_per_cut


def test_overhead_budget_no_cutting_needed():
    budget = compute_overhead_budget(n_qubits=12, max_cluster_size=13)
    assert budget.max_cuts == 0
    assert budget.kappa_actual == 1.0


def test_overhead_budget_large_system():
    budget = compute_overhead_budget(n_qubits=84, max_cluster_size=13)
    assert budget.max_cuts >= 6  # ceil(84/13)-1 = 6 minimum
    assert budget.kappa_max > 1.0


def test_overhead_budget_wire_cuts():
    budget = compute_overhead_budget(
        n_qubits=16, max_cluster_size=13
    )
    assert budget.gamma_per_cut == GAMMA_WIRE_CUT


def test_overhead_budget_custom_kappa():
    budget = compute_overhead_budget(
        n_qubits=16, max_cluster_size=13, kappa_max=100.0
    )
    assert budget.kappa_max == 100.0


def test_estimate_sampling_cost():
    cost = estimate_sampling_cost(n_cuts=2, n_pauli_terms=100)
    assert cost["kappa"] == 4.0 ** 2
    assert cost["n_configurations_exact"] == 8 ** 2
    assert cost["variance_factor"] == (4.0 ** 2) ** 2


def test_select_strategy_small():
    """Small problem: exact enumeration."""
    strategy = select_cut_strategy(
        n_qubits=16, max_cluster_size=13,
        entangling_gates=[(i, i + 1) for i in range(15)],
    )
    assert strategy["use_sampling"] is False


def test_select_strategy_large():
    """Large problem: sampling required."""
    strategy = select_cut_strategy(
        n_qubits=84, max_cluster_size=13,
        entangling_gates=[(i, i + 1) for i in range(83)],
    )
    assert strategy["use_sampling"] is True
    assert strategy["n_samples"] >= 200  # Minimum floor from select_cut_strategy
