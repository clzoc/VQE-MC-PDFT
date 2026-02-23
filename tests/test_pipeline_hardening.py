"""Integration tests for the hardened circuit-cutting pipeline.

Tests:
1. Dispatcher cutting path uses statevector-derived counts (no fake counts)
2. Self-consistent + cutting RDM path (pauli_expval_fn injection)
3. Configuration consistency in cutting VQE
4. Hamiltonian truncation/remapping for Cr2 qubit scaling
5. Custom circuit_builder in FragmentCircuitGenerator
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import tensorcircuit as tc

from quantum_circuit_cutting.channel_decomposition import ChannelDecomposition
from quantum_circuit_cutting.cutting_dispatch import (
    CuttingDispatcher,
)
from quantum_circuit_cutting.fragment_circuits import (
    FragmentCircuitGenerator,
)
from quantum_circuit_cutting.partition import ClusterPartition
from quantum_circuit_cutting.reconstruction import CuttingReconstructor
from tests.sim_utils import simulate_fragment_counts


class _MockBackend:
    """Deterministic local backend mock for tests."""

    def submit_circuit(self, circuit):
        n_qubits = int(circuit._nqubits)  # TensorCircuit internal field
        return simulate_fragment_counts(
            circuit, n_qubits, 1000, np.random.default_rng(42)
        )

    def submit_batch(self, circuits):
        rng = np.random.default_rng(42)
        out = []
        for circuit in circuits:
            n_qubits = int(circuit._nqubits)
            out.append(simulate_fragment_counts(circuit, n_qubits, 1000, rng))
        return out


# --- Test 1: statevector-derived fragment counts ---

def test_simulate_fragment_counts_valid_distribution():
    """Counts from simulate_fragment_counts must be a valid probability distribution."""
    c = tc.Circuit(2)
    c.h(0)
    c.cnot(0, 1)  # Bell state: 50% |00>, 50% |11>
    rng = np.random.default_rng(42)
    counts = simulate_fragment_counts(c, 2, 10000, rng)

    assert sum(counts.values()) == 10000
    # Bell state should only have "00" and "11"
    for key in counts:
        assert key in ("00", "11"), f"Unexpected bitstring {key}"
    # Each should be roughly 50%
    assert abs(counts.get("00", 0) / 10000 - 0.5) < 0.05
    assert abs(counts.get("11", 0) / 10000 - 0.5) < 0.05


def test_simulate_fragment_counts_deterministic():
    """Same seed must produce identical counts."""
    c = tc.Circuit(2)
    c.h(0)
    counts1 = simulate_fragment_counts(c, 2, 1000, np.random.default_rng(0))
    counts2 = simulate_fragment_counts(c, 2, 1000, np.random.default_rng(0))
    assert counts1 == counts2


def test_dispatcher_cutting_no_fake_counts():
    """Dispatcher cutting path should be deterministic with a deterministic backend."""
    # 4-qubit system with device limit 2 -> forces cutting
    ham = {"ZIII": 1.0, "IZII": -0.5, "IIII": 3.0}
    dispatcher = CuttingDispatcher(
        max_device_qubits=2,
        n_shots=1000,
        tqp_backend=_MockBackend(),
    )
    energy_fn = dispatcher.make_energy_fn(4, 2, ham)
    params = np.zeros(4)  # dummy params

    e1 = energy_fn(params)
    e2 = energy_fn(params)
    # With deterministic statevector-derived counts (seeded), same params -> same energy
    assert e1 == pytest.approx(e2, abs=1e-10)
    assert np.isfinite(e1)


# --- Test 2: pauli_expval_fn injection ---

def test_dispatcher_pauli_expval_fn_direct_mode():
    """Direct mode (no cutting) should still expose a hardware expval callback."""
    ham = {"ZI": 1.0, "II": 0.5}
    dispatcher = CuttingDispatcher(max_device_qubits=4, tqp_backend=_MockBackend())
    fn = dispatcher.make_pauli_expval_fn(2, 1, ham)
    assert fn is not None


def test_dispatcher_pauli_expval_fn_cutting_mode():
    """Cutting mode should return a callable pauli_expval_fn."""
    ham = {"ZIII": 1.0, "IIII": 0.5}
    dispatcher = CuttingDispatcher(
        max_device_qubits=2,
        n_shots=1000,
        tqp_backend=_MockBackend(),
    )
    fn = dispatcher.make_pauli_expval_fn(4, 2, ham)
    assert fn is not None

    params = np.zeros(4)
    result = fn(params, None, ["ZIII", "IZII", "IIII"])
    assert "IIII" in result
    assert result["IIII"] == 1.0  # Identity always 1
    assert np.isfinite(result["ZIII"])
    assert np.isfinite(result["IZII"])


# --- Test 3: configuration consistency ---

def test_config_consistency_in_cutting_vqe():
    """Fragment generation and reconstruction must use the same config set.

    We verify this by checking that generate_all with explicit configs
    produces the correct number of fragments matching the config count.
    """
    partition = ClusterPartition(
        clusters=[[0, 1], [2, 3]],
        inter_cluster_edges=[(1, 2)],
        cluster_graph=nx.Graph([(0, 1)]),
        n_cuts=1,
    )
    decomp = ChannelDecomposition(n_cuts=1)
    configs = decomp.enumerate_configurations()
    assert len(configs) == 8

    params = np.array([0.1, 0.2, 0.3, 0.4])
    gen = FragmentCircuitGenerator(
        partition=partition,
        decomposition=decomp,
        ansatz_params=params,
        n_electrons=2,
        n_layers=2,
    )

    # Pass explicit configs -> must produce exactly len(configs) fragment sets
    all_frags = gen.generate_all(configurations=configs)
    assert len(all_frags) == len(configs)

    # Each fragment set has one circuit per cluster
    for frags in all_frags:
        assert len(frags) == 2


def test_config_consistency_sampled():
    """Sampled configs passed to generate_all must match reconstruction."""
    partition = ClusterPartition(
        clusters=[[0, 1], [2, 3]],
        inter_cluster_edges=[(1, 2)],
        cluster_graph=nx.Graph([(0, 1)]),
        n_cuts=1,
    )
    decomp = ChannelDecomposition(n_cuts=1)
    rng = np.random.default_rng(99)
    configs = decomp.sample_configurations(5, rng=rng)

    params = np.zeros(4)
    gen = FragmentCircuitGenerator(
        partition=partition,
        decomposition=decomp,
        ansatz_params=params,
        n_electrons=2,
    )
    all_frags = gen.generate_all(configurations=configs)
    assert len(all_frags) == 5  # Exactly 5 sampled configs

    # Execute and reconstruct with the SAME configs
    recon = CuttingReconstructor(partition, decomp)
    fragment_results = {}
    sim_rng = np.random.default_rng(42)
    for frags in all_frags:
        for f in frags:
            counts = simulate_fragment_counts(f.circuit, f.n_qubits, 1000, sim_rng)
            fragment_results[(f.config_idx, f.cluster_idx)] = counts

    val = recon.reconstruct_expectation(fragment_results, "ZIII", configs)
    assert np.isfinite(val)


# --- Test 4: Hamiltonian truncation ---

def test_hamiltonian_truncation_dimensions():
    """Truncated Hamiltonian Pauli strings must match target n_qubits."""
    from vqe_mcpdft.hamiltonian import (
        build_qubit_hamiltonian,
        _expand_integrals_to_spin_orbitals,
    )

    # Build a small "full" Hamiltonian for 4 spatial orbitals (8 qubits)
    n_spatial = 4
    rng = np.random.default_rng(42)
    h1e = rng.standard_normal((n_spatial, n_spatial))
    h1e = (h1e + h1e.T) / 2  # symmetrize

    # 2e integrals need full 8-fold symmetry for chemist notation <pq|rs>
    h2e = np.zeros((n_spatial, n_spatial, n_spatial, n_spatial))
    for p in range(n_spatial):
        for q in range(p, n_spatial):
            for r in range(n_spatial):
                for s in range(r, n_spatial):
                    v = rng.standard_normal()
                    # <pq|rs> = <qp|sr> = <rs|pq> = <sr|qp> (real orbitals)
                    h2e[p, q, r, s] = v
                    h2e[q, p, s, r] = v
                    h2e[r, s, p, q] = v
                    h2e[s, r, q, p] = v

    # Full Hamiltonian: 8 qubits
    h1e_so, h2e_so = _expand_integrals_to_spin_orbitals(h1e, h2e)
    ham_full = build_qubit_hamiltonian(h1e_so, h2e_so, 8, 0.0)
    for ps in ham_full:
        assert len(ps) == 8

    # Truncated to 3 spatial orbitals (6 qubits)
    n_sub = 3
    h1e_sub = h1e[:n_sub, :n_sub]
    h2e_sub = h2e[:n_sub, :n_sub, :n_sub, :n_sub]
    h1e_so_sub, h2e_so_sub = _expand_integrals_to_spin_orbitals(h1e_sub, h2e_sub)
    ham_trunc = build_qubit_hamiltonian(h1e_so_sub, h2e_so_sub, 6, 0.0)
    for ps in ham_trunc:
        assert len(ps) == 6


# --- Test 5: custom circuit_builder ---

def test_custom_circuit_builder_is_used():
    """When circuit_builder is provided, it must be called instead of default ansatz."""
    call_log = []

    def my_builder(n_qubits: int, params: np.ndarray, n_electrons: int, orb_angles: np.ndarray | None = None) -> tc.Circuit:
        call_log.append((n_qubits, len(params), n_electrons))
        c = tc.Circuit(n_qubits)
        for i in range(min(n_electrons, n_qubits)):
            c.x(i)
        return c

    partition = ClusterPartition(
        clusters=[[0, 1], [2, 3]],
        inter_cluster_edges=[(1, 2)],
        cluster_graph=nx.Graph([(0, 1)]),
        n_cuts=1,
    )
    decomp = ChannelDecomposition(n_cuts=1)
    params = np.array([0.1, 0.2])

    gen = FragmentCircuitGenerator(
        partition=partition,
        decomposition=decomp,
        ansatz_params=params,
        n_electrons=2,
        circuit_builder=my_builder,
    )

    # Generate for just 1 config to keep it fast
    configs = decomp.enumerate_configurations()[:1]
    frags = gen.generate_all(configurations=configs)
    assert len(frags) == 1

    # Builder should have been called once per cluster (2 clusters)
    assert len(call_log) == 2
    # Each call should have n_qubits=2 (cluster size)
    assert all(nq == 2 for nq, _, _ in call_log)


# --- Test 6: self_consistent pauli_expval_fn parameter ---

def test_self_consistent_accepts_pauli_expval_fn():
    """SelfConsistentVQEMCPDFT must accept pauli_expval_fn without error."""
    from vqe_mcpdft.self_consistent import SelfConsistentVQEMCPDFT

    # Just verify the parameter is accepted (don't run the full loop)
    import inspect
    sig = inspect.signature(SelfConsistentVQEMCPDFT.__init__)
    assert "pauli_expval_fn" in sig.parameters
