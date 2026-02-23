"""Flow contract tests for large qubit counts and hardware-only routing."""
import numpy as np
from unittest.mock import patch
from quantum_circuit_cutting.cutting_dispatch import CuttingDispatcher
from tests.sim_utils import simulate_fragment_counts


class _MockBackend:
    """Deterministic local backend mock for static flow tests."""

    def submit_circuit(self, circuit):
        n_qubits = int(circuit._nqubits)
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


def test_large_system_uses_cutting_path():
    """n_qubits > 13 must route through cutting, never direct statevector."""
    n_qubits = 16
    n_electrons = 8
    ham = {"I" * n_qubits: 0.0, "Z" + "I" * (n_qubits - 1): 1.0}

    dispatcher = CuttingDispatcher(max_device_qubits=13, tqp_backend=_MockBackend())
    assert dispatcher.needs_cutting(n_qubits)

    # The energy function factory should use cutting path
    energy_fn_factory = dispatcher.make_energy_fn_factory(n_qubits, n_electrons, ham)
    energy_fn = energy_fn_factory(None)
    params = np.zeros(16)

    # Patch the direct statevector evaluator to ensure it's NOT called
    with patch("vqe_mcpdft.vqe_solver.evaluate_hamiltonian_statevector") as mock_sv:
        mock_sv.side_effect = AssertionError(
            "Full-system statevector must not be called for n_qubits > 13"
        )
        # This should succeed via cutting path using backend execution.
        energy = energy_fn(params)
        assert np.isfinite(energy)
        mock_sv.assert_not_called()


def test_small_systems_use_direct_hardware_path():
    """n_qubits <= 13 should use direct hardware path (no statevector fallback)."""
    n_qubits = 12
    n_electrons = 6
    ham = {"I" * n_qubits: 0.0, "Z" + "I" * (n_qubits - 1): 1.0}

    dispatcher = CuttingDispatcher(max_device_qubits=13, tqp_backend=_MockBackend())
    assert not dispatcher.needs_cutting(n_qubits)

    with patch("vqe_mcpdft.vqe_solver.evaluate_hamiltonian_statevector") as mock_eval:
        energy_fn_factory = dispatcher.make_energy_fn_factory(n_qubits, n_electrons, ham)
        energy_fn = energy_fn_factory(None)
        params = np.zeros(12)
        energy = energy_fn(params)
        assert np.isfinite(energy)
        mock_eval.assert_not_called()


def test_pauli_expval_fn_callable_for_small():
    """Direct mode should return a callable pauli_expval_fn."""
    ham = {"ZI": 1.0, "II": 0.5}
    dispatcher = CuttingDispatcher(max_device_qubits=4, tqp_backend=_MockBackend())
    fn = dispatcher.make_pauli_expval_fn(2, 1, ham)
    assert fn is not None
    result = fn(np.zeros(2), None, ["ZI", "II"])
    assert result["II"] == 1.0
    assert np.isfinite(result["ZI"])


def test_pauli_expval_fn_callable_for_large():
    """Cutting mode returns a callable pauli_expval_fn."""
    n_qubits = 4
    ham = {"ZIII": 1.0, "IIII": 0.5}
    dispatcher = CuttingDispatcher(
        max_device_qubits=2,
        n_shots=1000,
        tqp_backend=_MockBackend(),
    )
    fn = dispatcher.make_pauli_expval_fn(n_qubits, 2, ham)
    assert fn is not None
    result = fn(np.zeros(4), None, ["ZIII", "IIII"])
    assert result["IIII"] == 1.0
    assert np.isfinite(result["ZIII"])
