"""CASCI-inspired particle-number-conserving quantum ansatz.

Implements the parameterized circuit U_ansatz(theta) that prepares a
multireference trial state within the active space (Eq. 8 of manuscript).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import tensorcircuit as tc


def _hf_reference_circuit(circuit: tc.Circuit, n_electrons: int) -> tc.Circuit:
    """Prepare Hartree-Fock reference |1100...0> on the circuit."""
    for i in range(n_electrons):
        circuit.x(i)
    return circuit


def _particle_conserving_gate(
    circuit: tc.Circuit, q0: int, q1: int, theta: float
) -> tc.Circuit:
    """Apply a particle-number-conserving two-qubit gate.

    Implements the Givens rotation that preserves total particle number:
    |01> <-> |10> with angle theta, leaving |00> and |11> invariant.
    """
    circuit.cnot(q0, q1)
    circuit.ry(q0, theta=theta)
    circuit.cnot(q1, q0)
    circuit.ry(q0, theta=-theta)
    circuit.cnot(q0, q1)
    return circuit


class CASCIAnsatz:
    """CASCI-inspired variational ansatz for VQE-MC-PDFT.

    Constructs a particle-number-conserving circuit that explores the
    complete active space configuration interaction manifold (Eq. 6-8).

    Args:
        n_qubits: Number of qubits (= 2 * n_active_orbitals).
        n_electrons: Number of active electrons.
        n_layers: Number of variational layers.
    """

    def __init__(self, n_qubits: int, n_electrons: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.n_layers = n_layers
        self.n_params = self._count_params()

    def _count_params(self) -> int:
        """Count total variational parameters.

        Each layer applies particle-conserving gates to non-overlapping
        qubit pairs.  Even layers pair (0,1),(2,3),... and odd layers
        pair (1,2),(3,4),...  Each gate consumes exactly one parameter.
        """
        total = 0
        for layer in range(self.n_layers):
            offset = layer % 2
            for i in range(offset, self.n_qubits - 1, 2):
                total += 1
        return total

    def build_circuit(
        self, params: np.ndarray, orbital_rotation_angles: np.ndarray | None = None
    ) -> tc.Circuit:
        """Build the full VQE-MC-PDFT circuit.

        Args:
            params: Variational parameters theta for the CASCI ansatz.
            orbital_rotation_angles: Fixed angles lambda for U_orb (Eq. 27).
                If None, no orbital rotation is appended.

        Returns:
            TensorCircuit Circuit object ready for execution.
        """
        circuit = tc.Circuit(self.n_qubits)

        # Step 1: Hartree-Fock reference (blue block in Fig. S2)
        _hf_reference_circuit(circuit, self.n_electrons)

        # Step 2: Particle-conserving ansatz layers (orange block in Fig. S2)
        idx = 0
        for layer in range(self.n_layers):
            offset = layer % 2
            for i in range(offset, self.n_qubits - 1, 2):
                _particle_conserving_gate(circuit, i, i + 1, params[idx])
                idx += 1

        # Step 3: Orbital rotation unitary (green/yellow/brown blocks)
        if orbital_rotation_angles is not None:
            self._append_orbital_rotation(circuit, orbital_rotation_angles)

        return circuit

    def _append_orbital_rotation(self, circuit: tc.Circuit, angles: np.ndarray) -> None:
        """Append Trotterized orbital rotation U_orb(lambda) to circuit.

        Uses OrbitalRotationCircuit.apply_to_circuit for consistent
        Pauli-term-based Trotter decomposition (Eqs. 24-27).
        The angle vector must be fixed-length from update(include_zeros=True).
        """
        from vqe_mcpdft.orbital_rotation import OrbitalRotationCircuit
        orb = OrbitalRotationCircuit(self.n_qubits // 2)
        orb.apply_to_circuit(circuit, angles)

    def initial_params(self, seed: int = 42) -> np.ndarray:
        """Generate initial variational parameters near zero."""
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.01, size=self.n_params)

    def get_qubit_pairs(self) -> List[Tuple[int, int]]:
        """Return list of entangling gate qubit pairs for partitioning."""
        pairs: List[Tuple[int, int]] = []
        for layer in range(self.n_layers):
            offset = layer % 2
            for i in range(offset, self.n_qubits - 1, 2):
                pairs.append((i, i + 1))
        return pairs
