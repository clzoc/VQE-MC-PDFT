"""Orbital rotation circuit compilation.

Compiles the classical orbital-rotation generator kappa into a quantum
circuit U_orb via Suzuki-Trotter decomposition (Eqs. 16-27 of manuscript).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def compute_orbital_rotation_generator(
    orbital_gradient: NDArray[np.float64],
    step_size: float = 0.01,
) -> NDArray[np.float64]:
    """Construct antisymmetric orbital-rotation matrix kappa from gradient.

    kappa_pq = -alpha * g_pq  (Eq. 22)

    Args:
        orbital_gradient: MC-PDFT orbital gradient g_pq.
        step_size: Learning rate alpha > 0.

    Returns:
        Antisymmetric matrix kappa.
    """
    n = orbital_gradient.shape[0]
    kappa = np.zeros((n, n))
    for p in range(n):
        for q in range(p + 1, n):
            kappa[p, q] = -step_size * orbital_gradient[p, q]
            kappa[q, p] = step_size * orbital_gradient[p, q]
    return kappa


def kappa_to_qubit_hamiltonian(
    kappa: NDArray[np.float64],
    include_zeros: bool = False,
) -> List[Tuple[float, List[Tuple[int, str]]]]:
    """Map orbital-rotation generator to qubit Pauli terms.

    Constructs H_orb = sum_pq K_pq E_pq (Eq. 24) and maps via
    Jordan-Wigner to Pauli strings for Trotterization.

    Args:
        kappa: Antisymmetric orbital-rotation matrix (spatial orbitals).
        include_zeros: If True, include terms with zero coefficient (for fixed term ordering).

    Returns:
        List of (coefficient, pauli_ops) for Trotter steps.
    """
    n_spatial = kappa.shape[0]
    pauli_terms: List[Tuple[float, List[Tuple[int, str]]]] = []

    for p in range(n_spatial):
        for q in range(p + 1, n_spatial):
            angle = kappa[p, q]
            if not include_zeros and abs(angle) < 1e-12:
                continue
            # E_pq - E_qp maps to (XX + YY) terms for each spin
            for spin_offset in (0, 1):
                i = 2 * p + spin_offset
                j = 2 * q + spin_offset
                # XX term
                ops_xx: List[Tuple[int, str]] = [(i, "X"), (j, "X")]
                for k in range(i + 1, j):
                    ops_xx.append((k, "Z"))
                pauli_terms.append((0.5 * angle, ops_xx))
                # YY term
                ops_yy: List[Tuple[int, str]] = [(i, "Y"), (j, "Y")]
                for k in range(i + 1, j):
                    ops_yy.append((k, "Z"))
                pauli_terms.append((0.5 * angle, ops_yy))

    return pauli_terms


def trotterize_orbital_rotation(
    pauli_terms: List[Tuple[float, List[Tuple[int, str]]]],
    dt: float = 1.0,
) -> NDArray[np.float64]:
    """Compute Trotter angles for orbital rotation circuit.

    Each Pauli term exp(-i k_j P_j dt) yields a rotation angle (Eq. 26).

    Args:
        pauli_terms: Output of kappa_to_qubit_hamiltonian.
        dt: Trotter time step.

    Returns:
        Array of rotation angles for the compiled circuit.
    """
    return np.array([coeff * dt for coeff, _ in pauli_terms])


class OrbitalRotationCircuit:
    """Manages the compilation of orbital rotations into quantum circuits.

    Implements the pipeline: kappa -> H_orb -> Trotter -> circuit angles
    (Eqs. 24-27 of manuscript).

    Args:
        n_spatial_orbitals: Number of spatial orbitals in the active space.
    """

    def __init__(self, n_spatial_orbitals: int):
        self.n_spatial = n_spatial_orbitals
        self.n_qubits = 2 * n_spatial_orbitals
        self._current_angles: NDArray[np.float64] | None = None
        self._current_kappa: NDArray[np.float64] | None = None

    def update(
        self,
        orbital_gradient: NDArray[np.float64],
        step_size: float = 0.01,
    ) -> NDArray[np.float64]:
        """Compute new orbital rotation angles from the MC-PDFT gradient.

        Returns a fixed-length angle vector aligned with _get_pauli_terms()
        (include_zeros=True), so every term has a corresponding angle slot.

        Args:
            orbital_gradient: g_pq from MC-PDFT Fock matrix (Eq. 21).
            step_size: Steepest-descent step size alpha.

        Returns:
            Rotation angles lambda for the circuit (fixed-length).
        """
        self._current_kappa = compute_orbital_rotation_generator(
            orbital_gradient, step_size
        )
        # Use include_zeros=True for fixed-length, stable ordering
        pauli_terms = kappa_to_qubit_hamiltonian(self._current_kappa, include_zeros=True)
        self._current_angles = trotterize_orbital_rotation(pauli_terms)
        return self._current_angles

    @property
    def angles(self) -> NDArray[np.float64] | None:
        return self._current_angles

    @property
    def kappa(self) -> NDArray[np.float64] | None:
        return self._current_kappa

    def _get_pauli_terms(self) -> List[Tuple[float, List[Tuple[int, str]]]]:
        """Return fixed Pauli term structure for this orbital rotation circuit."""
        dummy_kappa = np.zeros((self.n_spatial, self.n_spatial))
        return kappa_to_qubit_hamiltonian(dummy_kappa, include_zeros=True)

    def gradient_norm(self) -> float:
        """Return the Frobenius norm of the current orbital gradient."""
        if self._current_kappa is None:
            return float("inf")
        return float(np.linalg.norm(self._current_kappa))
    
    def apply_to_circuit(self, circuit, angles: np.ndarray) -> None:
        """Apply orbital rotation circuit to a given quantum circuit.
        
        Args:
            circuit: Target quantum circuit to modify
            angles: Rotation angles from update() or trotterize_orbital_rotation()
        """
        # Generate fixed Pauli term structure for current n_spatial orbitals (all terms included)
        # Term order is identical to what update() produces, matching angle ordering
        dummy_kappa = np.zeros((self.n_spatial, self.n_spatial))
        pauli_terms = kappa_to_qubit_hamiltonian(dummy_kappa, include_zeros=True)
        
        # Validate angle count matches expected terms
        assert len(angles) == len(pauli_terms), \
            f"Angle count mismatch: expected {len(pauli_terms)} terms, got {len(angles)} angles"
        
        # Apply each Pauli rotation term
        for (coeff, ops), angle in zip(pauli_terms, angles):
            if abs(angle) < 1e-12:
                continue
            
            # Get qubit indices and Pauli types
            qubits = [q for q, _ in ops]
            pauli_types = [p for _, p in ops]
            
            # Apply basis rotations for X/Y terms
            for q, p in zip(qubits, pauli_types):
                if p == "X":
                    circuit.h(q)
                elif p == "Y":
                    circuit.sdg(q)
                    circuit.h(q)
            
            # Apply CNOT ladder
            for i in range(len(qubits) - 1):
                circuit.cx(qubits[i], qubits[i + 1])
            
            # Apply Rz rotation
            circuit.rz(qubits[-1], angle)
            
            # Apply inverse CNOT ladder
            for i in range(len(qubits) - 2, -1, -1):
                circuit.cx(qubits[i], qubits[i + 1])
            
            # Apply inverse basis rotations
            for q, p in reversed(list(zip(qubits, pauli_types))):
                if p == "X":
                    circuit.h(q)
                elif p == "Y":
                    circuit.h(q)
                    circuit.s(q)
