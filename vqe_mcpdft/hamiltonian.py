"""Fermionic Hamiltonian construction and qubit mapping.

Builds the second-quantized electronic Hamiltonian from PySCF integrals
and maps it to a qubit representation via Jordan-Wigner transformation.
Ref: VQE-MCPDFT manuscript Eqs. (2)-(5).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


# --- Pauli algebra helpers ---------------------------------------------------

PauliString = Tuple[Tuple[int, str], ...]  # ((qubit, 'X'|'Y'|'Z'|'I'), ...)

# Pauli multiplication table: (phase, result_pauli)
_PAULI_MULT = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"),
    ("X", "X"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Y"): (1, "I"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"),
    ("Z", "X"): (1j, "Y"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "Z"): (1, "I"),
}


def _jordan_wigner_one_body(
    p: int, q: int, n_qubits: int
) -> List[Tuple[complex, PauliString]]:
    """Map a+_p a_q to Pauli strings via Jordan-Wigner (Eq. 5)."""
    if p == q:
        # a+_p a_p = (I - Z_p) / 2
        coeff = 0.5
        term_z: PauliString = tuple(
            (i, "Z") if i == p else (i, "I") for i in range(n_qubits)
        )
        term_i: PauliString = tuple((i, "I") for i in range(n_qubits))
        return [(coeff, term_i), (-coeff, term_z)]

    # p != q: produces X-X and Y-Y chains with Z parity string
    terms: List[Tuple[complex, PauliString]] = []
    lo, hi = min(p, q), max(p, q)
    for pauli_pair, sign in [
        ("XX", 0.25),
        ("YY", 0.25),
        ("XY", -0.25j),
        ("YX", 0.25j),
    ]:
        ops: List[Tuple[int, str]] = []
        for i in range(n_qubits):
            if i == p:
                ops.append((i, pauli_pair[0]))
            elif i == q:
                ops.append((i, pauli_pair[1]))
            elif lo < i < hi:
                ops.append((i, "Z"))
            else:
                ops.append((i, "I"))
        terms.append((sign, tuple(ops)))
    return terms


def _multiply_pauli_strings(
    ps1: PauliString, ps2: PauliString, n_qubits: int
) -> Tuple[complex, PauliString]:
    """Multiply two Pauli strings, returning (phase, result)."""
    phase: complex = 1.0
    result: List[Tuple[int, str]] = []
    for (i1, s1), (i2, s2) in zip(ps1, ps2):
        p, s = _PAULI_MULT[(s1, s2)]
        phase *= p
        result.append((i1, s))
    return (phase, tuple(result))


def _simplify_hamiltonian(
    terms: List[Tuple[complex, PauliString]],
    coeff_threshold: float = 1e-12,
) -> Dict[str, float]:
    """Combine like Pauli strings and drop negligible terms.

    Returns a dict with string keys (e.g. ``"IXYZ"``) and real coefficients.

    Args:
        terms: Raw Pauli terms with complex coefficients.
        coeff_threshold: Drop terms with |coefficient| below this value.
            Default 1e-12 removes only numerically zero terms.  For large
            active spaces (>40 qubits), set to 1e-8 or higher to control
            the total number of Pauli terms and keep hardware measurement
            overhead feasible.

    Raises:
        ValueError: If any Pauli coefficient has a non-negligible imaginary
            part, indicating a bug in the Hamiltonian construction (a
            Hermitian operator must have real Pauli coefficients).
    """
    combined: Dict[str, complex] = {}
    for coeff, ps in terms:
        label = "".join(op for _, op in ps)
        combined[label] = combined.get(label, 0.0) + coeff
    result: Dict[str, float] = {}
    for k, v in combined.items():
        if abs(v) <= coeff_threshold:
            continue
        if abs(v.imag) > 1e-10:
            raise ValueError(
                f"Pauli term '{k}' has non-negligible imaginary coefficient "
                f"{v.imag:.2e}; Hermitian Hamiltonians must have real coefficients."
            )
        result[k] = v.real
    return result


def _expand_integrals_to_spin_orbitals(
    h1e: NDArray[np.float64],
    h2e: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Expand spatial-orbital integrals to spin-orbital basis (vectorized).

    Spin-orbital convention: even indices = alpha, odd = beta.
    h2e is in chemist notation <pq|rs>.
    """
    n_spatial = h1e.shape[0]
    n_so = 2 * n_spatial

    # 1-electron: h1e_so[2p+s, 2q+s] = h1e[p, q] for same spin s
    h1e_so = np.zeros((n_so, n_so))
    for s in range(2):
        h1e_so[s::2, s::2] = h1e

    # 2-electron: h2e_so[2p+s1, 2q+s2, 2r+s1, 2s+s2] = h2e[p,q,r,s]
    # Only nonzero when spin of p matches spin of r, and spin of q matches s.
    h2e_so = np.zeros((n_so, n_so, n_so, n_so))
    for s1 in range(2):
        for s2 in range(2):
            h2e_so[s1::2, s2::2, s1::2, s2::2] = h2e

    return h1e_so, h2e_so


def build_qubit_hamiltonian(
    h1e: NDArray[np.float64],
    h2e: NDArray[np.float64],
    n_qubits: int,
    nuclear_repulsion: float = 0.0,
    coeff_threshold: float = 1e-12,
) -> Dict[str, float]:
    """Build qubit Hamiltonian from spin-orbital 1- and 2-electron integrals.

    Args:
        h1e: One-electron integrals, shape (n_qubits, n_qubits).
        h2e: Two-electron integrals in chemist notation <pq|rs>,
             shape (n_qubits, n_qubits, n_qubits, n_qubits).
        n_qubits: Number of spin-orbitals (= number of qubits).
        nuclear_repulsion: Nuclear repulsion energy constant.
        coeff_threshold: Drop Pauli terms with |coefficient| below this.
            For large active spaces (>40 qubits), use 1e-8 to keep
            the total term count manageable for hardware execution.

    Returns:
        Dictionary mapping Pauli label strings to real coefficients.
    """
    terms: List[Tuple[complex, PauliString]] = []

    # Nuclear repulsion as identity term
    if abs(nuclear_repulsion) > 1e-12:
        identity = tuple((i, "I") for i in range(n_qubits))
        terms.append((nuclear_repulsion, identity))

    # One-body: sum_pq h_pq a+_p a_q
    for p in range(n_qubits):
        for q in range(n_qubits):
            if abs(h1e[p, q]) < 1e-12:
                continue
            for coeff, ps in _jordan_wigner_one_body(p, q, n_qubits):
                terms.append((h1e[p, q] * coeff, ps))

    # Two-body: (1/2) sum_pqrs h_pqrs a+_p a+_q a_s a_r  (chemist notation)
    # Normal-ordered: a+_p a+_q a_s a_r = (a+_p a_r)(a+_q a_s) - delta_qr (a+_p a_s)
    # Use sparsity: skip zero integrals
    nz = np.argwhere(np.abs(h2e) > 1e-12)
    for idx in nz:
        p, q, r, s = idx
        val = 0.5 * h2e[p, q, r, s]
        # Product term: (a+_p a_r)(a+_q a_s)
        for c1, ps1 in _jordan_wigner_one_body(p, r, n_qubits):
            for c2, ps2 in _jordan_wigner_one_body(q, s, n_qubits):
                phase, merged = _multiply_pauli_strings(ps1, ps2, n_qubits)
                terms.append((val * c1 * c2 * phase, merged))
        # Correction: -delta_qr (a+_p a_s)
        if q == r:
            for c, ps in _jordan_wigner_one_body(p, s, n_qubits):
                terms.append((-val * c, ps))

    return _simplify_hamiltonian(terms, coeff_threshold)


def hamiltonian_from_pyscf(
    mol_pyscf: object,
    active_space: Tuple[int, int] | None = None,
) -> Tuple[Dict[str, float], int]:
    """Build qubit Hamiltonian directly from a PySCF molecule object.

    Args:
        mol_pyscf: PySCF Mole object (already built).
        active_space: (n_electrons, n_orbitals) for CAS. If None, use full space.

    Returns:
        (qubit_hamiltonian, n_qubits)
    """
    from pyscf import scf, mcscf

    mf = scf.RHF(mol_pyscf).run()

    if active_space is not None:
        n_elec, n_orb = active_space
        mc = mcscf.CASCI(mf, n_orb, n_elec)
        h1e, e_core = mc.get_h1eff()
        h2e = mc.get_h2eff()
        n_qubits = 2 * n_orb
    else:
        n_qubits = 2 * mol_pyscf.nao  # type: ignore[attr-defined]
        h1e = mf.get_hcore()
        h2e = mol_pyscf.intor("int2e")  # type: ignore[attr-defined]
        e_core = mol_pyscf.energy_nuc()  # type: ignore[attr-defined]

    # Expand spatial integrals to spin-orbital basis (vectorized)
    h1e_so, h2e_so = _expand_integrals_to_spin_orbitals(h1e, h2e)

    return build_qubit_hamiltonian(h1e_so, h2e_so, n_qubits, e_core), n_qubits
