"""RDM measurement schedules for quantum hardware.

Extracts active-space 1-RDM and 2-RDM from the optimized VQE state
by measuring Pauli-string expectation values (Eqs. 10-11).

The 2-RDM is constructed from *direct* Pauli measurements of the
two-body operator a+_p a+_q a_s a_r under Jordan-Wigner, not from
a cumulant approximation.  This is essential for multireference states
where the cumulant is non-negligible.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


def _rdm1_pauli_terms(
    n_qubits: int,
) -> Dict[Tuple[int, int], List[Tuple[complex, str]]]:
    """Generate Pauli measurement terms for 1-RDM elements gamma_pq.

    Under Jordan-Wigner, a+_p a_q maps to Pauli strings.  For p == q
    this is (I - Z_p)/2.  For p != q, the full decomposition includes
    XX, YY (real part) and XY, YX (imaginary part) terms with the
    Z parity chain between p and q.

    Returns:
        Dict mapping (p, q) to list of (coefficient, pauli_label) pairs.
    """
    terms: Dict[Tuple[int, int], List[Tuple[complex, str]]] = {}
    for p in range(n_qubits):
        for q in range(p, n_qubits):
            if p == q:
                # a+_p a_p = (I - Z_p) / 2
                label_i = "I" * n_qubits
                label_z = "I" * p + "Z" + "I" * (n_qubits - p - 1)
                terms[(p, q)] = [(0.5, label_i), (-0.5, label_z)]
            else:
                # Full JW decomposition of a+_p a_q (p < q):
                # a+_p a_q = 0.25*(XX + YY) + 0.25j*(YX - XY) with Z chain
                z_chain = "Z" * (q - p - 1)
                label_xx = (
                    "I" * p + "X" + z_chain + "X" + "I" * (n_qubits - q - 1)
                )
                label_yy = (
                    "I" * p + "Y" + z_chain + "Y" + "I" * (n_qubits - q - 1)
                )
                label_xy = (
                    "I" * p + "X" + z_chain + "Y" + "I" * (n_qubits - q - 1)
                )
                label_yx = (
                    "I" * p + "Y" + z_chain + "X" + "I" * (n_qubits - q - 1)
                )
                terms[(p, q)] = [
                    (0.25, label_xx),
                    (0.25, label_yy),
                    (-0.25j, label_xy),
                    (0.25j, label_yx),
                ]
    return terms


def _jw_one_body_pauli(
    p: int, q: int, n: int
) -> List[Tuple[complex, str]]:
    """Return Pauli decomposition of a+_p a_q under JW as label strings."""
    if p == q:
        label_i = "I" * n
        label_z = "I" * p + "Z" + "I" * (n - p - 1)
        return [(0.5, label_i), (-0.5, label_z)]
    lo, hi = min(p, q), max(p, q)
    result: List[Tuple[complex, str]] = []
    for pauli_pair, coeff in [("XX", 0.25), ("YY", 0.25), ("XY", -0.25j), ("YX", 0.25j)]:
        chars = ["I"] * n
        chars[p] = pauli_pair[0]
        chars[q] = pauli_pair[1]
        for k in range(lo + 1, hi):
            chars[k] = "Z"
        result.append((coeff, "".join(chars)))
    return result


# Pauli multiplication table for single-qubit Paulis
_PMULT: Dict[Tuple[str, str], Tuple[complex, str]] = {
    ("I", "I"): (1, "I"), ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"), ("X", "X"): (1, "I"), ("X", "Y"): (1j, "Z"), ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"), ("Y", "X"): (-1j, "Z"), ("Y", "Y"): (1, "I"), ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"), ("Z", "X"): (1j, "Y"), ("Z", "Y"): (-1j, "X"), ("Z", "Z"): (1, "I"),
}


def _multiply_pauli_labels(a: str, b: str) -> Tuple[complex, str]:
    """Multiply two Pauli label strings, returning (phase, result)."""
    phase: complex = 1.0
    result = []
    for ca, cb in zip(a, b):
        p, s = _PMULT[(ca, cb)]
        phase *= p
        result.append(s)
    return phase, "".join(result)


def _generate_single_rdm2_term(
    n_qubits: int, p: int, q: int, r: int, s: int
) -> List[Tuple[complex, str]] | None:
    """Generate a single 2-RDM Pauli term for Gamma_pqrs = <a+_p a+_q a_s a_r>.
    Returns None if the term is trivially zero by antisymmetry.
    """
    # Skip trivially zero elements by antisymmetry
    if p == q or r == s:
        return None
    
    element_terms: List[Tuple[complex, str]] = []

    # Product: (a+_p a_r)(a+_q a_s)
    for c1, l1 in _jw_one_body_pauli(p, r, n_qubits):
        for c2, l2 in _jw_one_body_pauli(q, s, n_qubits):
            phase, merged = _multiply_pauli_labels(l1, l2)
            element_terms.append((c1 * c2 * phase, merged))

    # Correction: -delta_{qr} (a+_p a_s)
    if q == r:
        for c, l in _jw_one_body_pauli(p, s, n_qubits):
            element_terms.append((-c, l))

    return element_terms if element_terms else None


def _rdm2_pauli_terms(
    n_qubits: int,
) -> Dict[Tuple[int, int, int, int], List[Tuple[complex, str]]]:
    """Generate Pauli measurement terms for 2-RDM elements Gamma_pqrs.

    Gamma_pqrs = <a+_p a+_q a_s a_r>

    Using normal ordering:
        a+_p a+_q a_s a_r = (a+_p a_r)(a+_q a_s) - delta_{qr} (a+_p a_s)

    Each one-body operator is expanded via JW, then the products are
    computed using the Pauli multiplication table.

    Returns:
        Dict mapping (p, q, r, s) to list of (coefficient, pauli_label).
    """
    terms: Dict[Tuple[int, int, int, int], List[Tuple[complex, str]]] = {}

    for p in range(n_qubits):
        for q in range(n_qubits):
            for r in range(n_qubits):
                for s in range(n_qubits):
                    # Skip trivially zero elements by antisymmetry
                    if p == q or r == s:
                        continue
                    element_terms: List[Tuple[complex, str]] = []

                    # Product: (a+_p a_r)(a+_q a_s)
                    for c1, l1 in _jw_one_body_pauli(p, r, n_qubits):
                        for c2, l2 in _jw_one_body_pauli(q, s, n_qubits):
                            phase, merged = _multiply_pauli_labels(l1, l2)
                            element_terms.append((c1 * c2 * phase, merged))

                    # Correction: -delta_{qr} (a+_p a_s)
                    if q == r:
                        for c, l in _jw_one_body_pauli(p, s, n_qubits):
                            element_terms.append((-c, l))

                    if element_terms:
                        terms[(p, q, r, s)] = element_terms

    return terms


class RDMMeasurement:
    """Manages RDM extraction from quantum measurement results.

    Constructs measurement schedules and assembles RDM elements from
    Pauli expectation values measured on quantum hardware (Eqs. 10-11).

    Args:
        n_qubits: Number of qubits in the active space.
    """

    def __init__(self, n_qubits: int, max_terms: int | None = None,
                 use_stochastic_sampling: bool | None = None,
                 rng: np.random.Generator | None = None):
        self.n_qubits = n_qubits
        self.max_terms = max_terms or (2000 if n_qubits > 13 else 10000)
        # Policy: >13 qubits forces stochastic mode unless explicitly overridden
        if use_stochastic_sampling is None:
            self.use_stochastic_sampling = n_qubits > 13
        else:
            self.use_stochastic_sampling = use_stochastic_sampling
        self._rng = rng or np.random.default_rng(42)
        self._rdm1_terms = _rdm1_pauli_terms(n_qubits)
        # Lazy compute 2-RDM terms with safety guards for large qubit counts
        self._rdm2_terms: Dict[
            Tuple[int, int, int, int], List[Tuple[complex, str]]
        ] | None = None
        
        # Measurement burden estimate
        n_rdm2_terms_est = n_qubits**2 * (n_qubits - 1)**2
        n_pauli_per_term = 16  # typical for JW 2-body
        total_circuits_est = n_rdm2_terms_est * n_pauli_per_term
        _BURDEN_THRESHOLD = 500_000
        if total_circuits_est > _BURDEN_THRESHOLD:
            import warnings
            warnings.warn(
                f"RDM measurement burden for {n_qubits} qubits: ~{total_circuits_est} "
                f"Pauli terms (threshold={_BURDEN_THRESHOLD}). "
                f"Stochastic mode {'enabled' if self.use_stochastic_sampling else 'DISABLED - consider enabling'}.",
                UserWarning,
            )
        
        if n_qubits > 13 and not self.use_stochastic_sampling:
            import warnings
            warnings.warn(
                f"Full 2-RDM enumeration for {n_qubits} qubits is combinatorial! "
                f"Policy recommends use_stochastic_sampling=True for n_qubits > 13.",
                UserWarning,
            )

    def _ensure_rdm2_terms(self) -> None:
        """Lazily compute 2-RDM Pauli terms on first use with budget guards.
        For stochastic mode, samples terms without generating full set first for large qubit counts.
        """
        if self._rdm2_terms is not None:
            return
            
        n = self.n_qubits
        if self.use_stochastic_sampling:
            # Sample random (p,q,r,s) tuples directly, respecting antisymmetry
            sampled_terms = {}
            attempts = 0
            max_attempts = self.max_terms * 20  # avoid infinite loop
            
            while len(sampled_terms) < self.max_terms and attempts < max_attempts:
                attempts += 1
                p = int(self._rng.integers(0, n))
                q = int(self._rng.integers(0, n))
                r = int(self._rng.integers(0, n))
                s = int(self._rng.integers(0, n))
                # Antisymmetry: skip if p==q or r==s
                if p == q or r == s:
                    continue
                key = (p, q, r, s)
                if key in sampled_terms:
                    continue
                term = _generate_single_rdm2_term(n, p, q, r, s)
                if term is not None:
                    sampled_terms[key] = term
            
            self._rdm2_terms = sampled_terms
            self._stochastic_n_sampled = len(sampled_terms)
            # Estimate total non-zero terms for rescaling
            self._stochastic_n_total = n * (n - 1) * n * (n - 1)  # upper bound
            import warnings
            warnings.warn(
                f"Stochastic RDM: sampled {len(sampled_terms)}/{self._stochastic_n_total} "
                f"2-RDM terms. Rescaling applied to correct for sampling.",
                UserWarning
            )
        else:
            # Full enumeration for small systems or non-stochastic mode
            all_terms = _rdm2_pauli_terms(n)
            if len(all_terms) > self.max_terms:
                raise RuntimeError(
                    f"Full 2-RDM enumeration for {n} qubits requires {len(all_terms)} terms, "
                    f"exceeding max_terms limit of {self.max_terms}. Either increase max_terms, "
                    f"or set use_stochastic_sampling=True to use stochastic RDM estimation."
                )
            self._rdm2_terms = all_terms
            self._stochastic_n_sampled = len(all_terms)
            self._stochastic_n_total = len(all_terms)

    def measurement_bases(self) -> List[str]:
        """Return unique Pauli measurement bases needed for full RDM extraction."""
        bases = set()
        identity = "I" * self.n_qubits
        for term_list in self._rdm1_terms.values():
            for _, label in term_list:
                if label != identity:
                    bases.add(label)
        self._ensure_rdm2_terms()
        assert self._rdm2_terms is not None
        for term_list in self._rdm2_terms.values():
            for _, label in term_list:
                if label != identity:
                    bases.add(label)
        return sorted(bases)

    def grouped_measurement_bases(self) -> List[List[str]]:
        """Return Pauli bases grouped by QWC compatibility.

        Bases within the same group share a single measurement circuit
        (only one set of basis rotations is needed).  This enables
        measurement-basis reuse: all Pauli terms in a group can be
        reconstructed from the same set of fragment results.

        Returns:
            List of QWC groups, each a list of Pauli label strings.
        """
        from quantum_circuit_cutting.reconstruction import group_commuting_paulis
        all_bases = self.measurement_bases()
        return group_commuting_paulis(all_bases)

    def rdm1_bases(self) -> List[str]:
        """Return Pauli bases needed for 1-RDM only."""
        bases = set()
        identity = "I" * self.n_qubits
        for term_list in self._rdm1_terms.values():
            for _, label in term_list:
                if label != identity:
                    bases.add(label)
        return sorted(bases)

    def assemble_rdm1(self, pauli_expvals: Dict[str, float]) -> NDArray[np.float64]:
        """Assemble 1-RDM from measured Pauli expectation values.

        gamma_pq = <a+_p a_q>.  For a Hermitian density matrix the 1-RDM
        satisfies gamma_qp = gamma_pq*.  We compute the full complex value
        from all four JW Pauli terms (XX, YY, XY, YX) and enforce
        Hermiticity: rdm1 = (gamma + gamma^dagger) / 2, then take the
        real part (imaginary part vanishes for Hermitian RDMs).

        Args:
            pauli_expvals: Dict mapping Pauli label strings to measured
                expectation values.

        Returns:
            1-RDM matrix gamma_pq, shape (n_qubits, n_qubits).
        """
        n = self.n_qubits
        rdm1_c = np.zeros((n, n), dtype=complex)
        identity_key = "I" * n

        for (p, q), term_list in self._rdm1_terms.items():
            val: complex = 0.0
            for coeff, label in term_list:
                if label == identity_key:
                    val += coeff * 1.0  # <I> = 1
                elif label in pauli_expvals:
                    val += coeff * pauli_expvals[label]
                # else: unmeasured term, skip (not silently zeroed)
            rdm1_c[p, q] = val
            if p != q:
                rdm1_c[q, p] = val.conjugate()  # Hermiticity

        return rdm1_c.real

    def assemble_rdm2(self, pauli_expvals: Dict[str, float]) -> NDArray[np.float64]:
        """Assemble 2-RDM from measured Pauli expectation values.

        The 2-RDM Gamma_pqrs = <a+_p a+_q a_s a_r> is constructed from
        direct Pauli measurements of the two-body operator under JW.

        In stochastic mode, the sampled terms are rescaled by
        (N_total / N_sampled) to produce an unbiased estimator.
        Missing Pauli labels in pauli_expvals are treated as unmeasured
        (not silently zeroed) and logged.

        Args:
            pauli_expvals: Dict mapping Pauli labels to expectation values.

        Returns:
            2-RDM tensor Gamma_pqrs, shape (n, n, n, n).
        """
        n = self.n_qubits
        rdm2 = np.zeros((n, n, n, n))
        identity_key = "I" * n

        self._ensure_rdm2_terms()
        assert self._rdm2_terms is not None

        # Rescaling factor for stochastic sampling
        if self.use_stochastic_sampling and self._stochastic_n_sampled > 0:
            scale = self._stochastic_n_total / self._stochastic_n_sampled
        else:
            scale = 1.0

        n_missing = 0
        for (p, q, r, s), term_list in self._rdm2_terms.items():
            val: complex = 0.0
            for coeff, label in term_list:
                if label == identity_key:
                    val += coeff * 1.0
                elif label in pauli_expvals:
                    val += coeff * pauli_expvals[label]
                else:
                    # Unmeasured Pauli: skip this contribution (not silently zero)
                    n_missing += 1
            rdm2[p, q, r, s] = val.real * scale

        if n_missing > 0:
            import logging
            logging.getLogger(__name__).warning(
                "2-RDM assembly: %d Pauli terms missing from measurements "
                "(unmeasured, not zeroed)", n_missing
            )

        return rdm2
