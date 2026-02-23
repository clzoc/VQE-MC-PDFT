"""Overlap estimation via random Pauli measurements.

Implements |<psi1|psi2>|^2 estimation using the random Pauli protocol:

    |<psi1|psi2>|^2 ≈ (1/N) Σ_P <psi1|P|psi1> <psi2|P|psi2>

where P is a random Pauli operator sampled uniformly from the Pauli group.
Estimates are clamped to the [0, 1] interval to eliminate non-physical values
due to statistical sampling noise.  The random Pauli operators are generated
once at initialization and reused across all VQE iterations to avoid
inter-iteration noise.

**Limitation**: This estimator uses the identity
    |<psi1|psi2>|^2 = (1/2^n) sum_P <P>_1 * <P>_2
which is exact when the sum runs over all 4^n Pauli strings.  With a
finite random subset of M Paulis, the estimator is unbiased but has
variance O(1/M).  For large n_qubits (>16), the variance can be
significant and the orthogonality constraint in excited-state
calculations may be unreliable.  Consider increasing n_random_paulis
or using an alternative protocol (e.g., randomized unitary measurements)
for production excited-state calculations.

Reference: Huggins et al., Quantum Sci. Technol. 6, 015004 (2021).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Type alias for Pauli expectation value backend
PauliExpvalFn = Callable[
    [NDArray[np.float64], Optional[NDArray[np.float64]], List[str]],
    Dict[str, float],
]


def _generate_random_pauli_strings(
    n_qubits: int,
    n_samples: int,
    rng: np.random.Generator,
) -> List[str]:
    """Generate random Pauli strings for overlap estimation.

    Each string is a tensor product of single-qubit Paulis drawn
    uniformly from {I, X, Y, Z}.  The all-identity string is excluded
    since <I> = 1 for all states and contributes no information.
    """
    ops = ["I", "X", "Y", "Z"]
    strings = []
    for _ in range(n_samples):
        ps = "".join(rng.choice(ops) for _ in range(n_qubits))
        if all(c == "I" for c in ps):
            # Replace all-I with a random non-trivial string
            idx = int(rng.integers(0, n_qubits))
            ps = ps[:idx] + rng.choice(["X", "Y", "Z"]) + ps[idx + 1:]
        strings.append(ps)
    return strings


class OverlapEstimator:
    """Unbiased overlap estimator using random Pauli measurements.

    Estimates |<psi1|psi2>|^2 = (1/2^n) sum_P <P>_1 * <P>_2 where the
    sum runs over all 4^n Pauli strings.  In practice, a random subset
    of Pauli strings is sampled and the sum is rescaled.

    The random Pauli set is fixed at construction time and reused across
    all calls to ``estimate()``, ensuring reproducibility and avoiding
    inter-iteration noise in the VQE loop.

    Args:
        n_qubits: Number of qubits.
        n_random_paulis: Number of random Pauli strings to sample.
        seed: Random seed for reproducibility.
        var_threshold: Variance threshold for adaptive refinement.
            If the variance of the mean exceeds this, additional
            samples are drawn (up to ``max_paulis``).
        max_paulis: Maximum number of Pauli strings to use.
    """

    def __init__(
        self,
        n_qubits: int,
        n_random_paulis: int = 200,
        seed: int = 42,
        var_threshold: float = 1e-3,
        max_paulis: int = 2000,
    ):
        self.n_qubits = n_qubits
        self.var_threshold = var_threshold
        self.max_paulis = max_paulis
        self._rng = np.random.default_rng(seed)
        # Pre-generate fixed random Pauli set
        self._pauli_strings = _generate_random_pauli_strings(
            n_qubits, n_random_paulis, self._rng
        )

    def estimate(
        self,
        pauli_expval_fn: PauliExpvalFn,
        params1: NDArray[np.float64],
        params2: NDArray[np.float64],
        orb_angles: NDArray[np.float64] | None,
    ) -> float:
        """Estimate |<psi1|psi2>|^2 from Pauli expectation values.

        Uses the identity:
            |<psi1|psi2>|^2 = (1/2^n) sum_P <P>_1 * <P>_2

        where the sum is over all Pauli strings P (including identity).
        With M random samples, the estimator is:
            overlap ≈ (4^n / M) * sum_{sampled P} <P>_1 * <P>_2

        The factor 4^n/M accounts for the uniform sampling of Paulis
        from the full 4^n set, and the 1/2^n normalization.

        Returns:
            Estimated overlap in [0, 1]. Out-of-range values caused by
            finite-sampling noise are clipped to the physical interval
            with a warning.
        """
        expvals1 = pauli_expval_fn(params1, orb_angles, self._pauli_strings)
        expvals2 = pauli_expval_fn(params2, orb_angles, self._pauli_strings)

        products = np.array([
            expvals1[p] * expvals2[p] for p in self._pauli_strings
        ])

        # Check variance and adaptively refine if needed
        var_of_mean = np.var(products) / len(products) if len(products) > 1 else float("inf")
        if var_of_mean > self.var_threshold and len(self._pauli_strings) < self.max_paulis:
            n_extra = min(
                len(self._pauli_strings),
                self.max_paulis - len(self._pauli_strings),
            )
            extra_paulis = _generate_random_pauli_strings(
                self.n_qubits, n_extra, self._rng
            )
            self._pauli_strings.extend(extra_paulis)
            extra_ev1 = pauli_expval_fn(params1, orb_angles, extra_paulis)
            extra_ev2 = pauli_expval_fn(params2, orb_angles, extra_paulis)
            extra_products = np.array([
                extra_ev1[p] * extra_ev2[p] for p in extra_paulis
            ])
            products = np.concatenate([products, extra_products])
            logger.info(
                "Overlap estimator: refined to %d Paulis (var=%.2e)",
                len(self._pauli_strings), var_of_mean,
            )

        # Rescale: (4^n / M) * sum = 2^n * mean * 2^n / 1 ... 
        # Actually: |<psi1|psi2>|^2 = sum_P <P>_1 <P>_2 / 2^n
        # With M uniform samples from 4^n Paulis:
        # estimate = (4^n / M) * sum_sampled / 2^n = 2^n / M * sum_sampled
        n = self.n_qubits
        overlap = float(2**n * np.mean(products))

        if overlap < 0.0:
            logger.warning(
                "Overlap estimate %.4f < 0 (insufficient sampling), returning 0",
                overlap,
            )
            return 0.0
        if overlap > 1.0:
            logger.warning(
                "Overlap estimate %.4f > 1 (statistical fluctuation), returning 1",
                overlap,
            )
            return 1.0
        return overlap
