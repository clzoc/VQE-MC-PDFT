"""Test-only simulation helpers.

These utilities are intentionally scoped to tests and must not be used by
production hardware paths.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import tensorcircuit as tc


def simulate_fragment_counts(
    circuit: tc.Circuit, n_qubits: int, n_shots: int, rng: np.random.Generator
) -> Dict[str, int]:
    """Sample measurement counts from a fragment statevector (test-only)."""
    state = np.asarray(circuit.state()).ravel()
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()
    n_basis = len(probs)
    samples = rng.choice(n_basis, size=n_shots, p=probs)
    counts: Dict[str, int] = {}
    for sample in samples:
        key = format(sample, f"0{n_qubits}b")
        counts[key] = counts.get(key, 0) + 1
    return counts
