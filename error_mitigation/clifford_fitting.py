"""Clifford Data Regression (CDR) / Clifford Fitting error mitigation.

Learns a linear model  E_ideal = a * E_noisy + b  from near-Clifford
training circuits that can be exactly simulated classically, then applies
the correction to the noisy result of the target circuit.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class CliffordFitter:
    """Linear Clifford-fitting error mitigator.

    Attributes:
        a: Slope of the fitted linear model.
        b: Intercept of the fitted linear model.
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    # ------------------------------------------------------------------
    # Training circuit generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_training_circuits(
        circuit,
        n_training: int = 20,
        rng: Optional[np.random.Generator] = None,
    ) -> List:
        """Create near-Clifford training circuits.

        Each non-Clifford gate in *circuit* is independently replaced by a
        randomly chosen Clifford gate (from {I, X, Y, Z, H, S}) with
        probability 1, producing circuits that are classically simulable.

        Args:
            circuit: A ``tensorcircuit.Circuit`` instance.
            n_training: Number of training circuits to generate.
            rng: Optional numpy random generator for reproducibility.

        Returns:
            List of ``tensorcircuit.Circuit`` training circuits.
        """
        import tensorcircuit as tc

        rng = rng or np.random.default_rng()
        clifford_gates = ["i", "x", "y", "z", "h", "s"]
        non_clifford = {"rx", "ry", "rz", "rxx", "ryy", "rzz", "u", "any"}

        ir = circuit.to_qir()
        n = circuit._nqubits
        training = []

        for _ in range(n_training):
            c = tc.Circuit(n)
            for gate_info in ir:
                name = gate_info["name"]
                qubits = gate_info["index"]
                if name in non_clifford:
                    replacement = rng.choice(clifford_gates)
                    getattr(c, replacement)(*qubits[:1])
                else:
                    params = gate_info.get("parameters", {})
                    getattr(c, name)(*qubits, **params)
            training.append(c)

        return training

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        ideal_values: np.ndarray | List[float],
        noisy_values: np.ndarray | List[float],
    ) -> Tuple[float, float]:
        """Fit the linear model E_ideal = a * E_noisy + b.

        Args:
            ideal_values: Classically computed expectation values of
                training circuits.
            noisy_values: Noisy (device/simulator) expectation values of
                the same training circuits.

        Returns:
            Tuple ``(a, b)`` of the fitted parameters.
        """
        coeffs = np.polyfit(noisy_values, ideal_values, 1)
        self.a, self.b = float(coeffs[0]), float(coeffs[1])
        return self.a, self.b

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def correct(self, noisy_value: float) -> float:
        """Apply the linear correction to a noisy expectation value.

        Args:
            noisy_value: Noisy expectation value from the target circuit.

        Returns:
            Mitigated expectation value.
        """
        return self.a * noisy_value + self.b
