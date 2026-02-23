"""Zero-Noise Extrapolation (ZNE) via local unitary folding.

Implements the folding rule G -> G (G^dag G)^k so that the effective
noise factor is lambda = 2k + 1, then extrapolates to the zero-noise
limit with a polynomial fit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class ZeroNoiseExtrapolator:
    """ZNE with local unitary folding for TensorCircuit circuits.

    Attributes:
        degree: Polynomial degree used for Richardson-like extrapolation.
    """

    def __init__(self, degree: int = 2) -> None:
        self.degree = degree

    # ------------------------------------------------------------------
    # Circuit folding
    # ------------------------------------------------------------------

    @staticmethod
    def fold_circuit(circuit, scale_factor: int):
        """Fold *circuit* to amplify noise by *scale_factor*.

        Applies the local unitary folding rule:
            G -> G (G^dag G)^k,  where scale_factor = 2k + 1.

        Args:
            circuit: A ``tensorcircuit.Circuit`` instance.
            scale_factor: Odd integer noise amplification factor (1, 3, 5, ...).

        Returns:
            New ``tensorcircuit.Circuit`` with the folded gate sequence.

        Raises:
            ValueError: If *scale_factor* is not a positive odd integer.
        """
        if scale_factor < 1 or scale_factor % 2 == 0:
            raise ValueError("scale_factor must be a positive odd integer")

        import tensorcircuit as tc

        k = (scale_factor - 1) // 2
        n = circuit._nqubits
        folded = tc.Circuit(n)

        # Replay original gates
        ir = circuit.to_qir()
        for gate_info in ir:
            name = gate_info["name"]
            qubits = gate_info["index"]
            params = gate_info.get("parameters", {})
            getattr(folded, name)(*qubits, **params)

        # Append (G^dag G) k times
        reversed_ir = list(reversed(ir))
        # Gates whose adjoint requires negating the angle parameter
        _parametric = {"rx", "ry", "rz", "rxx", "ryy", "rzz"}

        for _ in range(k):
            # G^dag: reversed order with conjugate-transposed parameters
            for gate_info in reversed_ir:
                name = gate_info["name"]
                qubits = gate_info["index"]
                params = dict(gate_info.get("parameters", {}))
                if name in _parametric:
                    # Adjoint of R(theta) = R(-theta)
                    for key in list(params):
                        params[key] = -params[key]
                elif name == "s":
                    # S† = Sdg
                    name = "sdg"
                elif name == "t":
                    name = "tdg"
                getattr(folded, name)(*qubits, **params)
            # G: forward pass again (original parameters)
            for gate_info in ir:
                name = gate_info["name"]
                qubits = gate_info["index"]
                params = gate_info.get("parameters", {})
                getattr(folded, name)(*qubits, **params)

        return folded

    # ------------------------------------------------------------------
    # Extrapolation
    # ------------------------------------------------------------------

    def extrapolate(
        self,
        noise_levels: Sequence[float],
        expectation_values: Sequence[float],
    ) -> float:
        """Extrapolate to the zero-noise limit via polynomial fit.

        Args:
            noise_levels: Noise scale factors (e.g. [1, 3, 5]).
            expectation_values: Corresponding measured expectation values.

        Returns:
            Estimated zero-noise expectation value.
        """
        coeffs = np.polyfit(noise_levels, expectation_values, self.degree)
        return float(np.polyval(coeffs, 0.0))

    def mitigate_expectation(
        self,
        circuit,
        backend,
        scale_factors: Sequence[int] = (1, 3, 5),
    ) -> float:
        """Mitigate a single expectation value via ZNE.

        Runs the circuit at multiple noise levels (via unitary folding),
        computes the Z-basis parity expectation at each level, and
        extrapolates to zero noise.

        This operates at the expectation-value level, preserving the
        full measurement distribution for each noise level.

        Args:
            circuit: Original circuit to mitigate.
            backend: TQP backend for circuit execution.
            scale_factors: Noise amplification factors (odd integers).

        Returns:
            Zero-noise extrapolated expectation value.
        """
        expvals = []
        for sf in scale_factors:
            folded = self.fold_circuit(circuit, sf)
            counts = backend.submit_circuit(folded)
            total = sum(counts.values())
            if total == 0:
                expvals.append(0.0)
                continue
            parity = sum(
                ((-1) ** bin(int(bs, 2)).count("1")) * c
                for bs, c in counts.items()
            ) / total
            expvals.append(parity)
        return self.extrapolate(list(scale_factors), expvals)
