"""Inner-loop VQE optimizer with Adam.

Optimizes the ansatz parameters theta to minimize <H> for fixed orbitals,
using the Adam optimizer (lr=0.01, beta1=0.9, beta2=0.999) as specified
in SI Section S1.3.

The energy function evaluates <H> by building the ansatz circuit,
measuring Pauli expectation values through the provided backend path,
and summing the weighted contributions.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

import tensorcircuit as tc

logger = logging.getLogger(__name__)

# Default hyperparameters from SI Section S1.3
ADAM_LR = 0.01
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
VQE_CONV_THRESHOLD = 1e-6  # Ha
VQE_MAX_ITER = 200
PARAM_SHIFT = np.pi / 2


class AdamOptimizer:
    """Adam optimizer for VQE parameter updates.

    Matches the settings from SI Section S1.3:
    lr=0.01, beta1=0.9, beta2=0.999.
    """

    def __init__(
        self,
        lr: float = ADAM_LR,
        beta1: float = ADAM_BETA1,
        beta2: float = ADAM_BETA2,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: NDArray[np.float64] | None = None
        self.v: NDArray[np.float64] | None = None
        self.t = 0

    def step(
        self, params: NDArray[np.float64], grad: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        assert self.v is not None
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        self.m = None
        self.v = None
        self.t = 0


def parameter_shift_gradient(
    energy_fn: Callable[[NDArray[np.float64]], float],
    params: NDArray[np.float64],
    shift: float = np.pi / 2,
) -> NDArray[np.float64]:
    """Compute gradient via the parameter-shift rule.

    For each parameter theta_i:
    dE/dtheta_i = [E(theta_i + s) - E(theta_i - s)] / (2 sin(s))
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += shift
        params_minus = params.copy()
        params_minus[i] -= shift
        grad[i] = (energy_fn(params_plus) - energy_fn(params_minus)) / (
            2 * np.sin(shift)
        )
    return grad


def evaluate_hamiltonian_statevector(
    circuit: tc.Circuit,
    hamiltonian: Dict[str, float],
) -> float:
    """Evaluate <H> via exact statevector simulation.

    This helper is retained for local diagnostics and unit tests.

    For each Pauli string P_j with coefficient h_j, computes
    <psi|P_j|psi> from the statevector and sums h_j * <P_j>.

    Args:
        circuit: TensorCircuit Circuit with the prepared state.
        hamiltonian: {pauli_label: coefficient} dictionary.

    Returns:
        Total energy expectation value.
    """
    energy = 0.0
    for pauli_str, coeff in hamiltonian.items():
        if all(c == "I" for c in pauli_str):
            energy += coeff
            continue
        # Build list of (qubit, pauli) for non-identity operators
        ops = []
        for i, c in enumerate(pauli_str):
            if c != "I":
                ops.append((i, c))
        # Use TensorCircuit's expectation value
        expval = circuit.expectation_ps(
            x=[q for q, p in ops if p == "X"],
            y=[q for q, p in ops if p == "Y"],
            z=[q for q, p in ops if p == "Z"],
        )
        energy += coeff * float(np.real(expval))
    return energy


def evaluate_pauli_expvals_statevector(
    circuit: tc.Circuit,
    pauli_labels: list[str],
) -> Dict[str, float]:
    """Evaluate a list of Pauli expectation values via statevector.

    This helper is retained for local diagnostics and unit tests.

    Args:
        circuit: TensorCircuit Circuit with the prepared state.
        pauli_labels: List of Pauli label strings to measure.

    Returns:
        Dict mapping each label to its expectation value.
    """
    results: Dict[str, float] = {}
    for label in pauli_labels:
        if all(c == "I" for c in label):
            results[label] = 1.0
            continue
        ops_x = [i for i, c in enumerate(label) if c == "X"]
        ops_y = [i for i, c in enumerate(label) if c == "Y"]
        ops_z = [i for i, c in enumerate(label) if c == "Z"]
        expval = circuit.expectation_ps(x=ops_x, y=ops_y, z=ops_z)
        results[label] = float(np.real(expval))
    return results


class VQESolver:
    """Inner-loop VQE optimizer for the MC-PDFT framework.

    Optimizes ansatz parameters theta to minimize <H> for fixed orbitals.
    Convergence criterion: |dE| < 1e-6 Ha (SI Section S1.3).

    Args:
        energy_fn: Callable that takes params and returns energy.
        n_params: Number of variational parameters.
        max_iter: Maximum optimization iterations.
        conv_threshold: Energy convergence threshold in Hartree.
    """

    def __init__(
        self,
        energy_fn: Callable[[NDArray[np.float64]], float],
        n_params: int,
        max_iter: int = VQE_MAX_ITER,
        conv_threshold: float = VQE_CONV_THRESHOLD,
    ):
        self.energy_fn = energy_fn
        self.n_params = n_params
        self.max_iter = max_iter
        self.conv_threshold = conv_threshold
        self.optimizer = AdamOptimizer()

    def optimize(
        self, initial_params: NDArray[np.float64] | None = None
    ) -> Tuple[NDArray[np.float64], float, Dict[str, list]]:
        """Run VQE optimization loop.

        Returns:
            (optimal_params, final_energy, history_dict)
        """
        if initial_params is None:
            initial_params = np.random.default_rng(42).normal(0, 0.01, self.n_params)

        params = initial_params.copy()
        self.optimizer.reset()

        history: Dict[str, list] = {"energy": [], "grad_norm": []}
        energy = self.energy_fn(params)
        prev_energy = energy

        for iteration in range(self.max_iter):
            grad = parameter_shift_gradient(self.energy_fn, params)
            params = self.optimizer.step(params, grad)
            energy = self.energy_fn(params)

            history["energy"].append(energy)
            history["grad_norm"].append(float(np.linalg.norm(grad)))

            delta_e = abs(energy - prev_energy)
            logger.info(
                "VQE iter %d: E=%.10f, dE=%.2e, |g|=%.2e",
                iteration,
                energy,
                delta_e,
                np.linalg.norm(grad),
            )

            if delta_e < self.conv_threshold:
                logger.info("VQE converged at iteration %d", iteration)
                break
            prev_energy = energy

        return params, energy, history
