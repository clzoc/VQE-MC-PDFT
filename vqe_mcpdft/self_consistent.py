"""Self-consistent VQE-MC-PDFT optimization loop.

Implements the two-level optimization: inner VQE loop (fixed orbitals)
and outer MC-PDFT orbital optimization loop (Fig. S1 of SI).

The inner loop builds the ansatz circuit, evaluates <H> through an injected
backend function (typically hardware via `CuttingDispatcher`), and optimizes
the variational parameters. Upon convergence, the 1- and 2-RDMs are
measured and passed to the classical MC-PDFT module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from vqe_mcpdft.ansatz import CASCIAnsatz
from vqe_mcpdft.mcpdft_energy import MCPDFTEnergy
from vqe_mcpdft.orbital_rotation import OrbitalRotationCircuit
from vqe_mcpdft.rdm import RDMMeasurement
from vqe_mcpdft.vqe_solver import (
    VQESolver,
    evaluate_pauli_expvals_statevector,
)

# Type alias for injectable Pauli expectation value backend.
# Signature: (params, orbital_angles, pauli_labels) -> Dict[str, float]
PauliExpvalFn = Callable[
    [NDArray[np.float64], Optional[NDArray[np.float64]], List[str]],
    Dict[str, float],
]

logger = logging.getLogger(__name__)


@dataclass
class SCResult:
    """Result container for self-consistent VQE-MC-PDFT."""

    energy_mcpdft: float = 0.0
    energy_mcscf: float = 0.0
    energy_ontop: float = 0.0
    energy_variance: float = 0.0  # Variance estimate (0 if not computed)
    rdm1: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    rdm2: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    optimal_params: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    n_outer_iterations: int = 0
    converged: bool = False
    history: Dict[str, List[float]] = field(default_factory=dict)


class SelfConsistentVQEMCPDFT:
    """Self-consistent VQE-MC-PDFT driver (Fig. S1 of SI).

    Alternates between:
    1. Inner loop: VQE optimization of theta for fixed orbitals
    2. Outer loop: MC-PDFT orbital gradient -> orbital rotation update

    The ``energy_fn_factory`` is a callable that, given the current orbital
    rotation angles (or None for the first iteration), returns an energy
    function ``params -> float`` for the VQE inner loop.

    The recommended mode is hardware-backed execution through dispatcher
    callbacks. A statevector fallback remains only for legacy/local use
    when no Pauli-expval callback is provided.

    Args:
        ansatz: CASCI ansatz circuit builder.
        mcpdft: MC-PDFT energy evaluator.
        rdm_measurer: RDM measurement manager.
        orbital_rotator: Orbital rotation circuit compiler.
        energy_fn_factory: Factory creating energy function for VQE.
        max_outer_iter: Maximum outer-loop iterations.
        orbital_conv_threshold: Convergence threshold for orbital gradient.
        orbital_step_size: Step size alpha for orbital update (Eq. 22).
        pauli_expval_fn: Optional backend function for Pauli expectation values.
            Signature: (params, orbital_angles, pauli_labels) -> Dict[str, float].
            If provided, used for hardware-backed execution. If None, falls back
            to statevector simulation (for local development/testing only).
    """

    def __init__(
        self,
        ansatz: CASCIAnsatz,
        mcpdft: MCPDFTEnergy,
        rdm_measurer: RDMMeasurement,
        orbital_rotator: OrbitalRotationCircuit,
        energy_fn_factory: Callable,
        max_outer_iter: int = 20,
        orbital_conv_threshold: float = 1e-5,
        orbital_step_size: float = 0.01,
        pauli_expval_fn: PauliExpvalFn | None = None,
    ):
        self.ansatz = ansatz
        self.mcpdft = mcpdft
        self.rdm_measurer = rdm_measurer
        self.orbital_rotator = orbital_rotator
        self.energy_fn_factory = energy_fn_factory
        self.max_outer_iter = max_outer_iter
        self.orbital_conv_threshold = orbital_conv_threshold
        self.orbital_step_size = orbital_step_size
        self.pauli_expval_fn = pauli_expval_fn

    def run(self, initial_params: NDArray[np.float64] | None = None) -> SCResult:
        """Execute the full self-consistent VQE-MC-PDFT loop.

        Returns:
            SCResult with converged energies, RDMs, and parameters.
        """
        result = SCResult()
        result.history = {"energy": [], "grad_norm": []}

        params = initial_params
        orbital_angles: NDArray[np.float64] | None = None
        e_mcpdft = 0.0
        e_mcscf = 0.0
        e_ot = 0.0
        rdm1 = np.array([])
        rdm2 = np.array([])

        for outer_iter in range(self.max_outer_iter):
            logger.info("=== Outer iteration %d ===", outer_iter)

            # Step 2: Inner VQE loop for CI coefficients
            energy_fn = self.energy_fn_factory(orbital_angles)
            vqe = VQESolver(
                energy_fn=energy_fn,
                n_params=self.ansatz.n_params,
            )
            params, vqe_energy, _ = vqe.optimize(params)

            # Step 2b: Measure RDMs from the converged circuit
            pauli_bases = self.rdm_measurer.measurement_bases()
            if self.pauli_expval_fn is not None:
                # Use injected backend (e.g. cutting-based reconstruction)
                pauli_expvals = self.pauli_expval_fn(params, orbital_angles, pauli_bases)
            else:
                # Legacy fallback path for local development only.
                circuit = self.ansatz.build_circuit(params, orbital_angles)
                pauli_expvals = evaluate_pauli_expvals_statevector(circuit, pauli_bases)

            rdm1 = self.rdm_measurer.assemble_rdm1(pauli_expvals)
            rdm2 = self.rdm_measurer.assemble_rdm2(pauli_expvals)

            # Step 3: Classical MC-PDFT analysis
            e_mcpdft, e_mcscf, e_ot = self.mcpdft.evaluate(rdm1, rdm2)
            logger.info(
                "E_MC-PDFT=%.10f (MCSCF=%.10f, E_ot=%.10f)",
                e_mcpdft,
                e_mcscf,
                e_ot,
            )

            # Step 4: Orbital gradient and convergence check
            orbital_gradient = self.mcpdft.orbital_gradient(rdm1, rdm2)
            grad_norm = float(np.linalg.norm(orbital_gradient))
            logger.info("Orbital gradient norm: %.6e", grad_norm)

            result.history["energy"].append(e_mcpdft)
            result.history["grad_norm"].append(grad_norm)

            if grad_norm < self.orbital_conv_threshold:
                logger.info(
                    "Orbital optimization converged at iteration %d", outer_iter
                )
                result.converged = True
                result.energy_mcpdft = e_mcpdft
                result.energy_mcscf = e_mcscf
                result.energy_ontop = e_ot
                result.rdm1 = rdm1
                result.rdm2 = rdm2
                result.optimal_params = params
                result.n_outer_iterations = outer_iter + 1
                return result

            # Step 5: Update orbital rotation and compile to circuit
            orbital_angles = self.orbital_rotator.update(
                orbital_gradient, self.orbital_step_size
            )

        logger.warning(
            "Outer loop did not converge in %d iterations", self.max_outer_iter
        )
        result.energy_mcpdft = e_mcpdft
        result.energy_mcscf = e_mcscf
        result.energy_ontop = e_ot
        result.rdm1 = rdm1
        result.rdm2 = rdm2
        result.optimal_params = params if params is not None else np.array([])
        result.n_outer_iterations = self.max_outer_iter
        return result
