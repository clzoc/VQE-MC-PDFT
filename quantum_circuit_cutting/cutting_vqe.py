"""Partitioned VQE-MC-PDFT driver with circuit cutting.

Integrates circuit partitioning, fragment execution on TQP, classical
reconstruction, and the self-consistent MC-PDFT loop for large active
spaces exceeding the device qubit limit (Fig. 6 of manuscript).

Upgraded with:
- Overhead-aware configuration sampling (shared seed for gradient pairs)
- Automatic strategy selection (exact vs sampling)
- Observable grouping in reconstruction
- Proper parameter-shift gradient with sin(s) denominator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from quantum_circuit_cutting.channel_decomposition import ChannelDecomposition
from quantum_circuit_cutting.fragment_circuits import FragmentCircuitGenerator
from quantum_circuit_cutting.overhead import (
    compute_overhead_budget,
    select_cut_strategy,
)
from quantum_circuit_cutting.partition import CircuitPartitioner, ClusterPartition
from quantum_circuit_cutting.reconstruction import CuttingReconstructor, MeasurementCache
from vqe_mcpdft.ansatz import CASCIAnsatz
from vqe_mcpdft.tqp_backend import TQPBackend
from vqe_mcpdft.vqe_solver import VQESolver

logger = logging.getLogger(__name__)

PARAM_SHIFT = np.pi / 2  # Standard parameter-shift rule


@dataclass
class CuttingVQEResult:
    """Result of a partitioned VQE-MC-PDFT calculation."""

    energy: float = 0.0
    energy_mcscf: float = 0.0
    energy_variance: float = 0.0  # Variance from cutting reconstruction
    binding_energy_eV: float = 0.0
    n_qubits_used: int = 0
    n_clusters: int = 0
    n_cuts: int = 0
    n_configurations: int = 0
    sampling_overhead_kappa: float = 1.0
    converged: bool = False
    history: Dict[str, list] = field(default_factory=dict)
    rdm1: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    rdm2: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


class PartitionedVQEMCPDFT:
    """Full partitioned VQE-MC-PDFT workflow with circuit cutting.

    Implements the scalability study from Fig. 6: partitions a large
    active-space circuit into fragments executable on Tianji-S2,
    reconstructs expectation values classically, and runs the
    self-consistent MC-PDFT loop.

    Args:
        n_qubits: Total logical qubits (= 2 * n_active_orbitals).
        n_electrons: Number of active electrons.
        max_fragment_qubits: Device qubit limit (13 for Tianji-S2).
        tqp_backend: Configured TQP backend for circuit execution.
        n_vqe_layers: Number of ansatz layers.
        use_sampling: Use sampling instead of exact enumeration.
            If None, auto-selected based on problem size.
        n_samples: Number of samples per VQE iteration.
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        max_fragment_qubits: int = 13,
        tqp_backend: TQPBackend | None = None,
        n_vqe_layers: int = 2,
        use_sampling: bool | None = None,
        n_samples: int = 1000,
        mcpdft = None,
        rdm_measurer = None,
        orb_rotator = None,
        max_outer_iter: int = 20,
        orbital_conv_threshold: float = 1e-5,
        orbital_step_size: float = 0.01,
        mitigation_pipeline: List[Callable] | None = None,
        expval_mitigator: Optional[Callable[[float, str], float]] = None,
    ):
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.max_fragment_qubits = max_fragment_qubits
        self.backend = tqp_backend
        self.n_vqe_layers = n_vqe_layers
        self.n_samples = n_samples
        self.mcpdft = mcpdft
        self.rdm_measurer = rdm_measurer
        self.orb_rotator = orb_rotator
        self.max_outer_iter = max_outer_iter
        self.orbital_conv_threshold = orbital_conv_threshold
        self.orbital_step_size = orbital_step_size
        self.mitigation_pipeline = mitigation_pipeline or []
        self.expval_mitigator = expval_mitigator

        # Build ansatz
        self.ansatz = CASCIAnsatz(n_qubits, n_electrons, n_vqe_layers)

        # Partition circuit
        self.partitioner = CircuitPartitioner(n_qubits, max_fragment_qubits)

        # Auto-select sampling strategy if not specified
        if use_sampling is None:
            strategy = select_cut_strategy(
                n_qubits,
                max_fragment_qubits,
                self.ansatz.get_qubit_pairs(),
            )
            self.use_sampling: bool = bool(strategy["use_sampling"])
            self.n_samples: int = int(strategy["n_samples"])
            logger.info("Auto-selected strategy: %s", strategy["reason"])
        else:
            self.use_sampling: bool = use_sampling

    def run(
        self,
        hamiltonian: Dict[str, float],
        entangling_gates: List[Tuple[int, int]] | None = None,
        orbital_symmetries: Dict[int, str] | None = None,
        atom_energy: float = 0.0,
        max_vqe_iter: int = 100,
        vqe_conv: float = 1e-6,
    ) -> CuttingVQEResult:
        """Execute the full partitioned VQE-MC-PDFT calculation.

        Args:
            hamiltonian: Qubit Hamiltonian as {pauli_string: coefficient}.
            entangling_gates: Two-qubit gate pairs. If None, inferred.
            orbital_symmetries: MO symmetry labels for guided partitioning.
            atom_energy: Isolated atom energy for binding energy calc.
            max_vqe_iter: Maximum VQE iterations.
            vqe_conv: VQE energy convergence threshold (Ha).

        Returns:
            CuttingVQEResult with energy and convergence info.
        """
        result = CuttingVQEResult()
        result.n_qubits_used = self.n_qubits
        result.history = {"energy": [], "iteration": []}

        # Hamiltonian term screening: drop terms with |coeff| below threshold
        _SCREEN_THRESHOLD = 1e-10
        n_original = len(hamiltonian)
        hamiltonian = {
            ps: c for ps, c in hamiltonian.items() if abs(c) > _SCREEN_THRESHOLD
        }
        if len(hamiltonian) < n_original:
            logger.info(
                "Hamiltonian screening: %d -> %d terms (threshold=%.1e)",
                n_original, len(hamiltonian), _SCREEN_THRESHOLD,
            )

        # Step 1: Partition
        if entangling_gates is None:
            entangling_gates = self.ansatz.get_qubit_pairs()

        partition = self.partitioner.partition(entangling_gates, orbital_symmetries)
        result.n_clusters = partition.n_clusters
        result.n_cuts = partition.n_cuts

        # Compute overhead budget and check feasibility
        budget = compute_overhead_budget(self.n_qubits, self.max_fragment_qubits)
        result.sampling_overhead_kappa = partition.overhead_kappa

        from quantum_circuit_cutting.overhead import estimate_sampling_cost
        from quantum_circuit_cutting.reconstruction import group_commuting_paulis
        non_id_terms = [ps for ps in hamiltonian if not all(c == "I" for c in ps)]
        n_qwc_groups = len(group_commuting_paulis(non_id_terms)) if non_id_terms else 0
        cost = estimate_sampling_cost(
            partition.n_cuts, len(non_id_terms), partition.n_clusters, n_qwc_groups,
        )
        _MAX_PROJECTED_CIRCUITS = 10_000_000
        if cost["total_circuits_sampled"] > _MAX_PROJECTED_CIRCUITS and not self.use_sampling:
            raise RuntimeError(
                f"Projected circuit budget {cost['total_circuits_sampled']:.0f} exceeds "
                f"limit {_MAX_PROJECTED_CIRCUITS}. Use sampling mode or reduce active space. "
                f"({partition.n_cuts} cuts, {n_qwc_groups} QWC groups, "
                f"{partition.n_clusters} clusters)"
            )

        logger.info(
            "Partitioned %d qubits into %d clusters with %d cuts "
            "(kappa=%.1f, budget max_cuts=%d, projected_circuits=%.0f)",
            self.n_qubits,
            partition.n_clusters,
            partition.n_cuts,
            partition.overhead_kappa,
            budget.max_cuts,
            cost["total_circuits_sampled"],
        )

        # Step 2: Set up decomposition and reconstructor
        decomposition = ChannelDecomposition(partition.n_cuts)
        reconstructor = CuttingReconstructor(
            partition, decomposition, expval_mitigator=self.expval_mitigator,
        )
        result.n_configurations = decomposition.n_configurations

        # Step 3: VQE optimization loop
        params = self.ansatz.initial_params()
        prev_energy = float("inf")
        energy = 0.0

        # Fix a master seed for reproducible config sampling
        master_rng = np.random.default_rng(42)

        for iteration in range(max_vqe_iter):
            # Generate shared configs for this iteration
            iter_seed = int(master_rng.integers(0, 2**31))
            if self.use_sampling:
                configs = decomposition.sample_configurations(
                    self.n_samples, rng=np.random.default_rng(iter_seed)
                )
            else:
                configs = decomposition.enumerate_configurations()

            # Reconstruct energy per QWC group with correct measurement bases
            energy = self._evaluate_energy_by_group(
                hamiltonian, partition, decomposition, reconstructor,
                params, None, configs,
            )

            result.history["energy"].append(energy)
            result.history["iteration"].append(iteration)

            delta_e = abs(energy - prev_energy)
            logger.info(
                "Cutting VQE iter %d: E=%.10f Ha, dE=%.2e",
                iteration,
                energy,
                delta_e,
            )

            if delta_e < vqe_conv:
                result.converged = True
                break
            prev_energy = energy

            # Parameter update using shared configs for gradient
            params = self._update_params(
                params,
                hamiltonian,
                partition,
                decomposition,
                reconstructor,
                configs,
            )

        # If MCPDFT components are provided, run full MC-PDFT self-consistent loop
        if self.mcpdft is not None and self.rdm_measurer is not None and self.orb_rotator is not None:
            logger.info("Running full self-consistent VQE-MC-PDFT loop")
            orbital_angles: NDArray[np.float64] | None = None
            e_mcpdft = 0.0
            e_mcscf = 0.0
            rdm1 = np.array([])
            rdm2 = np.array([])

            for outer_iter in range(self.max_outer_iter):
                logger.info("=== Outer iteration %d ===", outer_iter)
                outer_seed = int(master_rng.integers(0, 2**31))
                
                # Inner VQE loop
                def energy_fn_factory(orb_angles):
                    def energy_fn(params):
                        if self.use_sampling:
                            configs = decomposition.sample_configurations(
                                self.n_samples, rng=np.random.default_rng(outer_seed)
                            )
                        else:
                            configs = decomposition.enumerate_configurations()
                        return self._evaluate_energy_by_group(
                            hamiltonian, partition, decomposition, reconstructor,
                            params, orb_angles, configs,
                        )
                    return energy_fn

                energy_fn = energy_fn_factory(orbital_angles)
                vqe = VQESolver(
                    energy_fn=energy_fn,
                    n_params=self.ansatz.n_params,
                )
                params, vqe_energy, _ = vqe.optimize(params)

                # Measure RDMs: split Pauli bases into QWC groups,
                # execute each group with its own measurement rotations,
                # reconstruct each Pauli from its own group's results only.
                # Uses MeasurementCache to reuse results across groups that
                # share the same measurement basis on the same fragment.
                from quantum_circuit_cutting.reconstruction import (
                    group_commuting_paulis as _group_commuting_paulis,
                    measurement_basis_key,
                )
                
                pauli_bases = self.rdm_measurer.measurement_bases()
                if self.use_sampling:
                    configs = decomposition.sample_configurations(
                        self.n_samples, rng=np.random.default_rng(outer_seed)
                    )
                else:
                    configs = decomposition.enumerate_configurations()
                
                # Separate identity from non-identity
                pauli_expvals = {}
                non_identity_bases = []
                for label in pauli_bases:
                    if all(c == "I" for c in label):
                        pauli_expvals[label] = 1.0
                    else:
                        non_identity_bases.append(label)
                
                # Group into QWC-compatible sets
                qwc_groups = _group_commuting_paulis(non_identity_bases)
                logger.info(
                    "RDM measurement: %d bases -> %d QWC groups",
                    len(non_identity_bases), len(qwc_groups),
                )

                base_generator = FragmentCircuitGenerator(
                    partition=partition,
                    decomposition=decomposition,
                    ansatz_params=params,
                    orbital_rotation_angles=orbital_angles,
                    n_electrons=self.n_electrons,
                    n_layers=self.n_vqe_layers,
                )
                effective_configs = base_generator.expand_configurations(configs)
                
                # Cache: reuse fragment results when two groups share the
                # same measurement basis on the same (config, cluster).
                meas_cache = MeasurementCache()
                
                # Execute each QWC group with its own measurement basis
                for group in qwc_groups:
                    basis_key = measurement_basis_key(group)
                    group_fragments = base_generator.generate_all(
                        configurations=effective_configs, pauli_group=group
                    )

                    # Check cache before executing
                    group_results: Dict[Tuple[int, int], Dict[str, int]] = {}
                    uncached_frags = []
                    uncached_meta = []
                    for config_frags in group_fragments:
                        for f in config_frags:
                            ck = meas_cache.make_key(
                                f.config_idx, f.cluster_idx, basis_key
                            )
                            if ck in meas_cache:
                                group_results[(f.config_idx, f.cluster_idx)] = meas_cache[ck]
                            else:
                                uncached_frags.append(f)
                                uncached_meta.append(ck)

                    # Execute only uncached fragments
                    if uncached_frags:
                        if self.backend is None:
                            raise RuntimeError(
                                "TQP backend is required for execution. "
                                "Simulator fallback is not supported per hardware-only policy."
                            )
                        for f, ck in zip(uncached_frags, uncached_meta):
                            counts = self.backend.submit_circuit(f.circuit)
                            for mit_fn in self.mitigation_pipeline:
                                counts = mit_fn(counts, f.circuit, f.n_qubits)
                            meas_cache[ck] = counts
                            group_results[(f.config_idx, f.cluster_idx)] = counts
                    
                    # Reconstruct only this group's Paulis from this group's data
                    for label in group:
                        pauli_expvals[label] = reconstructor.reconstruct_expectation(
                            group_results, label, effective_configs
                        )

                logger.info("RDM measurement cache: %s", meas_cache.summary())
                
                rdm1 = self.rdm_measurer.assemble_rdm1(pauli_expvals)
                rdm2 = self.rdm_measurer.assemble_rdm2(pauli_expvals)

                # Evaluate MC-PDFT energy
                e_mcpdft, e_mcscf, e_ot = self.mcpdft.evaluate(rdm1, rdm2)
                logger.info(
                    "E_MC-PDFT=%.10f (MCSCF=%.10f, E_ot=%.10f)",
                    e_mcpdft,
                    e_mcscf,
                    e_ot,
                )

                # Compute orbital gradient
                orbital_gradient = self.mcpdft.orbital_gradient(rdm1, rdm2)
                grad_norm = float(np.linalg.norm(orbital_gradient))
                logger.info("Orbital gradient norm: %.6e", grad_norm)

                if grad_norm < self.orbital_conv_threshold:
                    logger.info(
                        "Orbital optimization converged at iteration %d", outer_iter
                    )
                    break

                # Update orbital angles
                orbital_angles = self.orb_rotator.update(
                    orbital_gradient, self.orbital_step_size
                )

            # Set final MC-PDFT energy as result
            result.energy = e_mcpdft
            result.energy_mcscf = e_mcscf
            result.rdm1 = rdm1
            result.rdm2 = rdm2
        else:
            # Fallback to pure VQE energy if no MC-PDFT components provided
            result.energy = energy
        
        if atom_energy != 0.0:
            result.binding_energy_eV = (result.energy - 2 * atom_energy) * 27.2114
        return result

    def _evaluate_energy_by_group(
        self,
        hamiltonian: Dict[str, float],
        partition: ClusterPartition,
        decomposition: ChannelDecomposition,
        reconstructor: CuttingReconstructor,
        params: NDArray[np.float64],
        orbital_angles: NDArray[np.float64] | None,
        configs: List[Tuple[Tuple[int, ...], complex]],
    ) -> float:
        """Evaluate <H> by splitting into QWC groups with correct measurement bases.

        Each group gets its own fragment circuits (with the correct
        measurement rotations) and reconstruction uses only that group's data.
        A shared MeasurementCache avoids re-executing fragments when
        multiple groups share the same measurement basis on the same fragment.
        """
        from quantum_circuit_cutting.reconstruction import (
            group_commuting_paulis as _group_commuting_paulis, measurement_basis_key,
        )

        energy = 0.0
        non_identity_ham: Dict[str, float] = {}
        for label, coeff in hamiltonian.items():
            if all(c == "I" for c in label):
                energy += coeff
            else:
                non_identity_ham[label] = coeff

        if not non_identity_ham:
            return energy

        generator = FragmentCircuitGenerator(
            partition=partition,
            decomposition=decomposition,
            ansatz_params=params,
            orbital_rotation_angles=orbital_angles,
            n_electrons=self.n_electrons,
            n_layers=self.n_vqe_layers,
        )
        effective_configs = generator.expand_configurations(configs)

        shared_cache = MeasurementCache()
        qwc_groups = _group_commuting_paulis(list(non_identity_ham.keys()))
        for group in qwc_groups:
            basis_key = measurement_basis_key(group)
            group_fragments = generator.generate_all(
                configurations=effective_configs, pauli_group=group
            )
            group_results = self._execute_fragments(
                generator, group_fragments,
                measurement_cache=shared_cache, basis_key=basis_key,
            )
            group_ham = {label: non_identity_ham[label] for label in group}
            energy += reconstructor.reconstruct_hamiltonian_energy(
                group_results, group_ham, effective_configs
            )

        if shared_cache.hits > 0:
            logger.info("Energy eval cache: %s", shared_cache.summary())
        return energy

    def _execute_fragments(
        self,
        generator: FragmentCircuitGenerator,
        all_fragments: list,
        measurement_cache: MeasurementCache | None = None,
        basis_key: str = "",
    ) -> Dict[Tuple[int, int], Dict[str, int]]:
        """Execute fragment circuits and collect results.

        Submits circuits to TQP backend for hardware execution.
        Applies mitigation_pipeline to each fragment's counts if configured.
        Uses measurement_cache to avoid re-executing fragments with the
        same (config_idx, cluster_idx, basis_key).

        Raises:
            RuntimeError: If no TQP backend is configured.
        """
        if measurement_cache is None:
            measurement_cache = MeasurementCache()

        fragment_results: Dict[Tuple[int, int], Dict[str, int]] = {}
        uncached_circuits = []
        uncached_meta = []
        uncached_frags = []

        # Check cache first
        for frags in all_fragments:
            for f in frags:
                ck = measurement_cache.make_key(f.config_idx, f.cluster_idx, basis_key)
                if ck in measurement_cache:
                    fragment_results[(f.config_idx, f.cluster_idx)] = measurement_cache[ck]
                else:
                    uncached_circuits.append(f.circuit)
                    uncached_meta.append((f.config_idx, f.cluster_idx))
                    uncached_frags.append(f)

        # Execute only uncached fragments
        if uncached_circuits:
            if self.backend is not None:
                results_list = self.backend.submit_batch(uncached_circuits)
            else:
                raise RuntimeError(
                    "TQP backend is required for execution. "
                    "Simulator fallback is not supported per hardware-only policy."
                )

            # Apply mitigation pipeline
            for i, (counts, frag) in enumerate(zip(results_list, uncached_frags)):
                for mit_fn in self.mitigation_pipeline:
                    counts = mit_fn(counts, frag.circuit, frag.n_qubits)
                results_list[i] = counts

            n_empty = 0
            for (config_idx, cluster_idx), counts in zip(uncached_meta, results_list):
                ck = measurement_cache.make_key(config_idx, cluster_idx, basis_key)
                measurement_cache[ck] = counts
                fragment_results[(config_idx, cluster_idx)] = counts
                if not counts:
                    n_empty += 1
            if n_empty:
                logger.warning(
                    "%d/%d fragment results are empty (hardware failure, degraded)",
                    n_empty, len(uncached_meta),
                )

        return fragment_results

    def _update_params(
        self,
        params: NDArray[np.float64],
        hamiltonian: Dict[str, float],
        partition: ClusterPartition,
        decomposition: ChannelDecomposition,
        reconstructor: CuttingReconstructor,
        configs: List[Tuple[Tuple[int, ...], complex]],
    ) -> NDArray[np.float64]:
        """Update VQE parameters via parameter-shift gradient.

        Uses the SAME sampled configurations for both E(+s) and E(-s)
        to ensure a valid, low-variance gradient estimate.
        Energy evaluation uses QWC-group-aware reconstruction.
        """
        grad = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += PARAM_SHIFT
            params_minus = params.copy()
            params_minus[i] -= PARAM_SHIFT

            e_plus = self._evaluate_energy_by_group(
                hamiltonian, partition, decomposition, reconstructor,
                params_plus, None, configs,
            )
            e_minus = self._evaluate_energy_by_group(
                hamiltonian, partition, decomposition, reconstructor,
                params_minus, None, configs,
            )

            # Proper parameter-shift rule: dE/dtheta = (E+ - E-) / (2 sin(s))
            grad[i] = (e_plus - e_minus) / (2 * np.sin(PARAM_SHIFT))

        # Simple gradient descent (Adam is in VQESolver for the full pipeline)
        return params - 0.01 * grad
