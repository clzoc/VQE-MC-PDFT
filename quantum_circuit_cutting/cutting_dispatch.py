"""Cutting dispatch: auto-routes VQE calculations through circuit cutting.

Provides a unified entry point that transparently handles both direct
execution (n_qubits <= device_limit) and partitioned execution via
circuit cutting (n_qubits > device_limit).

Execution modes:
- Direct + hardware: non-cut execution on TQP backend
- Cutting + hardware: fragment circuits submitted to TQP backend
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from vqe_mcpdft.tqp_backend import TIANJI_S2_N_QUBITS

logger = logging.getLogger(__name__)


class CuttingDispatcher:
    """Transparently dispatches VQE energy evaluations through cutting when needed.

    If n_qubits <= max_device_qubits, evaluates directly on hardware.
    If n_qubits > max_device_qubits, partitions and uses circuit cutting
    with hardware fragment execution.

    Args:
        max_device_qubits: Physical device qubit limit.
        n_vqe_layers: Ansatz layers for cutting mode.
        use_sampling: Force sampling mode (None = auto).
        n_samples: Samples for importance sampling.
        n_shots: Reserved for legacy simulation utility tests.
        tqp_backend: Optional hardware backend for fragment execution.
    """

    def __init__(
        self,
        max_device_qubits: int = TIANJI_S2_N_QUBITS,
        tqp_backend=None,
        n_vqe_layers: int = 2,
        use_sampling: bool | None = None,
        n_samples: int = 1000,
        n_shots: int = 10000,
        mitigation_pipeline: List[Callable] | None = None,
    ):
        self.max_device_qubits = max_device_qubits
        self.n_vqe_layers = n_vqe_layers
        self.use_sampling = use_sampling
        self.n_samples = n_samples
        self.n_shots = n_shots
        self.tqp_backend = tqp_backend
        self.mitigation_pipeline = mitigation_pipeline or []

    def needs_cutting(self, n_qubits: int) -> bool:
        """Check if a problem requires circuit cutting."""
        return n_qubits > self.max_device_qubits

    def make_energy_fn(
        self,
        n_qubits: int,
        n_electrons: int,
        hamiltonian: Dict[str, float],
        orbital_symmetries: Dict[int, str] | None = None,
    ) -> Callable[[np.ndarray, Optional[np.ndarray]], float]:
        """Create an energy function that auto-dispatches through cutting."""
        if not self.needs_cutting(n_qubits):
            return self._make_direct_energy_fn(n_qubits, n_electrons, hamiltonian)
        return self._make_cutting_energy_fn(
            n_qubits, n_electrons, hamiltonian, orbital_symmetries
        )

    def make_energy_fn_factory(
        self,
        n_qubits: int,
        n_electrons: int,
        hamiltonian: Dict[str, float],
        orbital_symmetries: Dict[int, str] | None = None,
    ) -> Callable:
        """Create an energy_fn_factory compatible with SelfConsistentVQEMCPDFT.

        Returns: energy_fn_factory(orb_angles) -> energy_fn(params)
        """
        if not self.needs_cutting(n_qubits):
            direct_energy_fn = self._make_direct_energy_fn(
                n_qubits, n_electrons, hamiltonian
            )

            def factory_direct(orb_angles):
                def energy_fn(params):
                    return direct_energy_fn(params, orb_angles)
                return energy_fn

            return factory_direct

        # Cutting path
        ctx = self._build_cutting_context(
            n_qubits, n_electrons, hamiltonian, orbital_symmetries
        )

        def factory_cutting(orb_angles):
            def energy_fn(params):
                return self._evaluate_cutting(ctx, params, orb_angles)
            return energy_fn

        return factory_cutting

    def make_pauli_expval_fn(
        self,
        n_qubits: int,
        n_electrons: int,
        hamiltonian: Dict[str, float],
        orbital_symmetries: Dict[int, str] | None = None,
    ) -> Callable | None:
        """Create a pauli_expval_fn for SelfConsistentVQEMCPDFT.

        Returns a callable:
            (params, orb_angles, pauli_labels) -> Dict[str, float]
        For n_qubits <= device limit, evaluates Pauli expectations from
        direct hardware measurements on the full circuit. For n_qubits >
        device limit, evaluates via cutting reconstruction.
        """
        if not self.needs_cutting(n_qubits):
            from vqe_mcpdft.ansatz import CASCIAnsatz
            from .reconstruction import group_commuting_paulis, measurement_basis_key

            ansatz = CASCIAnsatz(n_qubits, n_electrons, self.n_vqe_layers)

            def pauli_expval_fn_direct(
                params: np.ndarray,
                orb_angles: np.ndarray | None,
                pauli_labels: List[str],
            ) -> Dict[str, float]:
                if self.tqp_backend is None:
                    raise RuntimeError(
                        "TQP backend is required for execution. "
                        "Simulator fallback is not supported per hardware-only policy."
                    )

                result: Dict[str, float] = {}
                non_identity = []
                for label in pauli_labels:
                    if all(c == "I" for c in label):
                        result[label] = 1.0
                    else:
                        non_identity.append(label)

                if not non_identity:
                    return result

                circuit = ansatz.build_circuit(params, orb_angles)
                qwc_groups = group_commuting_paulis(non_identity)
                for group in qwc_groups:
                    meas_circuit = circuit.copy()
                    basis = measurement_basis_key(group)
                    for i, op in enumerate(basis):
                        if op == "X":
                            meas_circuit.h(i)
                        elif op == "Y":
                            meas_circuit.sdg(i)
                            meas_circuit.h(i)
                    counts = self.tqp_backend.submit_circuit(meas_circuit)
                    total = sum(counts.values())
                    for label in group:
                        parity = 0.0
                        for bitstring, count in counts.items():
                            sign = 1
                            for i, op in enumerate(label):
                                if op != "I" and bitstring[i] == "1":
                                    sign *= -1
                            parity += sign * count
                        result[label] = parity / total
                return result

            return pauli_expval_fn_direct

        ctx = self._build_cutting_context(
            n_qubits, n_electrons, hamiltonian, orbital_symmetries
        )

        def pauli_expval_fn(
            params: np.ndarray,
            orb_angles: np.ndarray | None,
            pauli_labels: List[str],
        ) -> Dict[str, float]:
            from .reconstruction import group_commuting_paulis
            
            result: Dict[str, float] = {}
            # Handle identity terms separately
            remaining_labels = []
            for label in pauli_labels:
                if all(c == "I" for c in label):
                    result[label] = 1.0
                else:
                    remaining_labels.append(label)
            
            if not remaining_labels:
                return result
            
            # Split into QWC groups, run each group separately
            qwc_groups = group_commuting_paulis(remaining_labels)
            configs = None
            results_by_group: Dict[tuple, Dict[tuple, Dict[str, int]]] = {}
            
            for group in qwc_groups:
                group_key = tuple(sorted(group))
                fragment_results, group_configs = self._run_fragments(
                    ctx, params, orb_angles, pauli_group=group
                )
                if configs is None:
                    configs = group_configs
                results_by_group[group_key] = fragment_results
            
            # Reconstruct each Pauli from its corresponding group's results
            for label in remaining_labels:
                # Find which group contains this Pauli
                for group in qwc_groups:
                    if label in group:
                        group_key = tuple(sorted(group))
                        break
                else:
                    raise ValueError(f"Pauli {label} not found in any QWC group")
                
                result[label] = ctx["reconstructor"].reconstruct_expectation(
                    results_by_group[group_key], label, configs
                )
            
            return result

        return pauli_expval_fn

    # --- Internal helpers ---

    def _build_cutting_context(
        self,
        n_qubits: int,
        n_electrons: int,
        hamiltonian: Dict[str, float],
        orbital_symmetries: Dict[int, str] | None,
    ) -> Dict:
        """Build reusable cutting infrastructure (partition, decomposition, etc.).

        Uses static wire-cut coefficients from ``ChannelDecomposition`` and
        applies gate-level QPD expansion for cross-cluster orbital-rotation
        terms during fragment generation.
        """
        from quantum_circuit_cutting.channel_decomposition import ChannelDecomposition
        from quantum_circuit_cutting.overhead import select_cut_strategy
        from quantum_circuit_cutting.partition import CircuitPartitioner
        from quantum_circuit_cutting.reconstruction import CuttingReconstructor
        from vqe_mcpdft.ansatz import CASCIAnsatz

        # Screen negligible Hamiltonian terms
        _SCREEN_THRESHOLD = 1e-10
        n_original = len(hamiltonian)
        hamiltonian = {ps: c for ps, c in hamiltonian.items() if abs(c) > _SCREEN_THRESHOLD}
        if len(hamiltonian) < n_original:
            logger.info("Hamiltonian screening: %d -> %d terms", n_original, len(hamiltonian))

        ansatz = CASCIAnsatz(n_qubits, n_electrons, self.n_vqe_layers)
        partitioner = CircuitPartitioner(n_qubits, self.max_device_qubits)
        entangling_gates = ansatz.get_qubit_pairs()
        partition = partitioner.partition(entangling_gates, orbital_symmetries)

        decomposition = ChannelDecomposition(
            partition.n_cuts
        )
        reconstructor = CuttingReconstructor(partition, decomposition)

        if self.use_sampling is None:
            strategy = select_cut_strategy(
                n_qubits, self.max_device_qubits, entangling_gates
            )
            use_sampling = bool(strategy["use_sampling"])
            n_samples = int(strategy["n_samples"])
        else:
            use_sampling = self.use_sampling
            n_samples = self.n_samples

        logger.info(
            "CuttingDispatcher: %d qubits -> %d clusters, %d cuts, "
            "kappa=%.1f, sampling=%s",
            n_qubits, partition.n_clusters, partition.n_cuts,
            partition.overhead_kappa, use_sampling
        )

        return {
            "ansatz": ansatz,
            "partition": partition,
            "decomposition": decomposition,
            "reconstructor": reconstructor,
            "use_sampling": use_sampling,
            "n_samples": n_samples,
            "n_qubits": n_qubits,
            "n_electrons": n_electrons,
            "hamiltonian": hamiltonian,
        }

    def _run_fragments(
        self,
        ctx: Dict,
        params: np.ndarray,
        orb_angles: np.ndarray | None,
        pauli_group: Optional[List[str]] = None,
    ) -> Tuple[Dict[Tuple[int, int], Dict[str, int]], List[Tuple[Tuple[int, ...], complex]]]:
        """Generate, execute, and return fragment results + configs.

        A single config set is generated and used for both fragment
        generation and reconstruction (fixes config mismatch bug).
        """
        from quantum_circuit_cutting.fragment_circuits import FragmentCircuitGenerator
        partition = ctx["partition"]
        decomposition = ctx["decomposition"]
        use_sampling = ctx["use_sampling"]
        n_samples = ctx["n_samples"]

        # Generate ONE config set used everywhere
        if use_sampling:
            configs = decomposition.sample_configurations(
                n_samples, rng=np.random.default_rng(42)
            )
        else:
            configs = decomposition.enumerate_configurations()

        # Group Paulis into QWC groups for measurement basis optimization

        # if pauli_group is not None and len(pauli_group) > 1:
        #     qwc_groups = group_commuting_paulis(pauli_group)
        #     # Use first group's representative for measurement basis
        #     pauli_group = qwc_groups[0]

        generator = FragmentCircuitGenerator(
            partition=partition,
            decomposition=decomposition,
            ansatz_params=params,
            orbital_rotation_angles=orb_angles,
            n_electrons=ctx["n_electrons"],
            n_layers=self.n_vqe_layers,
        )
        effective_configs = generator.expand_configurations(configs)
        all_fragments = generator.generate_all(
            configurations=effective_configs, pauli_group=pauli_group
        )

        # Execute fragments
        fragment_results: Dict[Tuple[int, int], Dict[str, int]] = {}
        for frags in all_fragments:
            for f in frags:
                if self.tqp_backend is not None:
                    counts = self.tqp_backend.submit_circuit(f.circuit)  # type: ignore
                else:
                    raise RuntimeError("TQP backend is required for execution. Simulator fallback is not supported per hardware-only policy.")
                # Apply error mitigation pipeline
                for mit_fn in self.mitigation_pipeline:
                    counts = mit_fn(counts, f.circuit, f.n_qubits)
                fragment_results[(f.config_idx, f.cluster_idx)] = counts

        return fragment_results, effective_configs

    def _evaluate_cutting(
        self, ctx: Dict, params: np.ndarray, orb_angles: np.ndarray | None
    ) -> float:
        """Evaluate energy via cutting with consistent configs.

        Uses static wire-cut coefficients from ``ChannelDecomposition`` and
        gate-level QPD expansion for cross-cluster orbital terms.
        """
        from .reconstruction import group_commuting_paulis

        hamiltonian = ctx["hamiltonian"]
        reconstructor = ctx["reconstructor"]
        # NOTE: Cross-cluster orbital terms are handled via gate-level QPD
        # expansion in FragmentCircuitGenerator. Static wire-cut coefficients
        # still govern cut-edge boundary channels.

        # Handle identity terms first
        energy = 0.0
        non_identity_ham = {}
        for label, coeff in hamiltonian.items():
            if all(c == "I" for c in label):
                energy += coeff
            else:
                non_identity_ham[label] = coeff

        if not non_identity_ham:
            return energy

        # Split into QWC groups, run each group separately
        pauli_labels = list(non_identity_ham.keys())
        qwc_groups = group_commuting_paulis(pauli_labels)
        configs = None
        results_by_group: Dict[tuple, Dict[tuple, Dict[str, int]]] = {}

        for group in qwc_groups:
            group_key = tuple(sorted(group))
            fragment_results, group_configs = self._run_fragments(
                ctx, params, orb_angles, pauli_group=group
            )
            if configs is None:
                configs = group_configs
            results_by_group[group_key] = fragment_results

        # Reconstruct each group's terms using only its own measurement results
        for group in qwc_groups:
            group_key = tuple(sorted(group))
            group_ham = {label: non_identity_ham[label] for label in group}
            group_energy = reconstructor.reconstruct_hamiltonian_energy(
                results_by_group[group_key], group_ham, configs
            )
            energy += group_energy

        return energy

    def _make_direct_energy_fn(
        self,
        n_qubits: int,
        n_electrons: int,
        hamiltonian: Dict[str, float],
    ) -> Callable[[np.ndarray, Optional[np.ndarray]], float]:
        """Direct energy evaluation (no cutting) on TQP hardware."""
        from vqe_mcpdft.ansatz import CASCIAnsatz

        ansatz = CASCIAnsatz(n_qubits, n_electrons, self.n_vqe_layers)

        def energy_fn(
            params: np.ndarray,
            orb_angles: Optional[np.ndarray] = None,
        ) -> float:
            circuit = ansatz.build_circuit(params, orb_angles)
            
            if self.tqp_backend is None:
                raise RuntimeError(
                    "TQP backend is required for execution. "
                    "Simulator fallback is not supported per hardware-only policy."
                )

            # Use TQP hardware for direct execution
            from .reconstruction import group_commuting_paulis

            expval = 0.0
            # First handle identity terms
            for pauli_str, coeff in hamiltonian.items():
                if all(c == "I" for c in pauli_str):
                    expval += coeff

            # Group remaining Paulis into QWC groups
            non_id_paulis = {
                p: c for p, c in hamiltonian.items() if not all(c_i == "I" for c_i in p)
            }
            if non_id_paulis:
                pauli_list = list(non_id_paulis.keys())
                qwc_groups = group_commuting_paulis(pauli_list)

                for group in qwc_groups:
                    # Build measurement circuit for this group
                    meas_circuit = circuit.copy()
                    # Use the most general observable across the group
                    from .reconstruction import measurement_basis_key
                    general_basis = measurement_basis_key(group)
                    for i, op in enumerate(general_basis):
                        if op == "X":
                            meas_circuit.h(i)
                        elif op == "Y":
                            meas_circuit.sdg(i)
                            meas_circuit.h(i)
                    # TQP backend automatically adds Z-basis measurements

                    # Run circuit
                    counts = self.tqp_backend.submit_circuit(meas_circuit)
                    total = sum(counts.values())

                    # Compute expectation for all Paulis in this group
                    for pauli_str in group:
                        coeff = non_id_paulis[pauli_str]
                        parity = 0.0
                        for bitstring, count in counts.items():
                            sign = 1
                            for i, op in enumerate(pauli_str):
                                if op != "I" and bitstring[i] == "1":
                                    sign *= -1
                            parity += sign * count
                        expval += coeff * (parity / total)
            return expval

        return energy_fn

    def _make_cutting_energy_fn(
        self,
        n_qubits: int,
        n_electrons: int,
        hamiltonian: Dict[str, float],
        orbital_symmetries: Dict[int, str] | None = None,
    ) -> Callable[[np.ndarray, Optional[np.ndarray]], float]:
        """Cutting-based energy evaluation with hardware fragment counts."""
        ctx = self._build_cutting_context(
            n_qubits, n_electrons, hamiltonian, orbital_symmetries
        )

        def energy_fn(
            params: np.ndarray,
            orb_angles: Optional[np.ndarray] = None,
        ) -> float:
            return self._evaluate_cutting(ctx, params, orb_angles)

        return energy_fn
