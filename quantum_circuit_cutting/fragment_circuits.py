"""Fragment circuit generation from partitioned VQE circuits.

Generates executable sub-circuits for each cluster and channel
configuration, with boundary state-preparation / measurement-basis
operations inserted at cut points.

This module implements static wire-cut channel semantics and optional
gate-level QPD expansion for cross-cluster orbital-rotation terms.
Mid-circuit measurement + classical feedforward is intentionally rejected
at runtime until a backend path is implemented.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorcircuit as tc

from quantum_circuit_cutting.channel_decomposition import (
    Channel,
    ChannelDecomposition,
    PauliBasis,
    PauliRotationQPDDecomposition,
)
from quantum_circuit_cutting.partition import ClusterPartition

logger = logging.getLogger(__name__)

# Type alias: (n_qubits, params, n_electrons, orbital_rotation_angles) -> Circuit
CircuitBuilder = Callable[[int, np.ndarray, int, Optional[np.ndarray]], tc.Circuit]


@dataclass
class FragmentCircuit:
    """A single fragment circuit ready for hardware execution.

    Attributes:
        cluster_idx: Which cluster this fragment belongs to.
        config_idx: Which channel configuration this corresponds to.
        circuit: TensorCircuit Circuit object.
        n_qubits: Number of qubits in this fragment.
        cut_qubit_roles: Dict mapping local qubit index to role
            ("source_measure" or "target_prep") and channel info.
        classical_bit_map: Reserved for future mid-circuit classical
            control support. Kept for API compatibility.
    """

    cluster_idx: int
    config_idx: int
    circuit: tc.Circuit
    n_qubits: int
    cut_qubit_roles: Dict[int, Tuple[str, Channel]]
    classical_bit_map: Dict[int, int] = field(default_factory=dict)


@dataclass
class _OrbitalCrossClusterQPDSpec:
    """QPD spec for one cross-cluster orbital-rotation Pauli term."""

    term_index: int
    qpd: PauliRotationQPDDecomposition
    local_ops_by_cluster: Dict[int, List[Tuple[int, str]]]


class FragmentCircuitGenerator:
    """Generates fragment circuits from a partitioned VQE circuit.

    For each channel configuration (i_1, ..., i_K):
    - Source cluster: append measurement rotation for O_{i_k}
    - Target cluster: prepend state preparation for rho_{i_k}

    Args:
        partition: ClusterPartition from CircuitPartitioner.
        decomposition: ChannelDecomposition for the wire cuts.
        ansatz_params: Current VQE ansatz parameters.
        orbital_rotation_angles: Current orbital rotation angles (or None).
        n_electrons: Number of active electrons for HF reference.
        n_layers: Number of ansatz layers.
        circuit_builder: Optional callable (n_qubits, params, n_electrons) -> Circuit.
            If provided, used instead of the built-in local ansatz.
    """

    def __init__(
        self,
        partition: ClusterPartition,
        decomposition: ChannelDecomposition,
        ansatz_params: np.ndarray,
        orbital_rotation_angles: np.ndarray | None = None,
        n_electrons: int = 2,
        n_layers: int = 2,
        circuit_builder: Optional[CircuitBuilder] = None,
    ):
        self.partition = partition
        self.decomposition = decomposition
        self.ansatz_params = ansatz_params
        self.orbital_rotation_angles = orbital_rotation_angles
        self.n_electrons = n_electrons
        self.n_layers = n_layers
        self.circuit_builder = circuit_builder

        # Build qubit mapping: global -> (cluster_idx, local_idx)
        self._global_to_local: Dict[int, Tuple[int, int]] = {}
        for ci, qubits in enumerate(partition.clusters):
            for local_idx, global_idx in enumerate(qubits):
                self._global_to_local[global_idx] = (ci, local_idx)
        self._orbital_qpd_specs = self._build_orbital_qpd_specs()
        self._orbital_spec_by_term = {
            spec.term_index: spec for spec in self._orbital_qpd_specs
        }
        self._orbital_spec_pos_by_term = {
            spec.term_index: i for i, spec in enumerate(self._orbital_qpd_specs)
        }

    def expand_configurations(
        self, configurations: List[Tuple[Tuple[int, ...], complex]]
    ) -> List[Tuple[Tuple[int, ...], complex]]:
        """Expand base wire-cut configurations with orbital gate-level QPD terms."""
        if not self._orbital_qpd_specs:
            return configurations

        n_wire = self.partition.n_cuts
        expected_len = n_wire + len(self._orbital_qpd_specs)
        if configurations and all(len(idx) == expected_len for idx, _c in configurations):
            # Already expanded.
            return configurations

        expanded: List[Tuple[Tuple[int, ...], complex]] = []
        for wire_indices, wire_coeff in configurations:
            if len(wire_indices) != n_wire:
                raise ValueError(
                    f"Base configuration index length {len(wire_indices)} != n_cuts {n_wire}."
                )
            index_ranges = [range(spec.qpd.n_terms) for spec in self._orbital_qpd_specs]
            for qpd_choices in itertools.product(*index_ranges):
                coeff = complex(wire_coeff)
                for spec, q_idx in zip(self._orbital_qpd_specs, qpd_choices):
                    coeff *= spec.qpd.get_coefficient(q_idx)
                expanded.append((tuple(wire_indices) + tuple(qpd_choices), coeff))
        return expanded

    def generate_all(
        self,
        use_sampling: bool = False,
        n_samples: int = 1000,
        configurations: list | None = None,
        pauli_group: Optional[List[str]] = None,
    ) -> List[List[FragmentCircuit]]:
        """Generate fragment circuits for all channel configurations.

        Args:
            use_sampling: If True, sample configurations instead of
                enumerating all 8^K (necessary for large K).
            n_samples: Number of samples if use_sampling=True.
            configurations: Pre-generated list of (channel_indices, coeff)
                tuples.  If provided, use_sampling/n_samples are ignored.
                This ensures the same config set is used for fragment
                generation and reconstruction.
            pauli_group: Optional list of Pauli strings to use as measurement
                basis for all fragments. If None, default measurement bases
                are used.

        Returns:
            List of lists: outer index = configuration, inner = clusters.
        """
        if configurations is not None:
            configs = configurations
        elif use_sampling:
            configs = self.decomposition.sample_configurations(n_samples)
        else:
            configs = self.decomposition.enumerate_configurations()

        configs = self.expand_configurations(configs)

        all_fragments: List[List[FragmentCircuit]] = []

        for config_idx, (channel_indices, _coeff) in enumerate(configs):
            fragments = self._generate_for_config(config_idx, channel_indices, pauli_group)
            all_fragments.append(fragments)

        logger.info(
            "Generated %d configurations x %d clusters = %d fragments",
            len(configs),
            self.partition.n_clusters,
            len(configs) * self.partition.n_clusters,
        )
        return all_fragments

    def _generate_for_config(
        self, config_idx: int, channel_indices: Tuple[int, ...], pauli_group: Optional[List[str]] = None
    ) -> List[FragmentCircuit]:
        """Generate fragment circuits for one channel configuration.

        Uses static wire-cut boundary semantics:
        - Target-side cut qubits are prepared in the channel state.
        - Local (intra-cluster) ansatz/orbital gates are applied.
        - Source-side cut qubits are rotated into channel measurement basis.
        """
        fragments: List[FragmentCircuit] = []

        # Map each cut edge to its channel
        cut_channels: Dict[Tuple[int, int], Channel] = {}
        n_wire = self.partition.n_cuts
        wire_channel_indices = channel_indices[:n_wire]
        orbital_qpd_choices = channel_indices[n_wire:]
        for k, (q_i, q_j) in enumerate(self.partition.inter_cluster_edges):
            cut_channels[(q_i, q_j)] = self.decomposition.get_channel(
                wire_channel_indices[k]
            )

        # Build per-cluster circuits
        circuits: Dict[int, tc.Circuit] = {}
        cut_roles_map: Dict[int, Dict[int, Tuple[str, Channel]]] = {}
        classical_bit_maps: Dict[int, Dict[int, int]] = {}
        for ci, cluster_qubits in enumerate(self.partition.clusters):
            circuits[ci] = tc.Circuit(len(cluster_qubits))
            cut_roles_map[ci] = {}
            classical_bit_maps[ci] = {}

        for ci, cluster_qubits in enumerate(self.partition.clusters):
            # Prepend target-side channel state preparation.
            for (q_src, q_tgt), channel in cut_channels.items():
                if q_tgt in cluster_qubits:
                    local_idx = cluster_qubits.index(q_tgt)
                    _apply_state_preparation(circuits[ci], local_idx, channel)
                    cut_roles_map[ci][local_idx] = ("target_prep", channel)

            # Local ansatz dynamics.
            if self.circuit_builder is not None:
                self._apply_from_builder(circuits[ci], cluster_qubits)
            else:
                self._apply_local_ansatz(
                    circuits[ci], cluster_qubits, orbital_qpd_choices
                )

            # Append source-side measurement-basis rotations.
            for (q_src, q_tgt), channel in cut_channels.items():
                if q_src in cluster_qubits:
                    local_idx = cluster_qubits.index(q_src)
                    _apply_measurement_rotation(circuits[ci], local_idx, channel)
                    cut_roles_map[ci][local_idx] = ("source_measure", channel)

        # Observable measurement rotations (QWC group)
        for ci, cluster_qubits in enumerate(self.partition.clusters):
            if pauli_group is not None and pauli_group:
                from quantum_circuit_cutting.reconstruction import measurement_basis_key
                general_basis = measurement_basis_key(pauli_group)
                for global_q, op in enumerate(general_basis):
                    if op in ("X", "Y", "Z") and global_q in self._global_to_local:
                        qubit_ci, local_idx = self._global_to_local[global_q]
                        if qubit_ci == ci:
                            if local_idx in cut_roles_map[ci]:
                                continue
                            if op == "X":
                                circuits[ci].h(local_idx)
                            elif op == "Y":
                                circuits[ci].sdg(local_idx)
                                circuits[ci].h(local_idx)

            fragments.append(
                FragmentCircuit(
                    cluster_idx=ci,
                    config_idx=config_idx,
                    circuit=circuits[ci],
                    n_qubits=len(cluster_qubits),
                    cut_qubit_roles=cut_roles_map[ci],
                    classical_bit_map=classical_bit_maps[ci],
                )
            )

        return fragments

    def _build_orbital_qpd_specs(self) -> List[_OrbitalCrossClusterQPDSpec]:
        """Build gate-level QPD specs for non-local orbital-rotation Pauli terms."""
        if self.orbital_rotation_angles is None:
            return []

        from vqe_mcpdft.orbital_rotation import OrbitalRotationCircuit

        n_total = sum(len(c) for c in self.partition.clusters)
        n_spatial = n_total // 2
        orb_rot = OrbitalRotationCircuit(n_spatial_orbitals=n_spatial)
        pauli_terms = orb_rot._get_pauli_terms()
        if len(pauli_terms) != len(self.orbital_rotation_angles):
            raise ValueError(
                f"Orbital rotation angle/term mismatch: "
                f"{len(self.orbital_rotation_angles)} angles vs "
                f"{len(pauli_terms)} terms for {n_spatial} spatial orbitals. "
                f"Ensure update() uses include_zeros=True."
            )

        specs: List[_OrbitalCrossClusterQPDSpec] = []
        for term_index, ((_coeff, ops), angle) in enumerate(
            zip(pauli_terms, self.orbital_rotation_angles)
        ):
            if abs(angle) < 1e-12:
                continue

            local_ops_by_cluster: Dict[int, List[Tuple[int, str]]] = {}
            for q_global, pauli in ops:
                ci, local = self._global_to_local[q_global]
                local_ops_by_cluster.setdefault(ci, []).append((local, pauli))

            touched = set(local_ops_by_cluster.keys())
            if len(touched) == 1:
                continue
            if len(touched) != 2:
                raise RuntimeError(
                    "Cross-cluster orbital-rotation term spans more than two clusters; "
                    "gate-level QPD currently supports exactly two clusters per term."
                )
            specs.append(
                _OrbitalCrossClusterQPDSpec(
                    term_index=term_index,
                    qpd=PauliRotationQPDDecomposition(float(angle)),
                    local_ops_by_cluster=local_ops_by_cluster,
                )
            )
        return specs

    def _apply_from_builder(
        self, circuit: tc.Circuit, cluster_qubits: List[int]
    ) -> None:
        """Apply gates from the custom circuit_builder to a fragment.

        The builder is called with (n_local_qubits, params, n_local_electrons, orbital_angles)
        and must return a tc.Circuit of the correct local size.  We then
        transfer its state into the fragment circuit via a unitary gate.
        """
        assert self.circuit_builder is not None
        n_local = len(cluster_qubits)
        n_local_elec = sum(1 for q in cluster_qubits if q < self.n_electrons)
        built = self.circuit_builder(n_local, self.ansatz_params, n_local_elec, self.orbital_rotation_angles)
        # Transfer the builder's prepared state into the fragment circuit
        # by applying U such that U|current> = |built_state>.
        # Since the fragment circuit may already have state-prep gates
        # (from cut target qubits), we compose by applying the builder's
        # unitary.  For the common case where the circuit is still |0...0>
        # at the non-cut qubits, this is equivalent to just using the
        # builder's state.
        built_state = np.asarray(built.state()).ravel()
        n_states = 2 ** n_local
        # Build unitary: columns are the images of each basis state.
        # Column 0 = built_state (image of |0...0>).
        # Extend to a full unitary via Gram-Schmidt.
        U = np.zeros((n_states, n_states), dtype=complex)
        U[:, 0] = built_state
        for k in range(1, n_states):
            v = np.zeros(n_states, dtype=complex)
            v[k] = 1.0
            for j in range(k):
                v -= np.vdot(U[:, j], v) * U[:, j]
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                U[:, k] = v / norm
            else:
                U[:, k] = 0
        circuit.any(*range(n_local), unitary=U)

    def _apply_local_ansatz(
        self,
        circuit: tc.Circuit,
        cluster_qubits: List[int],
        orbital_qpd_choices: Tuple[int, ...] = (),
    ) -> None:
        """Apply ansatz gates that act only within this cluster."""
        from vqe_mcpdft.orbital_rotation import OrbitalRotationCircuit
        
        cluster_set = set(cluster_qubits)
        n_local = len(cluster_qubits)
        current_cluster_idx = (
            self._global_to_local[cluster_qubits[0]][0] if cluster_qubits else -1
        )

        # HF reference: set occupied qubits
        for i, global_q in enumerate(cluster_qubits):
            if global_q < self.n_electrons:
                circuit.x(i)

        # Particle-conserving gates within cluster
        param_idx = 0
        n_total = sum(len(c) for c in self.partition.clusters)
        for layer in range(self.n_layers):
            offset = layer % 2
            global_q = offset
            while global_q < n_total - 1:
                q0, q1 = global_q, global_q + 1
                if q0 in cluster_set and q1 in cluster_set:
                    local_0 = cluster_qubits.index(q0)
                    local_1 = cluster_qubits.index(q1)
                    if param_idx < len(self.ansatz_params):
                        theta = self.ansatz_params[param_idx]
                        circuit.cnot(local_0, local_1)
                        circuit.ry(local_0, theta=theta)
                        circuit.cnot(local_1, local_0)
                        circuit.ry(local_0, theta=-theta)
                        circuit.cnot(local_0, local_1)
                param_idx += 1
                global_q += 2
        
        # Apply orbital rotation angles if provided
        if self.orbital_rotation_angles is not None and n_local > 0:
            # Apply global orbital rotation terms that act entirely within this fragment
            n_total_qubits = sum(len(c) for c in self.partition.clusters)
            n_global_spatial = n_total_qubits // 2
            orb_rot = OrbitalRotationCircuit(n_spatial_orbitals=n_global_spatial)
            # _get_pauli_terms() returns fixed-length term list (include_zeros=True),
            # matching the fixed-length angle vector from update()
            global_to_local = {q: i for i, q in enumerate(cluster_qubits)}
            pauli_terms = orb_rot._get_pauli_terms()
            
            if len(pauli_terms) != len(self.orbital_rotation_angles):
                raise ValueError(
                    f"Orbital rotation angle/term mismatch: {len(self.orbital_rotation_angles)} "
                    f"angles vs {len(pauli_terms)} terms for {n_global_spatial} spatial orbitals. "
                    f"Ensure update() uses include_zeros=True."
                )
            
            for term_index, ((coeff, ops), angle) in enumerate(
                zip(pauli_terms, self.orbital_rotation_angles)
            ):
                if abs(angle) < 1e-12:
                    continue
                # Check if all qubits in this term are in the fragment
                all_local = True
                local_ops = []
                for q_global, p in ops:
                    if q_global not in global_to_local:
                        all_local = False
                        break
                    local_ops.append((global_to_local[q_global], p))
                if not all_local:
                    # Cross-cluster orbital term: apply gate-level QPD local
                    # operators for this fragment according to configuration.
                    spec = self._orbital_spec_by_term.get(term_index)
                    if spec is None:
                        continue
                    spec_idx = self._orbital_spec_pos_by_term[term_index]
                    if spec_idx >= len(orbital_qpd_choices):
                        raise ValueError(
                            "Orbital QPD configuration index missing for non-local orbital term."
                        )
                    choice = orbital_qpd_choices[spec_idx]
                    apply_before, apply_after = spec.qpd.get_local_application_flags(choice)
                    local_ops = spec.local_ops_by_cluster.get(current_cluster_idx, [])
                    if apply_before:
                        _apply_pauli_ops(circuit, local_ops)
                    if apply_after:
                        _apply_pauli_ops(circuit, local_ops)
                    continue
                
                # Apply this term using local qubit indices
                qubits = [q for q, _ in local_ops]
                pauli_types = [p for _, p in local_ops]
                
                # Apply basis rotations for X/Y terms
                for q, p in zip(qubits, pauli_types):
                    if p == "X":
                        circuit.h(q)
                    elif p == "Y":
                        circuit.sdg(q)
                        circuit.h(q)
                
                # Apply CNOT ladder
                for i in range(len(qubits) - 1):
                    circuit.cx(qubits[i], qubits[i + 1])
                
                # Apply Rz rotation
                circuit.rz(qubits[-1], angle)
                
                # Apply inverse CNOT ladder
                for i in range(len(qubits) - 2, -1, -1):
                    circuit.cx(qubits[i], qubits[i + 1])
                
                # Apply inverse basis rotations
                for q, p in reversed(list(zip(qubits, pauli_types))):
                    if p == "X":
                        circuit.h(q)
                    elif p == "Y":
                        circuit.h(q)
                        circuit.s(q)

    def flatten_circuits(
        self, all_fragments: List[List[FragmentCircuit]]
    ) -> Tuple[List[tc.Circuit], List[Tuple[int, int]]]:
        """Flatten fragment circuits for batch submission to TQP.

        Returns:
            (circuits, metadata) where metadata[i] = (config_idx, cluster_idx).
        """
        circuits: List[tc.Circuit] = []
        metadata: List[Tuple[int, int]] = []
        for fragments in all_fragments:
            for frag in fragments:
                circuits.append(frag.circuit)
                metadata.append((frag.config_idx, frag.cluster_idx))
        return circuits, metadata


def _apply_state_preparation(
    circuit: tc.Circuit,
    qubit: int,
    channel: Channel,
    classical_bit: int | None = None,
) -> None:
    """Prepare the target qubit in channel state rho_i.

    Classical feedforward is not implemented in this pipeline yet.
    """
    if classical_bit is not None:
        raise RuntimeError(
            "Classical feedforward is not implemented in fragment generation. "
            "Use static wire-cut mode without classical_bit."
        )
    if channel.prep_state == "|0>":
        pass  # Already in |0>
    elif channel.prep_state == "|1>":
        circuit.x(qubit)
    elif channel.prep_state == "|+>":
        circuit.h(qubit)
    elif channel.prep_state == "|->":
        circuit.x(qubit)
        circuit.h(qubit)
    elif channel.prep_state == "|+i>":
        circuit.h(qubit)
        circuit.s(qubit)
    elif channel.prep_state == "|-i>":
        circuit.x(qubit)
        circuit.h(qubit)
        circuit.s(qubit)


def _apply_pauli_ops(
    circuit: tc.Circuit,
    local_ops: List[Tuple[int, str]],
) -> None:
    """Apply a local Pauli-string operator to a fragment circuit."""
    for qubit, pauli in local_ops:
        if pauli == "I":
            continue
        if pauli == "X":
            circuit.x(qubit)
        elif pauli == "Y":
            circuit.y(qubit)
        elif pauli == "Z":
            circuit.z(qubit)
        else:
            raise ValueError(f"Unsupported Pauli operator in local ops: {pauli}")


def _apply_measurement_rotation(
    circuit: tc.Circuit,
    qubit: int,
    channel: Channel,
    classical_bit: int | None = None,
) -> int | None:
    """Apply basis rotation for source-side channel observable measurement.

    Mid-circuit measurement/reset is not implemented in this pipeline yet.
    """
    if channel.observable == PauliBasis.I:
        pass  # No rotation needed (identity measurement)
    elif channel.observable == PauliBasis.X:
        circuit.h(qubit)  # X -> Z basis
    elif channel.observable == PauliBasis.Y:
        circuit.sdg(qubit)  # type: ignore[attr-defined]
        circuit.h(qubit)  # Y -> Z basis
    elif channel.observable == PauliBasis.Z:
        pass  # Already in Z basis

    if classical_bit is not None:
        raise RuntimeError(
            "Mid-circuit measurement/reset is not implemented in fragment generation. "
            "Use static wire-cut mode without classical_bit."
        )

    return None
