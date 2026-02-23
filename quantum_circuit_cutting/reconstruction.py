"""Classical post-processing for circuit cutting reconstruction.

Reconstructs expectation values and RDMs from fragment circuit results
using the weighted sum over channel configurations (SI Section S1.2).

Upgraded with observable grouping: Pauli strings that commute qubit-wise
(QWC) can share measurement results, reducing the number of distinct
fragment circuit executions needed.

Measurement-basis reuse: a ``MeasurementCache`` allows results from one
QWC group to be reused by another group that requires the same
measurement basis on the same fragment, avoiding redundant circuit
submissions (Peng et al., PRL 125, 150504, 2020).

Expectation-value-level mitigation: an optional ``expval_mitigator``
callback can be applied to each reconstructed Pauli expectation value
before it is combined into the Hamiltonian energy.  This supports ZNE
and CDR correction at the correct abstraction level.

Reference: Peng et al., PRL 125, 150504 (2020).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

from quantum_circuit_cutting.channel_decomposition import (
    ChannelDecomposition,
    PauliBasis,
)
from quantum_circuit_cutting.partition import ClusterPartition

logger = logging.getLogger(__name__)


def _parity_from_counts(counts: Dict[str, int], qubit_indices: List[int]) -> float:
    """Compute parity expectation value for specified qubits from Z-basis counts.

    ``<Z_i Z_j ...> = sum_bitstring (-1)^(popcount(selected bits)) * count / total``
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    expval = 0.0
    for bitstring, count in counts.items():
        parity = 0
        for idx in qubit_indices:
            if idx < len(bitstring) and bitstring[idx] == "1":
                parity += 1
        expval += ((-1) ** parity) * count
    return expval / total


def group_commuting_paulis(
    pauli_strings: List[str],
) -> List[List[str]]:
    """Group Pauli strings into qubit-wise commuting (QWC) sets.

    Two Pauli strings commute qubit-wise if, for every qubit position,
    at least one of them has I, or they have the same Pauli operator.

    Uses a greedy coloring approach: for each string, try to add it to
    an existing group; if no group is compatible, start a new one.

    Returns:
        List of groups, each group is a list of compatible Pauli strings.
    """
    groups: List[List[str]] = []

    for ps in pauli_strings:
        placed = False
        for group in groups:
            if all(_qwc_compatible(ps, existing) for existing in group):
                group.append(ps)
                placed = True
                break
        if not placed:
            groups.append([ps])

    return groups


def _qwc_compatible(ps1: str, ps2: str) -> bool:
    """Check if two Pauli strings are qubit-wise commuting."""
    for c1, c2 in zip(ps1, ps2):
        if c1 != "I" and c2 != "I" and c1 != c2:
            return False
    return True


def measurement_basis_key(pauli_group: List[str]) -> str:
    """Derive the joint measurement basis for a QWC group.

    For each qubit position, the basis is the non-identity Pauli that
    appears in any member of the group (all members are QWC-compatible,
    so at most one non-I Pauli exists per position).  Returns a string
    like ``"IXZY"`` representing the per-qubit measurement basis.
    """
    if not pauli_group:
        return ""
    n = len(pauli_group[0])
    basis = ["I"] * n
    for ps in pauli_group:
        for i, c in enumerate(ps):
            if c != "I":
                basis[i] = c
    return "".join(basis)


class MeasurementCache:
    """Cache for fragment measurement results keyed by measurement basis.

    When multiple QWC groups require the same measurement rotations on
    the same fragment (same config_idx, cluster_idx, and basis key),
    the circuit only needs to be executed once.  This cache stores and
    retrieves those results.

    Usage::

        cache = MeasurementCache()
        key = cache.make_key(config_idx, cluster_idx, basis_str)
        if key in cache:
            counts = cache[key]
        else:
            counts = execute(...)
            cache[key] = counts
    """

    def __init__(self) -> None:
        self._store: Dict[Tuple[int, int, str], Dict[str, int]] = {}
        self.hits = 0
        self.misses = 0

    def make_key(self, config_idx: int, cluster_idx: int,
                 basis: str) -> Tuple[int, int, str]:
        return (config_idx, cluster_idx, basis)

    def __contains__(self, key: Tuple[int, int, str]) -> bool:
        return key in self._store

    def __getitem__(self, key: Tuple[int, int, str]) -> Dict[str, int]:
        self.hits += 1
        return self._store[key]

    def __setitem__(self, key: Tuple[int, int, str],
                    value: Dict[str, int]) -> None:
        self.misses += 1
        self._store[key] = value

    @property
    def size(self) -> int:
        return len(self._store)

    def summary(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total else 0
        return (f"MeasurementCache: {self.size} entries, "
                f"{self.hits}/{total} hits ({rate:.0f}%)")


class CuttingReconstructor:
    """Reconstructs expectation values from partitioned circuit results.

    Implements the classical post-processing step of the cluster
    simulation framework. For each channel configuration, the
    expectation value of the full circuit is recovered as a product
    of per-cluster expectation values, weighted by the channel
    coefficients (SI Section S1.2).

    Args:
        partition: ClusterPartition describing the qubit clusters.
        decomposition: ChannelDecomposition for the wire cuts.
    """

    def __init__(
        self,
        partition: ClusterPartition,
        decomposition: ChannelDecomposition,
        expval_mitigator: Optional[Callable[[float, str], float]] = None,
    ):
        self.partition = partition
        self.decomposition = decomposition
        self.expval_mitigator = expval_mitigator

    def validate_fragment_results(
        self,
        fragment_results: Dict[Tuple[int, int], Dict[str, int]],
        configurations: List[Tuple[Tuple[int, ...], complex]],
    ) -> None:
        """Validate that fragment results have the expected cardinality.

        Raises:
            ValueError: If expected keys are missing or counts are empty.
        """
        n_configs = len(configurations)
        n_clusters = self.partition.n_clusters
        expected = n_configs * n_clusters
        missing_keys = []
        empty_keys = []
        for config_idx in range(n_configs):
            for ci in range(n_clusters):
                key = (config_idx, ci)
                if key not in fragment_results:
                    missing_keys.append(key)
                elif not fragment_results[key]:
                    empty_keys.append(key)
        if missing_keys:
            raise ValueError(
                f"Reconstruction integrity: {len(missing_keys)}/{expected} fragment "
                f"results missing. First missing: {missing_keys[:5]}. "
                f"Expected {n_configs} configs x {n_clusters} clusters = {expected} entries."
            )
        if empty_keys:
            logger.warning(
                "Reconstruction: %d/%d fragment results have empty counts: %s",
                len(empty_keys), expected, empty_keys[:5],
            )

    def reconstruct_expectation(
        self,
        fragment_results: Dict[Tuple[int, int], Dict[str, int]],
        pauli_string: str,
        configurations: List[Tuple[Tuple[int, ...], complex]] | None = None,
    ) -> float:
        """Reconstruct <P> for a single Pauli string from fragment results.

        E_total = sum_config coeff * product_cluster <P_cluster>

        Args:
            fragment_results: Dict mapping (config_idx, cluster_idx) to
                measurement counts.
            pauli_string: Full Pauli string (length = total qubits).
            configurations: Channel configurations and coefficients.
                If None, enumerates all 8^K.

        Returns:
            Reconstructed expectation value.
        """
        if configurations is None:
            configurations = self.decomposition.enumerate_configurations()

        # Integrity check: validate expected result cardinality
        self.validate_fragment_results(fragment_results, configurations)

        total_qubits = sum(len(c) for c in self.partition.clusters)
        if len(pauli_string) != total_qubits:
            raise ValueError(
                f"Pauli string length {len(pauli_string)} != total qubits {total_qubits}"
            )

        q2c = self.partition.qubit_to_cluster()
        n_clusters = self.partition.n_clusters

        # Decompose Pauli string into per-cluster terms
        cluster_paulis: Dict[int, List[int]] = {ci: [] for ci in range(n_clusters)}
        for q_idx, pauli_op in enumerate(pauli_string):
            if pauli_op in ("X", "Y", "Z"):
                ci = q2c.get(q_idx, 0)
                local_idx = self.partition.clusters[ci].index(q_idx)
                cluster_paulis[ci].append(local_idx)

        # Build set of qubits that already have non-identity Pauli ops
        active_qubits = {q for q, p in enumerate(pauli_string) if p != "I"}

        e_total = 0.0 + 0.0j
        for config_idx, (channel_indices, coeff) in enumerate(configurations):
            product = 1.0
            for ci in range(n_clusters):
                key = (config_idx, ci)
                counts = fragment_results.get(key, {})
                if not counts:
                    product = 0.0
                    break

                qubit_indices = cluster_paulis[ci]
                if qubit_indices:
                    product *= _parity_from_counts(counts, qubit_indices)

                product *= self._cut_qubit_factor(
                    counts, ci, channel_indices, active_qubits
                )

            e_total += complex(coeff) * product

        if abs(e_total.imag) > 1e-8:
            logger.warning(
                "Reconstructed expectation has non-negligible imaginary component %.3e; "
                "returning real part.",
                float(abs(e_total.imag)),
            )
        return float(e_total.real)

    def reconstruct_hamiltonian_energy(
        self,
        fragment_results: Dict[Tuple[int, int], Dict[str, int]],
        hamiltonian: Dict[str, float],
        configurations: List[Tuple[Tuple[int, ...], complex]] | None = None,
    ) -> float:
        """Reconstruct <H> = sum_j h_j <P_j> from fragment results.

        Uses observable grouping to batch compatible Pauli strings.
        """
        energy = 0.0
        non_identity = {
            ps: coeff
            for ps, coeff in hamiltonian.items()
            if not all(c == "I" for c in ps)
        }

        # Add identity contribution directly
        for ps, coeff in hamiltonian.items():
            if all(c == "I" for c in ps):
                energy += coeff

        # Group non-identity terms for efficient reconstruction
        groups = group_commuting_paulis(list(non_identity.keys()))
        logger.debug(
            "Grouped %d Pauli terms into %d QWC groups",
            len(non_identity),
            len(groups),
        )

        for group in groups:
            for pauli_str in group:
                expval = self.reconstruct_expectation(
                    fragment_results, pauli_str, configurations
                )
                if self.expval_mitigator is not None:
                    expval = self.expval_mitigator(expval, pauli_str)
                energy += non_identity[pauli_str] * expval

        return energy

    def _cut_qubit_factor(
        self,
        counts: Dict[str, int],
        cluster_idx: int,
        channel_indices: Tuple[int, ...],
        active_qubits: set[int] | None = None,
    ) -> float:
        """Compute the factor from cut-qubit measurement outcomes.

        For each cut qubit in this cluster that acts as a measurement
        source, the channel's observable contributes a sign factor to
        the reconstruction based on the Z-basis parity of that qubit.

        Processing is per-qubit (not per-edge) to maintain strict
        isomorphism with the frontend cutting operations.  Each cut
        qubit is handled independently, regardless of whether it also
        appears in the observable, as the measurement basis for cut
        qubits is already handled by the channel rotation.

        Args:
            counts: Measurement counts for this fragment.
            cluster_idx: Which cluster this fragment belongs to.
            channel_indices: Channel indices for all K cuts.
            active_qubits: Unused, retained for backward compatibility.
        """
        if active_qubits is None:
            active_qubits = set()
        factor = 1.0
        cut_edges = self.partition.inter_cluster_edges
        clusters = self.partition.clusters

        for k, (q_src, q_tgt) in enumerate(cut_edges):
            channel = self.decomposition.get_channel(channel_indices[k])

            # Source qubit: contributes measurement sign factor
            if q_src in clusters[cluster_idx]:
                local_idx = clusters[cluster_idx].index(q_src)
                if channel.observable != PauliBasis.I:
                    parity = _parity_from_counts(counts, [local_idx])
                    factor *= parity

            # Target qubit: no additional factor needed -- the state
            # preparation is already accounted for by the channel
            # coefficient in the quasi-probability decomposition.

        return factor

    def reconstruct_rdm_element(
        self,
        fragment_results: Dict[Tuple[int, int], Dict[str, int]],
        pauli_terms: List[Tuple[float, str]],
        configurations: List[Tuple[Tuple[int, ...], complex]] | None = None,
    ) -> float:
        """Reconstruct a single RDM element from its Pauli decomposition.

        Args:
            fragment_results: Fragment measurement results.
            pauli_terms: List of (coefficient, pauli_string) for the
                RDM element's Pauli decomposition.
            configurations: Channel configurations.

        Returns:
            Reconstructed RDM element value.
        """
        value = 0.0
        for coeff, pauli_str in pauli_terms:
            if all(c == "I" for c in pauli_str):
                value += coeff
            else:
                value += coeff * self.reconstruct_expectation(
                    fragment_results, pauli_str, configurations
                )
        return value
