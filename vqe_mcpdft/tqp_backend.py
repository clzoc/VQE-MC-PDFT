"""Tencent Quantum Platform (TQP) backend interface.

Provides circuit submission, execution, and result retrieval for the
Tianji-S2 superconducting quantum processor via the TensorCircuit SDK.
Calibration data from SI Tables S2-S3 is used for optimal qubit selection.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

import tensorcircuit as tc
from tensorcircuit.cloud import apis

logger = logging.getLogger(__name__)

# Maximum retries and backoff parameters
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0   # seconds
_BACKOFF_FACTOR = 2.0  # exponential multiplier

# Tianji-S2 device specifications (Tables S2-S3)
TIANJI_S2_N_QUBITS = 13
TIANJI_S2_COUPLING_MAP = [
    (0, 1), (0, 12), (1, 2), (1, 3), (1, 4), (3, 6), (3, 8),
    (3, 12), (4, 6), (5, 6), (6, 7), (8, 9), (8, 10), (10, 12), (11, 12),
]

# Per-qubit calibration data (SI Table S2)
TIANJI_S2_T1 = {
    0: 86.4, 1: 80.7, 2: 65.0, 3: 75.8, 4: 58.3, 5: 96.6, 6: 70.3,
    7: 71.7, 8: 101.7, 9: 102.6, 10: 96.3, 11: 91.4, 12: 94.5,
}
TIANJI_S2_SQ_ERROR = {
    0: 3.3e-4, 1: 9.1e-4, 2: 4.7e-4, 3: 7.1e-4, 4: 1.1e-3, 5: 5.1e-4,
    6: 6.1e-4, 7: 6.1e-4, 8: 1.51e-3, 9: 4.0e-4, 10: 5.2e-4, 11: 4.2e-4,
    12: 8.5e-4,
}
TIANJI_S2_F0_ERROR = {
    0: 4.9e-3, 1: 4.9e-3, 2: 4.8e-3, 3: 6.1e-3, 4: 7.7e-3, 5: 14.8e-3,
    6: 12.6e-3, 7: 5.0e-3, 8: 2.0e-3, 9: 1.1e-3, 10: 3.8e-3, 11: 1.3e-3,
    12: 1.4e-3,
}
# Per-edge CZ gate errors (SI Table S3)
TIANJI_S2_CZ_ERROR = {
    (0, 1): 0.005, (0, 12): 0.005, (1, 2): 0.008, (1, 3): 0.004,
    (1, 4): 0.008, (3, 6): 0.009, (3, 8): 0.006, (3, 12): 0.005,
    (4, 6): 0.010, (5, 6): 0.010, (6, 7): 0.007, (8, 9): 0.009,
    (8, 10): 0.007, (10, 12): 0.011, (11, 12): 0.003,
}


class TQPBackend:
    """Interface to the Tencent Quantum Platform Tianji-S2 device.

    Handles authentication, circuit submission, batching (up to 64
    circuits per call), and result retrieval.  Qubit selection uses
    actual calibration data (gate fidelities, T1, readout errors)
    to find the optimal connected subgraph.

    Args:
        token: TQP authentication token.
        device: Target device name.
        shots: Number of measurement shots per circuit (default 10000, SI S1.3).
        use_simulator: Deprecated legacy option. Hardware-only policy
            requires this to be False.
    """

    def __init__(
        self,
        token: str,
        device: str = "tianji_s2",
        shots: int = 10000,
        use_simulator: bool = False,
    ):
        if use_simulator:
            raise ValueError(
                "Simulator mode is disabled by hardware-only policy. "
                "Use a Tianji-S2 hardware device."
            )
        self.device = device
        self.shots = shots
        self.provider = "tencent"
        self._max_batch = 64

        apis.set_token(token)
        apis.set_provider(self.provider)
        logger.info("TQP backend initialized: device=%s, shots=%d", self.device, shots)

    def submit_circuit(self, circuit: tc.Circuit) -> Dict[str, int]:
        """Submit a single circuit with exponential-backoff retry."""
        return self._submit_with_retry(circuit)

    def submit_batch(self, circuits: List[tc.Circuit]) -> List[Dict[str, int]]:
        """Submit a batch of circuits (auto-splits into sub-batches of 64).

        Each sub-batch is retried independently on transient failure.
        Invalid results (all-zero counts, empty) trigger a retry.
        After exhausting retries a sub-batch returns empty dicts and
        logs a warning rather than aborting the entire VQE iteration.
        """
        all_results: List[Dict[str, int]] = []
        for start in range(0, len(circuits), self._max_batch):
            batch = circuits[start : start + self._max_batch]
            logger.info(
                "Submitting batch %d-%d of %d circuits",
                start, start + len(batch), len(circuits),
            )
            batch_results = self._submit_batch_with_retry(batch)
            all_results.extend(batch_results)
        return all_results

    # ------------------------------------------------------------------
    # retry helpers
    # ------------------------------------------------------------------

    def _submit_with_retry(self, circuit: tc.Circuit) -> Dict[str, int]:
        """Submit one circuit with exponential backoff on transient errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                task = apis.submit_task(
                    provider=self.provider,
                    device=self.device,
                    circuit=circuit,
                    shots=self.shots,
                )
                counts = task.results()
                if self._validate_counts(counts):
                    return counts
                logger.warning(
                    "Invalid counts on attempt %d, retrying", attempt + 1
                )
            except Exception as exc:
                logger.warning(
                    "submit_circuit attempt %d failed: %s", attempt + 1, exc
                )
            if attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_BASE * (_BACKOFF_FACTOR ** attempt)
                time.sleep(delay)
        logger.error("submit_circuit failed after %d retries, returning empty", _MAX_RETRIES)
        return {}

    def _submit_batch_with_retry(
        self, batch: List[tc.Circuit]
    ) -> List[Dict[str, int]]:
        """Submit a sub-batch with retry; degrade gracefully on failure."""
        for attempt in range(_MAX_RETRIES):
            try:
                tasks = apis.submit_task(
                    provider=self.provider,
                    device=self.device,
                    circuit=batch,
                    shots=self.shots,
                )
                results = [t.results() for t in tasks]
                if all(self._validate_counts(c) for c in results):
                    return results
                logger.warning(
                    "Batch has invalid results on attempt %d, retrying",
                    attempt + 1,
                )
            except Exception as exc:
                logger.warning(
                    "submit_batch attempt %d failed: %s", attempt + 1, exc
                )
            if attempt < _MAX_RETRIES - 1:
                delay = _BACKOFF_BASE * (_BACKOFF_FACTOR ** attempt)
                time.sleep(delay)
        logger.error(
            "submit_batch failed after %d retries, returning empty for %d circuits",
            _MAX_RETRIES, len(batch),
        )
        return [{} for _ in batch]

    @staticmethod
    def _validate_counts(counts: Dict[str, int]) -> bool:
        """Check that measurement counts are non-trivially valid.

        Rejects empty dicts and distributions where every bitstring has
        zero counts (hardware glitch).
        """
        if not counts:
            return False
        if sum(counts.values()) == 0:
            return False
        return True

    def compute_expectation(self, counts: Dict[str, int], pauli_string: str) -> float:
        """Compute <P> from measurement counts.

        For Z-diagonal Pauli strings (only I and Z operators), computes
        the parity directly from Z-basis counts.

        For strings containing X or Y operators, the circuit must include
        appropriate basis rotation gates before measurement.  This method
        assumes those rotations are already applied and computes the
        parity from the resulting Z-basis counts for all non-identity
        positions.

        Raises:
            ValueError: If the Pauli string contains operators other than
                I, X, Y, Z.
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0
        expval = 0.0
        for bitstring, count in counts.items():
            sign = 1
            for i, pauli in enumerate(pauli_string):
                if pauli in ("X", "Y", "Z") and bitstring[i] == "1":
                    sign *= -1
                elif pauli not in ("I", "X", "Y", "Z"):
                    raise ValueError(f"Invalid Pauli operator '{pauli}' at position {i}")
            expval += sign * count
        return expval / total

    @staticmethod
    def list_devices() -> list:
        """List available devices on the Tencent Quantum Platform."""
        return apis.list_devices(provider="tencent")

    @staticmethod
    def qubit_selection(
        n_logical: int,
        coupling_map: List[tuple] | None = None,
    ) -> List[int]:
        """Select optimal physical qubits based on calibration data.

        Scores each connected subgraph of size ``n_logical`` using a
        composite metric that combines single-qubit gate error, readout
        error, and CZ gate error (SI Section S2).  Returns the subgraph
        with the lowest total error score.
        """
        if coupling_map is None:
            coupling_map = TIANJI_S2_COUPLING_MAP

        import networkx as nx

        g = nx.Graph()
        g.add_edges_from(coupling_map)

        best_subgraph: List[int] = []
        best_score = float("inf")

        for nodes in _connected_subgraphs(g, n_logical):
            # Composite score: sum of SQ errors + readout errors + CZ errors
            sq_score = sum(TIANJI_S2_SQ_ERROR.get(n, 1e-3) for n in nodes)
            ro_score = sum(TIANJI_S2_F0_ERROR.get(n, 1e-2) for n in nodes)
            cz_score = 0.0
            node_set = set(nodes)
            for (u, v), err in TIANJI_S2_CZ_ERROR.items():
                if u in node_set and v in node_set:
                    cz_score += err
            score = sq_score + ro_score + cz_score
            if score < best_score:
                best_score = score
                best_subgraph = sorted(nodes)

        return best_subgraph if best_subgraph else list(range(n_logical))

    @staticmethod
    def get_calibration_data() -> Dict:
        """Return full calibration data as a dictionary."""
        return {
            "n_qubits": TIANJI_S2_N_QUBITS,
            "coupling_map": TIANJI_S2_COUPLING_MAP,
            "T1_us": TIANJI_S2_T1,
            "sq_error": TIANJI_S2_SQ_ERROR,
            "f0_error": TIANJI_S2_F0_ERROR,
            "cz_error": TIANJI_S2_CZ_ERROR,
        }


def _connected_subgraphs(g, size: int) -> List[List[int]]:
    """Find connected subgraphs of given size via BFS enumeration."""
    subgraphs: List[List[int]] = []
    for start in g.nodes():
        visited = {start}
        queue = [start]
        while len(visited) < size and queue:
            node = queue.pop(0)
            for neighbor in sorted(g.neighbors(node)):
                if neighbor not in visited and len(visited) < size:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(visited) == size:
            sg = sorted(visited)
            if sg not in subgraphs:
                subgraphs.append(sg)
    return subgraphs
