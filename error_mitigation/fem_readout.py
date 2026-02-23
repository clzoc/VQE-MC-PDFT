"""FEM-inspired iterative readout error mitigation.

Implements the multi-stage iterative correction from Eq. 33 of the manuscript:
    b^(i+1) = (M_i1 x M_i2 x ... x M_iK)^{-1} b^(i)

Aligned with the QuFEM ParticalLocalMitigator / MultiStageMitigator approach:
qubits are partitioned into groups, per-group calibration matrices are
constructed from protocol results, and the correction is applied via
sparse Kronecker-product inversion iteratively with re-partitioning.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class FEMReadoutMitigator:
    """FEM-inspired iterative readout error mitigator.

    Implements the divide-and-conquer strategy from Eq. 33 and the QuFEM
    framework: partition qubits into clusters, build per-cluster noise
    matrices from calibration data, and apply iterative correction with
    different random partitionings at each stage.

    Attributes:
        n_qubits: Number of qubits in the system.
        cal_matrices: Per-group calibration matrices from characterization.
        groups: Qubit index groupings used for tensor-product factorization.
        protocol_results: Full calibration data for re-partitioning.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.cal_matrices: List[np.ndarray] = []
        self.groups: List[List[int]] = []
        self.protocol_results: Dict[str, Dict[str, int]] = {}

    def characterize(
        self,
        protocol_results: Dict[str, Dict[str, int]],
        groups: List[List[int]],
    ) -> None:
        """Build per-group calibration matrices from measurement data.

        Args:
            protocol_results: Mapping ``prepared_state -> {bitstring: counts}``
                for every computational-basis preparation.
            groups: Partition of qubit indices, e.g. ``[[0,1],[2,3]]``.
        """
        self.protocol_results = protocol_results
        self.groups = groups
        self.cal_matrices = self._build_group_matrices(groups, protocol_results)

    def _build_group_matrices(
        self,
        groups: List[List[int]],
        protocol_results: Dict[str, Dict[str, int]],
    ) -> List[np.ndarray]:
        """Construct per-group noise matrices from calibration data."""
        matrices = []
        for group in groups:
            k = len(group)
            dim = 2**k
            M = np.zeros((dim, dim))
            for col_idx in range(dim):
                # Build the full bitstring for this prepared state
                label = format(col_idx, f"0{k}b")
                counts = protocol_results.get(label, {})
                total = sum(counts.values()) or 1
                for row_idx in range(dim):
                    row_label = format(row_idx, f"0{k}b")
                    M[row_idx, col_idx] = counts.get(row_label, 0) / total
            # Ensure columns sum to 1 (stochastic matrix)
            col_sums = M.sum(axis=0)
            col_sums[col_sums == 0] = 1
            M /= col_sums
            matrices.append(M)
        return matrices

    def mitigate(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Apply single-stage readout correction (Eq. 33, one iteration).

        Uses sparse Kronecker-product inversion following QuFEM's
        ParticalLocalMitigator approach.
        """
        dim = 2**self.n_qubits
        total = sum(counts.values()) or 1
        b = np.zeros(dim)
        for bitstring, c in counts.items():
            idx = int(bitstring, 2)
            if idx < dim:
                b[idx] = c / total

        # Compute inverse of tensor product of group matrices
        M_inv = self.cal_matrices[0].copy()
        M_inv = np.linalg.inv(M_inv)
        for M in self.cal_matrices[1:]:
            M_inv = np.kron(M_inv, np.linalg.inv(M))

        b_corr = M_inv @ b

        # Clip negative values and renormalize
        b_corr = np.maximum(b_corr, 0.0)
        total_corr = b_corr.sum()
        if total_corr > 0:
            b_corr /= total_corr

        return {
            format(i, f"0{self.n_qubits}b"): float(v)
            for i, v in enumerate(b_corr)
            if abs(v) > 1e-10
        }

    def multi_stage_mitigate(
        self,
        counts: Dict[str, int],
        n_stages: int = 3,
        group_size: int = 2,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        """Iteratively correct with re-partitioned qubit groups each stage.

        At each stage, multiple random partitionings are tried and the
        best one (evaluated on calibration data) is selected, following
        the QuFEM MultiStageMitigator algorithm.

        Args:
            counts: Raw measurement counts.
            n_stages: Number of iterative correction rounds.
            group_size: Target group size for random partitioning.
            rng: Optional random generator for reproducibility.

        Returns:
            Mitigated quasi-probability distribution.
        """
        rng = rng or np.random.default_rng(42)
        dim = 2**self.n_qubits
        total = sum(counts.values()) or 1
        b = np.zeros(dim)
        for bitstring, c in counts.items():
            idx = int(bitstring, 2)
            if idx < dim:
                b[idx] = c / total

        n_candidates = 3  # Try 3 random partitions per stage, pick best

        for stage in range(n_stages):
            best_groups = None
            best_score = -1.0

            for _ in range(n_candidates):
                groups = self.random_partition(self.n_qubits, group_size, rng)
                matrices = self._build_group_matrices(groups, self.protocol_results)

                # Evaluate: how well does this partition recover calibration states
                score = self._evaluate_partition(matrices, groups)
                if score > best_score:
                    best_score = score
                    best_groups = groups

            if best_groups is None:
                continue

            # Apply correction with best partition
            matrices = self._build_group_matrices(best_groups, self.protocol_results)
            M_inv = np.linalg.inv(matrices[0])
            for M in matrices[1:]:
                M_inv = np.kron(M_inv, np.linalg.inv(M))
            b = M_inv @ b

        # Clip and renormalize
        b = np.maximum(b, 0.0)
        total_b = b.sum()
        if total_b > 0:
            b /= total_b

        return {
            format(i, f"0{self.n_qubits}b"): float(v)
            for i, v in enumerate(b)
            if abs(v) > 1e-10
        }

    def _evaluate_partition(
        self,
        matrices: List[np.ndarray],
        groups: List[List[int]],
    ) -> float:
        """Score a partition by how well it recovers calibration states."""
        score = 0.0
        n_states = 0
        for prepared, measured in self.protocol_results.items():
            total = sum(measured.values()) or 1
            b = np.zeros(2**self.n_qubits)
            for bs, c in measured.items():
                idx = int(bs, 2)
                if idx < len(b):
                    b[idx] = c / total

            M_inv = np.linalg.inv(matrices[0])
            for M in matrices[1:]:
                M_inv = np.kron(M_inv, np.linalg.inv(M))
            b_corr = M_inv @ b
            b_corr = np.maximum(b_corr, 0.0)

            target_idx = int(prepared, 2)
            if target_idx < len(b_corr):
                score += b_corr[target_idx]
            n_states += 1

        return score / max(n_states, 1)

    @staticmethod
    def random_partition(
        n_qubits: int,
        group_size: int = 2,
        rng: Optional[np.random.Generator] = None,
    ) -> List[List[int]]:
        """Randomly partition qubit indices into groups."""
        rng = rng or np.random.default_rng(42)
        indices = rng.permutation(n_qubits).tolist()
        return [indices[i : i + group_size] for i in range(0, n_qubits, group_size)]

    @classmethod
    def from_error_rates(
        cls,
        n_qubits: int,
        f0_errors: Dict[str, float],
        f1_errors: Dict[str, float],
    ) -> "FEMReadoutMitigator":
        """Create a calibrated mitigator directly from per-qubit error rates.

        Constructs per-qubit confusion matrices from F0 (P(1|0)) and
        F1 (P(0|1)) error rates without requiring full protocol results.
        This enables FEM to work out-of-the-box with the calibration
        data already present in ``tianji_s2_calibration.json``.

        Args:
            n_qubits: Number of qubits.
            f0_errors: Dict mapping qubit name (e.g. "Q0") to F0 error rate.
            f1_errors: Dict mapping qubit name (e.g. "Q0") to F1 error rate.

        Returns:
            Calibrated FEMReadoutMitigator instance.
        """
        mitigator = cls(n_qubits)
        groups = [[i] for i in range(n_qubits)]
        matrices = []
        for i in range(n_qubits):
            qname = f"Q{i}"
            e0 = f0_errors.get(qname, 0.005)
            e1 = f1_errors.get(qname, 0.015)
            # Confusion matrix: M[measured][prepared]
            # Column 0 = prepared |0>: P(0|0)=1-e0, P(1|0)=e0
            # Column 1 = prepared |1>: P(0|1)=e1, P(1|1)=1-e1
            M = np.array([[1 - e0, e1], [e0, 1 - e1]])
            matrices.append(M)
        mitigator.groups = groups
        mitigator.cal_matrices = matrices
        return mitigator
