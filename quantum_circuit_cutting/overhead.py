"""Sampling overhead budget and cut optimization.

Implements overhead-constrained circuit knitting: given a maximum
sampling overhead budget kappa_max, determines the optimal set of
wire/gate cuts that minimizes the number of cuts K while respecting
the device qubit limit d.

Key insight from Peng et al. (PRL 125, 150504):
- Wire cut overhead per cut: gamma_wire = 4
- Gate cut overhead per CNOT: gamma_gate = 3
- Total overhead: kappa = prod(gamma_k) for k in cuts
- Sampling cost scales as kappa^2 * (1/epsilon^2)

Reference: "Overhead-constrained circuit knitting for variational
quantum dynamics" (arXiv:2309.xxxxx).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Overhead constants from Peng et al.
GAMMA_WIRE_CUT = 4.0  # Identity channel decomposition: sum(|c_i|) = 4
# Note: Gate-level QPD (GateQPDDecomposition) is defined in
# channel_decomposition.py but not yet integrated into the pipeline.
# The overhead model below uses wire-cut gamma=4 exclusively.


@dataclass
class OverheadBudget:
    """Overhead budget for circuit cutting.

    Attributes:
        kappa_max: Maximum total sampling overhead.
        max_cuts: Maximum number of cuts allowed (derived from kappa_max).
        gamma_per_cut: Overhead factor per cut (wire or gate).
        n_samples_base: Base number of samples before overhead scaling.
        effective_samples: Actual samples needed = n_samples_base * kappa^2.
    """

    kappa_max: float
    max_cuts: int
    gamma_per_cut: float
    n_samples_base: int
    effective_samples: int

    @property
    def kappa_actual(self) -> float:
        """Actual overhead for max_cuts cuts."""
        return self.gamma_per_cut**self.max_cuts


def compute_overhead_budget(
    n_qubits: int,
    max_cluster_size: int,
    n_samples_base: int = 10000,
    kappa_max: float | None = None,
) -> OverheadBudget:
    """Compute the overhead budget for a given problem size.

    If kappa_max is not specified, uses a heuristic based on the
    minimum number of clusters needed.

    Args:
        n_qubits: Total logical qubits.
        max_cluster_size: Device qubit limit d.
        n_samples_base: Base measurement shots.
        kappa_max: User-specified overhead limit. If None, computed
            from minimum required cuts.
    Note:
        Current implementation only supports wire cuts (gamma=4).
    """
    gamma = GAMMA_WIRE_CUT
    min_clusters = math.ceil(n_qubits / max_cluster_size)

    # Minimum cuts for a connected partition of min_clusters clusters
    # is min_clusters - 1 (tree connectivity).
    min_cuts = max(min_clusters - 1, 0)

    if kappa_max is None:
        # Default: allow overhead up to gamma^(2*min_cuts) to give room
        # for non-optimal partitions.
        max_cuts = min(min_cuts * 2, 20)  # Cap at 20 cuts
        kappa_max = gamma**max_cuts
    else:
        max_cuts = int(math.log(kappa_max) / math.log(gamma)) if kappa_max > 1 else 0

    effective = int(n_samples_base * (gamma**max_cuts) ** 2)

    return OverheadBudget(
        kappa_max=kappa_max,
        max_cuts=max_cuts,
        gamma_per_cut=gamma,
        n_samples_base=n_samples_base,
        effective_samples=effective,
    )


def estimate_sampling_cost(
    n_cuts: int,
    n_pauli_terms: int,
    n_clusters: int = 2,
    n_qwc_groups: int | None = None,
    n_shots: int = 10000,
    gamma: float = GAMMA_WIRE_CUT,
) -> Dict[str, float]:
    """Estimate the total sampling cost for a cutting configuration.

    Returns:
        Dict with 'kappa', 'total_circuits', 'total_shots', 'variance_factor'.
    """
    kappa = gamma**n_cuts
    n_configs = 8**n_cuts  # Wire cut only, 8 configurations per cut
    # If QWC groups not provided, assume no grouping (n_groups = n_terms)
    if n_qwc_groups is None:
        n_qwc_groups = n_pauli_terms
    # Total circuits = number of configs * clusters * QWC groups
    total_circuits = n_configs * n_clusters * n_qwc_groups
    variance_factor = kappa**2
    total_shots = total_circuits * n_shots

    return {
        "kappa": kappa,
        "n_configurations_exact": n_configs,
        "total_circuits_sampled": total_circuits,
        "total_shots": total_shots,
        "variance_factor": variance_factor,
    }


def select_cut_strategy(
    n_qubits: int,
    max_cluster_size: int,
    entangling_gates: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Select between exact enumeration and importance sampling.

    For small K (n_cuts <= 4), exact enumeration of 8^K configs is
    feasible. For larger K, importance sampling is required.

    Uses entangling gate graph to estimate minimum required cuts.
    """
    min_clusters = math.ceil(n_qubits / max_cluster_size)
    min_cuts = max(min_clusters - 1, 0)
    
    # Use entangling gates to adjust min_cuts estimate: count cross-partition gates
    if entangling_gates:
        # Simple balanced partition heuristic to estimate cross gates
        partition_point = n_qubits // 2
        cross_gate_count = 0
        for q0, q1 in entangling_gates:
            q_left = min(q0, q1)
            q_right = max(q0, q1)
            if q_left < partition_point and q_right >= partition_point:
                cross_gate_count += 1
        # Adjust min_cuts based on cross gate count (each cut handles up to 2 cross gates)
        min_cuts = max(min_cuts, (cross_gate_count + 1) // 2)

    EXACT_THRESHOLD = 4  # 8^4 = 4096 configs is manageable

    if min_cuts <= EXACT_THRESHOLD:
        return {
            "use_sampling": False,
            "n_samples": 8**min_cuts,
            "reason": f"{min_cuts} cuts -> {8**min_cuts} configs (exact enumeration feasible)",
        }
    else:
        # Scale samples with sqrt of exact count, capped to avoid explosion
        n_samples = min(int(math.sqrt(8**min_cuts)), 2000)
        return {
            "use_sampling": True,
            "n_samples": max(n_samples, 200),
            "reason": f"{min_cuts} cuts -> 8^{min_cuts} configs (sampling required)",
        }
