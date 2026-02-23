"""Quantum circuit cutting for scalable VQE-MC-PDFT simulations.

Implements the cluster simulation framework (Peng et al., PRL 125, 150504)
for partitioning large quantum circuits into smaller fragments executable
on limited-qubit hardware, with classical post-processing reconstruction.

Modules:
    partition: Qubit partitioning (spectral bisection, KL refinement, symmetry-guided)
    channel_decomposition: Identity channel QPD (8-channel decomposition)
    fragment_circuits: Fragment circuit generation with cut-point operations
    reconstruction: Classical post-processing with observable grouping
    overhead: Sampling overhead budget and cut optimization
    cutting_vqe: Full partitioned VQE-MC-PDFT driver
    cutting_dispatch: Auto-routing dispatcher for transparent cutting
"""

from quantum_circuit_cutting.channel_decomposition import (
    ChannelDecomposition,
    decompose_identity_channel,
)
from quantum_circuit_cutting.overhead import (
    OverheadBudget,
    compute_overhead_budget,
    estimate_sampling_cost,
    select_cut_strategy,
)
from quantum_circuit_cutting.partition import CircuitPartitioner, ClusterPartition
from quantum_circuit_cutting.reconstruction import (
    CuttingReconstructor,
    group_commuting_paulis,
)


def __getattr__(name: str):
    """Lazy imports for modules that depend on tensorcircuit."""
    if name == "FragmentCircuitGenerator":
        from quantum_circuit_cutting.fragment_circuits import FragmentCircuitGenerator

        return FragmentCircuitGenerator
    if name == "PartitionedVQEMCPDFT":
        from quantum_circuit_cutting.cutting_vqe import PartitionedVQEMCPDFT

        return PartitionedVQEMCPDFT
    if name == "CuttingDispatcher":
        from quantum_circuit_cutting.cutting_dispatch import CuttingDispatcher

        return CuttingDispatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChannelDecomposition",
    "decompose_identity_channel",
    "CircuitPartitioner",
    "ClusterPartition",
    "CuttingReconstructor",
    "group_commuting_paulis",
    "OverheadBudget",
    "compute_overhead_budget",
    "estimate_sampling_cost",
    "select_cut_strategy",
    "FragmentCircuitGenerator",
    "PartitionedVQEMCPDFT",
    "CuttingDispatcher",
]
