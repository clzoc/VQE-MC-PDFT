"""VQE-MC-PDFT: Hybrid quantum-classical framework for strongly correlated systems.

Combines the Variational Quantum Eigensolver (VQE) with Multiconfiguration
Pair-Density Functional Theory (MC-PDFT) for accurate treatment of both
static and dynamic electron correlation on near-term quantum hardware.
"""

from vqe_mcpdft.hamiltonian import build_qubit_hamiltonian, hamiltonian_from_pyscf
from vqe_mcpdft.ansatz import CASCIAnsatz
from vqe_mcpdft.orbital_rotation import OrbitalRotationCircuit
from vqe_mcpdft.mcpdft_energy import MCPDFTEnergy
from vqe_mcpdft.rdm import RDMMeasurement
from vqe_mcpdft.vqe_solver import VQESolver
from vqe_mcpdft.self_consistent import SelfConsistentVQEMCPDFT
from vqe_mcpdft.tqp_backend import TQPBackend

__all__ = [
    "build_qubit_hamiltonian",
    "hamiltonian_from_pyscf",
    "CASCIAnsatz",
    "OrbitalRotationCircuit",
    "MCPDFTEnergy",
    "RDMMeasurement",
    "VQESolver",
    "SelfConsistentVQEMCPDFT",
    "TQPBackend",
]
