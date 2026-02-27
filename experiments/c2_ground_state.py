"""C2 ground-state potential energy curve (Fig. 2).

CAS(8e, 8o) / cc-pVTZ, self-consistent VQE-MC-PDFT with ftPBE functional.
Bond distances from 0.80 to 3.00 A (18-point grid matching SI Table S4).

When run directly, this script performs the full VQE-MC-PDFT calculation
at each geometry.  Pre-computed reference data from the SI is stored in
``data/c2_ground_state_pec.csv`` for figure generation without re-running.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd
from pyscf import gto, scf, mcscf, dft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqe_mcpdft import (
    CASCIAnsatz,
    MCPDFTEnergy,
    OrbitalRotationCircuit,
    RDMMeasurement,
    SelfConsistentVQEMCPDFT,
    build_qubit_hamiltonian,
)
from vqe_mcpdft.hamiltonian import _expand_integrals_to_spin_orbitals
from vqe_mcpdft.tqp_backend import TQPBackend
from quantum_circuit_cutting.cutting_dispatch import CuttingDispatcher

import argparse

N_ELEC, N_ORB = 8, 8
BASIS = "cc-pvtz"
BOND_DISTANCES = [
    0.80, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30, 1.40, 1.50,
    1.60, 1.70, 1.80, 2.00, 2.20, 2.40, 2.60, 2.80, 3.00,
]
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "c2_ground_state_pec.csv"


def build_c2(bond_a: float) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = f"C 0 0 0; C 0 0 {bond_a}"
    mol.basis = BASIS
    mol.spin = 0
    mol.build()
    return mol


def run_point(bond_a: float, tqp_backend=None) -> dict:
    """Run HF, DFT/PBE, CASSCF, VQE-SA-CASSCF, and VQE-MC-PDFT at one bond length."""
    mol = build_c2(bond_a)

    # HF
    mf = scf.RHF(mol).run()
    e_hf = mf.e_tot

    # DFT/PBE
    mf_dft = dft.RKS(mol)
    mf_dft.xc = "pbe"
    mf_dft.run()
    e_dft = mf_dft.e_tot

    # CASSCF
    mc = mcscf.CASSCF(mf, N_ORB, N_ELEC).run()
    e_casscf = mc.e_tot

    # VQE-MC-PDFT
    n_qubits = 2 * N_ORB
    h1e_cas, e_core = mc.get_h1eff()
    h2e_cas = mc.get_h2eff()
    mo_coeffs = mc.mo_coeff[:, mc.ncore : mc.ncore + N_ORB]

    # Build qubit Hamiltonian
    h1e_so, h2e_so = _expand_integrals_to_spin_orbitals(h1e_cas, h2e_cas)
    ham = build_qubit_hamiltonian(h1e_so, h2e_so, n_qubits, e_core)

    # Grid for MC-PDFT on-top functional
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    ao_vals = mol.eval_gto("GTOval_sph", grids.coords)

    mcpdft = MCPDFTEnergy(
        mo_coeffs=mo_coeffs,
        ao_values=ao_vals,
        grid_weights=grids.weights,
        h1e=h1e_cas,
        h2e=h2e_cas,
        e_core=e_core,
    )
    MCPDFTEnergy.validate_e_core(e_core, ham)

    ansatz = CASCIAnsatz(n_qubits, N_ELEC, n_layers=2)
    rdm_meas = RDMMeasurement(n_qubits)
    orb_rot = OrbitalRotationCircuit(N_ORB)

    # Use CuttingDispatcher: auto-routes through circuit cutting when
    # n_qubits > 13 (Tianji-S2 limit).  CAS(8e,8o) -> 16 qubits -> cutting.
    dispatcher = CuttingDispatcher(max_device_qubits=13, tqp_backend=tqp_backend)
    energy_fn_factory = dispatcher.make_energy_fn_factory(
        n_qubits, N_ELEC, ham,
    )
    pauli_expval_fn = dispatcher.make_pauli_expval_fn(
        n_qubits, N_ELEC, ham,
    )

    sc = SelfConsistentVQEMCPDFT(
        ansatz=ansatz,
        mcpdft=mcpdft,
        rdm_measurer=rdm_meas,
        orbital_rotator=orb_rot,
        energy_fn_factory=energy_fn_factory,
        pauli_expval_fn=pauli_expval_fn,
        max_outer_iter=20,
    )
    result = sc.run()

    return {
        "bond_A": bond_a,
        "HF": e_hf,
        "DFT_PBE": e_dft,
        "CASSCF": e_casscf,
        "VQE_SA_CASSCF": result.energy_mcscf,
        "VQE_MC_PDFT": result.energy_mcpdft,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2 ground-state PEC (Fig.2)")
    parser.add_argument("--token", type=str, required=True,
                        help="TQP authentication token")
    args = parser.parse_args()

    tqp_backend = TQPBackend(token=args.token)

    np.random.seed(42)  # Reproducibility
    rows = [run_point(d, tqp_backend=tqp_backend) for d in BOND_DISTANCES]
    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(df.to_string(index=False))
    logger.info(f"\nResults saved to {OUTPUT_CSV}")
