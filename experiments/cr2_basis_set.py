"""Cr2 basis-set convergence study (Fig. 5).

CAS(12e,22o) with cc-pVXZ-DK (X = T, Q, 5).
Self-consistent VQE-MC-PDFT with ftPBE functional.
Pre-computed data is in ``data/cr2_basis_set.csv``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyscf import gto, scf, mcscf, dft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqe_mcpdft import (  # noqa: E402
    CASCIAnsatz, MCPDFTEnergy, OrbitalRotationCircuit,
    RDMMeasurement, SelfConsistentVQEMCPDFT, build_qubit_hamiltonian,
)
from vqe_mcpdft.hamiltonian import _expand_integrals_to_spin_orbitals  # noqa: E402
from quantum_circuit_cutting.cutting_dispatch import CuttingDispatcher  # noqa: E402
from vqe_mcpdft.tqp_backend import TQPBackend  # noqa: E402

import argparse

N_ELEC, N_ORB = 12, 22
BASIS_SETS = ["cc-pvtz-dk", "cc-pvqz-dk", "cc-pv5z-dk"]
BASIS_LABELS = ["TZ", "QZ", "5Z"]
BOND_DISTANCES = [
    1.50, 1.55, 1.60, 1.68, 1.70, 1.80,
    2.00, 2.25, 2.50, 2.75, 3.00, 3.25,
]
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "cr2_basis_set.csv"


def run_basis(bond_a: float, basis: str, tqp_backend=None) -> float:
    """Run VQE-MC-PDFT for one bond length and basis set."""
    mol = gto.Mole()
    mol.atom = f"Cr 0 0 0; Cr 0 0 {bond_a}"
    mol.basis = basis
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, N_ORB, N_ELEC).run()

    n_qubits = 2 * N_ORB
    h1e_cas, e_core = mc.get_h1eff()
    h2e_cas = mc.get_h2eff()
    mo_coeffs = mc.mo_coeff[:, mc.ncore : mc.ncore + N_ORB]

    h1e_so, h2e_so = _expand_integrals_to_spin_orbitals(h1e_cas, h2e_cas)
    ham = build_qubit_hamiltonian(h1e_so, h2e_so, n_qubits, e_core)

    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    ao_vals = mol.eval_gto("GTOval_sph", grids.coords)

    mcpdft = MCPDFTEnergy(
        mo_coeffs=mo_coeffs, ao_values=ao_vals,
        grid_weights=grids.weights, h1e=h1e_cas, h2e=h2e_cas,
        e_core=e_core,
    )
    MCPDFTEnergy.validate_e_core(e_core, ham)
    ansatz = CASCIAnsatz(n_qubits, N_ELEC, n_layers=2)
    rdm_meas = RDMMeasurement(n_qubits)
    orb_rot = OrbitalRotationCircuit(N_ORB)

    # CuttingDispatcher: auto-routes through circuit cutting when
    # n_qubits > 13.  CAS(12e,22o) -> 44 qubits -> cutting.
    dispatcher = CuttingDispatcher(max_device_qubits=13, tqp_backend=tqp_backend)
    energy_fn_factory = dispatcher.make_energy_fn_factory(
        n_qubits, N_ELEC, ham,
    )
    pauli_expval_fn = dispatcher.make_pauli_expval_fn(
        n_qubits, N_ELEC, ham,
    )

    sc = SelfConsistentVQEMCPDFT(
        ansatz=ansatz, mcpdft=mcpdft, rdm_measurer=rdm_meas,
        orbital_rotator=orb_rot, energy_fn_factory=energy_fn_factory,
        pauli_expval_fn=pauli_expval_fn,
        max_outer_iter=20,
    )
    return sc.run().energy_mcpdft


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cr2 basis-set convergence (Fig.5)")
    parser.add_argument("--token", type=str, required=True,
                        help="TQP authentication token")
    args = parser.parse_args()

    tqp_backend = TQPBackend(token=args.token)

    np.random.seed(42)  # Reproducibility
    rows: list = []
    for bond_a in BOND_DISTANCES:
        row: dict = {"bond_A": bond_a}
        for basis, label in zip(BASIS_SETS, BASIS_LABELS):
            e = run_basis(bond_a, basis, tqp_backend=tqp_backend)
            row[f"MCPDFT_{label}"] = e
            print(f"R={bond_a}, {label}: E = {e:.5f} Ha")
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")
