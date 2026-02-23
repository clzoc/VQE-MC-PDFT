"""Cr2 active-space scaling study (Fig. 4).

Three active spaces: (12e,12o), (12e,22o), (12e,28o) with cc-pVTZ-DK basis.
Self-consistent VQE-MC-PDFT with ftPBE functional.
Pre-computed data is in ``data/cr2_active_space.csv``.
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

N_ELEC = 12
ACTIVE_SPACES = [(12, 12), (12, 22), (12, 28)]
BASIS = "cc-pvtz-dk"
BOND_DISTANCES = [
    1.50, 1.55, 1.60, 1.68, 1.70, 1.80,
    2.00, 2.25, 2.50, 2.75, 3.00, 3.25,
]
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "cr2_active_space.csv"


def build_cr2(bond_a: float) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = f"Cr 0 0 0; Cr 0 0 {bond_a}"
    mol.basis = BASIS
    mol.spin = 0
    mol.build()
    return mol


def run_active_space(mol, mf, n_elec: int, n_orb: int, tqp_backend=None) -> tuple:
    """Return (CASSCF energy, MC-PDFT energy) for one active space."""
    mc = mcscf.CASSCF(mf, n_orb, n_elec).run()
    e_casscf = mc.e_tot

    n_qubits = 2 * n_orb
    h1e_cas, e_core = mc.get_h1eff()
    h2e_cas = mc.get_h2eff()
    mo_coeffs = mc.mo_coeff[:, mc.ncore : mc.ncore + n_orb]

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
    ansatz = CASCIAnsatz(n_qubits, n_elec, n_layers=2)
    rdm_meas = RDMMeasurement(n_qubits)
    orb_rot = OrbitalRotationCircuit(n_orb)

    # CuttingDispatcher: auto-routes through circuit cutting when
    # n_qubits > 13.  All Cr2 active spaces (24/44/56 qubits) need cutting.
    dispatcher = CuttingDispatcher(max_device_qubits=13, tqp_backend=tqp_backend)
    energy_fn_factory = dispatcher.make_energy_fn_factory(
        n_qubits, n_elec, ham,
    )
    pauli_expval_fn = dispatcher.make_pauli_expval_fn(
        n_qubits, n_elec, ham,
    )

    sc = SelfConsistentVQEMCPDFT(
        ansatz=ansatz, mcpdft=mcpdft, rdm_measurer=rdm_meas,
        orbital_rotator=orb_rot, energy_fn_factory=energy_fn_factory,
        pauli_expval_fn=pauli_expval_fn,
        max_outer_iter=20,
    )
    return e_casscf, sc.run().energy_mcpdft


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cr2 active-space scaling (Fig.4)")
    parser.add_argument("--token", type=str, required=True,
                        help="TQP authentication token")
    args = parser.parse_args()

    tqp_backend = TQPBackend(token=args.token)

    np.random.seed(42)  # Reproducibility
    rows: list = []
    for bond_a in BOND_DISTANCES:
        mol = build_cr2(bond_a)
        mf = scf.RHF(mol).run()
        row: dict = {"bond_A": bond_a}
        for ne, no in ACTIVE_SPACES:
            e_cas, e_pdft = run_active_space(mol, mf, ne, no, tqp_backend=tqp_backend)
            row[f"CASSCF_{ne}e{no}o"] = e_cas
            row[f"MCPDFT_{ne}e{no}o"] = e_pdft
            print(f"R={bond_a} ({ne}e,{no}o): CASSCF={e_cas:.5f}, MC-PDFT={e_pdft:.5f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")
