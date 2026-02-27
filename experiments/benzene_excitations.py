"""Benzene excitation energies (Fig. 7).

CAS(6e,6o) / aug-cc-pVTZ, VQE-MC-PDFT with PBE functional.
5 pi->pi* transitions compared to TBE/CBS, TDDFT, and NES-VMC references.
Pre-computed data is in ``data/benzene_excitations.csv``.
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyscf import gto, scf, mcscf, dft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vqe_mcpdft import (
    CASCIAnsatz, MCPDFTEnergy, OrbitalRotationCircuit,
    RDMMeasurement, SelfConsistentVQEMCPDFT, build_qubit_hamiltonian,
)
from vqe_mcpdft.hamiltonian import _expand_integrals_to_spin_orbitals
from vqe_mcpdft.tqp_backend import TQPBackend
from quantum_circuit_cutting.cutting_dispatch import CuttingDispatcher

N_ELEC, N_ORB = 6, 6
BASIS = "aug-cc-pvtz"
N_STATES = 6  # ground + 5 excited
TRANSITIONS = [
    "11A1g_13B1u", "11A1g_13E1u", "11A1g_11B2u",
    "11A1g_13B2u", "11A1g_11B1u",
]
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "benzene_excitations.csv"

BENZENE_GEOM = """
C  1.3950  0.0000  0.0
C  0.6975  1.2083  0.0
C -0.6975  1.2083  0.0
C -1.3950  0.0000  0.0
C -0.6975 -1.2083  0.0
C  0.6975 -1.2083  0.0
H  2.4803  0.0000  0.0
H  1.2401  2.1479  0.0
H -1.2401  2.1479  0.0
H -2.4803  0.0000  0.0
H -1.2401 -2.1479  0.0
H  1.2401 -2.1479  0.0
"""


def build_benzene() -> gto.Mole:
    mol = gto.Mole()
    mol.atom = BENZENE_GEOM
    mol.basis = BASIS
    mol.build()
    return mol


def run_benzene(tqp_backend: TQPBackend) -> pd.DataFrame:
    """Run SA-CASSCF + VQE-MC-PDFT for benzene ground + 5 excited states."""
    mol = build_benzene()
    mf = scf.RHF(mol).run()

    mc = mcscf.CASSCF(mf, N_ORB, N_ELEC)
    weights = [1.0 / N_STATES] * N_STATES
    mc.state_average_(weights)
    mc.run()

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
    ansatz = CASCIAnsatz(n_qubits, N_ELEC, n_layers=2)
    rdm_meas = RDMMeasurement(n_qubits)
    orb_rot = OrbitalRotationCircuit(N_ORB)
    dispatcher = CuttingDispatcher(max_device_qubits=13, tqp_backend=tqp_backend)
    energy_fn_factory = dispatcher.make_energy_fn_factory(n_qubits, N_ELEC, ham)
    pauli_expval_fn = dispatcher.make_pauli_expval_fn(n_qubits, N_ELEC, ham)

    state_energies: list = []
    for si in range(N_STATES):
        sc = SelfConsistentVQEMCPDFT(
            ansatz=ansatz, mcpdft=mcpdft, rdm_measurer=rdm_meas,
            orbital_rotator=orb_rot, energy_fn_factory=energy_fn_factory,
            pauli_expval_fn=pauli_expval_fn,
            max_outer_iter=20,
        )
        state_energies.append(sc.run().energy_mcpdft)

    ha_to_ev = 27.2114
    excitations_ev = [(e - state_energies[0]) * ha_to_ev for e in state_energies[1:]]

    rows = []
    for label, vqe_ev in zip(TRANSITIONS, excitations_ev):
        rows.append({"transition": label, "VQE_MCPDFT_eV": round(vqe_ev, 2)})
    df_new = pd.DataFrame(rows)

    # Merge with existing reference data if available
    if OUTPUT_CSV.exists():
        df_ref = pd.read_csv(OUTPUT_CSV)
        ref_cols = [c for c in df_ref.columns if c not in ("transition", "VQE_MCPDFT_eV")]
        if ref_cols:
            df_new = df_new.merge(df_ref[["transition"] + ref_cols], on="transition", how="left")

    return df_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benzene excitations (Fig.7)")
    parser.add_argument("--token", type=str, required=True,
                        help="TQP authentication token")
    args = parser.parse_args()
    tqp_backend = TQPBackend(token=args.token)

    np.random.seed(42)  # Reproducibility
    df = run_benzene(tqp_backend=tqp_backend)
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_CSV}")
