"""C2 excited-state potential energy curves (Fig. 3).

9 states via SA-CASSCF(8e,8o)/cc-pVTZ + VQE-MC-PDFT(ftPBE).
States: X1Sg+, a3Pu, b3Sg-, A1Pu, c3Su+, B1Dg, Bp1Sg+, d3Pg, C1Pg.

Uses state-averaged CASSCF to obtain orbitals, then runs VQE-MC-PDFT
for each state individually using the SA-CASSCF orbitals as the starting
point.  Pre-computed data is in ``data/c2_excited_state_pec.csv``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
from quantum_circuit_cutting.cutting_dispatch import CuttingDispatcher

from vqe_mcpdft.tqp_backend import TQPBackend

import argparse

N_ELEC, N_ORB = 8, 8
BASIS = "cc-pvtz"
N_STATES = 9
STATE_LABELS = [
    "X1Sg+", "a3Pu", "b3Sg-", "A1Pu", "c3Su+",
    "B1Dg", "Bp1Sg+", "d3Pg", "C1Pg",
]
BOND_DISTANCES = [
    0.80, 0.90, 1.00, 1.10, 1.20, 1.25, 1.30, 1.40, 1.50,
    1.60, 1.70, 1.80, 2.00, 2.20, 2.40, 2.60, 2.80, 3.00,
]
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "c2_excited_state_pec.csv"


def _estimate_overlap_symmetric(
    pauli_expval_fn, params1, params2, orb_angles, n_qubits,
    initial_samples=100, var_threshold=1e-3, max_samples=2000,
):
    """Estimate |<psi1|psi2>|^2 via unbiased random Pauli measurement.

    Delegates to ``OverlapEstimator`` which uses the identity:
        |<psi1|psi2>|^2 = (1/2^n) sum_P <P>_1 * <P>_2

    with a fixed random Pauli set for reproducibility.  No artificial
    post-processing beyond [0, 1] clipping of statistical outliers.
    """
    from vqe_mcpdft.overlap_estimator import OverlapEstimator

    estimator = OverlapEstimator(
        n_qubits=n_qubits,
        n_random_paulis=initial_samples,
        seed=42,
        var_threshold=var_threshold,
        max_paulis=max_samples,
    )
    return estimator.estimate(pauli_expval_fn, params1, params2, orb_angles)


def build_c2(bond_a: float) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = f"C 0 0 0; C 0 0 {bond_a}"
    mol.basis = BASIS
    mol.spin = 0
    mol.build()
    return mol


def run_point(bond_a: float, tqp_backend=None) -> dict:
    """Run SA-CASSCF + VQE-MC-PDFT for all 9 states at one geometry.

    Each state is targeted by initializing the VQE from the corresponding
    SA-CASSCF CI vector and adding a penalty term that enforces
    orthogonality to previously converged lower-energy states.
    """
    mol = build_c2(bond_a)
    mf = scf.RHF(mol).run()

    # State-averaged CASSCF for orbital optimization
    mc = mcscf.CASSCF(mf, N_ORB, N_ELEC)
    weights = [1.0 / N_STATES] * N_STATES
    mc.state_average_(weights)
    mc.run()

    n_qubits = 2 * N_ORB
    h1e_cas, e_core = mc.get_h1eff()
    h2e_cas = mc.get_h2eff()
    mo_coeffs = mc.mo_coeff[:, mc.ncore : mc.ncore + N_ORB]

    # Build qubit Hamiltonian
    h1e_so, h2e_so = _expand_integrals_to_spin_orbitals(h1e_cas, h2e_cas)
    ham = build_qubit_hamiltonian(h1e_so, h2e_so, n_qubits, e_core)

    # Grid for MC-PDFT
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

    row: dict = {"bond_A": bond_a}
    ansatz = CASCIAnsatz(n_qubits, N_ELEC, n_layers=2)
    rdm_meas = RDMMeasurement(n_qubits)
    orb_rot = OrbitalRotationCircuit(N_ORB)

    # CuttingDispatcher: auto-routes through circuit cutting when
    # n_qubits > 13.  CAS(8e,8o) -> 16 qubits -> cutting.
    dispatcher = CuttingDispatcher(max_device_qubits=13, tqp_backend=tqp_backend)
    pauli_expval_fn = dispatcher.make_pauli_expval_fn(n_qubits, N_ELEC, ham)

    # Collect converged states for orthogonality penalty
    converged_params: list = []
    PENALTY_WEIGHT = 10.0  # Ha, large enough to enforce orthogonality

    for state_idx, label in enumerate(STATE_LABELS):
        # Build base energy function via dispatcher (handles cutting)
        base_energy_fn_factory = dispatcher.make_energy_fn_factory(
            n_qubits, N_ELEC, ham,
        )

        def energy_fn_factory(orb_angles, _si=state_idx, _prev=list(converged_params),
                              _base_factory=base_energy_fn_factory):
            base_fn = _base_factory(orb_angles)
            def energy_fn(params):
                e = base_fn(params)
                # Add overlap penalty with previously converged states
                for prev_params in _prev:
                    overlap = _estimate_overlap_symmetric(
                        pauli_expval_fn, prev_params, params,
                        orb_angles, n_qubits,
                    )
                    e += PENALTY_WEIGHT * overlap
                return e
            return energy_fn

        # Use different initial params for each state to break symmetry
        init_params = ansatz.initial_params(seed=42 + state_idx)

        sc = SelfConsistentVQEMCPDFT(
            ansatz=ansatz, mcpdft=mcpdft, rdm_measurer=rdm_meas,
            orbital_rotator=orb_rot, energy_fn_factory=energy_fn_factory,
            pauli_expval_fn=pauli_expval_fn,
            max_outer_iter=20,
        )
        result = sc.run(initial_params=init_params)
        row[label] = result.energy_mcpdft
        converged_params.append(result.optimal_params)

    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C2 excited-state PEC (Fig.3)")
    parser.add_argument("--token", type=str, required=True,
                        help="TQP authentication token (required)")
    args = parser.parse_args()

    tqp_backend = TQPBackend(token=args.token)

    np.random.seed(42)  # Reproducibility
    rows = [run_point(d, tqp_backend=tqp_backend) for d in BOND_DISTANCES]
    df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_CSV.parent, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")
