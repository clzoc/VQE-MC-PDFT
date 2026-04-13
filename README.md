# VQE-MC-PDFT

**Multiconfiguration Pair-Density Functional Theory Calculations of Low-lying States of Complex Chemical Systems with Quantum Computers**

Code repository accompanying the PNAS manuscript. This package implements a hybrid quantum-classical framework combining the Variational Quantum Eigensolver (VQE) with Multiconfiguration Pair-Density Functional Theory (MC-PDFT), featuring quantum circuit cutting for scalable active-space simulations and FEM-inspired readout error mitigation.

## Method Overview

VQE-MC-PDFT decouples electron correlation into two components:

1. **Static correlation** is captured by a multiconfigurational wavefunction prepared on a quantum processor via a particle-number-conserving CASCI ansatz.
2. **Dynamic correlation** is recovered classically through the fully-translated PBE (ftPBE) on-top pair-density functional, evaluated from the measured 1- and 2-RDMs.

Self-consistent orbital optimization is achieved by computing the MC-PDFT orbital gradient (generalized Fock matrix), compiling the resulting orbital-rotation generator into a Trotterized quantum circuit, and appending it to the ansatz. The framework is validated on C2, Cr2, and benzene using the Tencent Tianji-S2 superconducting quantum processor.

## Repository Structure

```
VQE-MC-PDFT/
├── vqe_mcpdft/                  # Core VQE-MC-PDFT implementation
├── quantum_circuit_cutting/     # Circuit cutting framework
├── error_mitigation/            # FEM / ZNE / CDR mitigation utilities
├── experiments/                 # Hardware-facing reproduction scripts
│   ├── c2_ground_state.py
│   ├── c2_excited_states.py
│   ├── cr2_active_space.py
│   ├── cr2_basis_set.py
│   ├── cr2_1p5A_cutting.py
│   ├── benzene_excitations.py
│   └── generate_figures.py
├── plot/                        # Figure renderers used by generate_figures.py
│   ├── plot_fig2_c2_ground_state.py
│   ├── plot_fig3_c2_excited_states.py
│   ├── plot_fig4_cr2_active_space.py
│   ├── plot_fig5_cr2_basis_set.py
│   ├── plot_fig6_cr2_qubit_utilization.py
│   ├── plot_fig7_benzene_vertical_excitations.py
│   └── plot_supp_tianji_s2_hardware_fidelity.py
├── data/                        # SI tables plus figure-rendering reference artifacts
│   ├── c2_ground_state_pec.csv
│   ├── c2_excited_state_pec.csv
│   ├── c2_excitation_energies.csv
│   ├── cr2_active_space.csv
│   ├── cr2_basis_set.csv
│   ├── cr2_boxplot_raw.csv
│   ├── benzene_excitations.csv
│   ├── cr2_active_space_figure.csv
│   ├── cr2_basis_set_figure.csv
│   ├── cr2_larsson_reference_curve.csv
│   └── tianji_s2_calibration.json
├── tests/
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Execution Model

The framework supports two hardware execution modes, selected automatically based on qubit count:

**Direct mode** (n_qubits <= 13): The full circuit is executed as one job on Tianji-S2 through `TQPBackend`.

**Cutting mode** (n_qubits > 13): The circuit is partitioned into fragments of at most 13 qubits via spectral bisection with Kernighan-Lin refinement. Each fragment is executed independently through `TQPBackend`, and expectation values/RDMs are reconstructed classically via quasi-probability decomposition (Peng et al., PRL 125, 150504).

The `CuttingDispatcher` class handles mode selection transparently. In cutting mode, both energy evaluation and RDM measurement use the cutting reconstruction path.
Current cutting implementation uses static wire-cut channels and gate-level QPD expansion for cross-cluster orbital-rotation terms; mid-circuit measurement/feedforward remains unsupported.

| Experiment | Active Space | Qubits | Mode |
|---|---|---|---|
| C2 ground/excited | CAS(8e,8o) | 16 | Cutting |
| Cr2 active space | CAS(12e,12o/22o/28o) | 24/44/56 | Cutting |
| Cr2 basis set | CAS(12e,22o) | 44 | Cutting |
| Cr2 1.5Å scaling | CAS(48e,42o) | 52-84 | Cutting |
| Benzene | CAS(6e,6o) | 12 | Direct |

## Installation

```bash
git clone <repository-url>
cd <repo-dir>
pip install -r requirements.txt   # exact version pins for reproducibility
pip install -e ".[dev]"
```

### Prerequisites

- Python >= 3.9
- NumPy, SciPy, PySCF (>= 2.4), TensorCircuit (>= 0.12), NetworkX, pandas, matplotlib
- For quantum hardware execution: Tencent Quantum Platform SDK token

## Reproducing Results

### Generate Figures from Pre-computed Data

All numerical data used by the publication and reviewer plotting scripts is provided in `data/`, including SI Tables S4-S10 and auxiliary high-precision/reference artifacts used for figure rendering. To generate publication figures without re-running quantum computations:

```bash
python experiments/generate_figures.py
```

Figures are saved to `figures/` using manuscript-style filenames such as `fig2_c2_ground_state.pdf` through `fig7_benzene_vertical_excitations.pdf`, with supplementary reviewer-facing plots saved separately, for example `figS1_tianji_s2_hardware_fidelity.pdf`.

For the Cr2 qubit-utilization boxplot figure, the raw scatter points use a small Gaussian horizontal jitter inside each box (`np.random.normal(i + 1, 0.03, len(y))`) purely for visual separation. This does not modify the underlying energies, boxplot statistics, or reported numerical results.

### Running Experiments

**Hardware execution policy**: All experiment scripts require a valid TQP hardware token.

```bash
# Fig. 2: C2 ground-state PEC (16 qubits, cutting mode)
python experiments/c2_ground_state.py --token YOUR_TQP_TOKEN

# Fig. 3: C2 excited-state PEC (9 states)
python experiments/c2_excited_states.py --token YOUR_TQP_TOKEN

# Fig. 4: Cr2 active-space scaling
python experiments/cr2_active_space.py --token YOUR_TQP_TOKEN

# Fig. 5: Cr2 basis-set convergence
python experiments/cr2_basis_set.py --token YOUR_TQP_TOKEN

# Fig. 6: Cr2 84-qubit circuit cutting (with optional error mitigation)
python experiments/cr2_1p5A_cutting.py --token YOUR_TQP_TOKEN
python experiments/cr2_1p5A_cutting.py --token YOUR_TQP_TOKEN --mitigation fem

# Fig. 7: Benzene excitations (12 qubits, direct hardware mode)
python experiments/benzene_excitations.py --token YOUR_TQP_TOKEN
```

### Error Mitigation

The `cr2_1p5A_cutting.py` experiment supports configurable error mitigation via `--mitigation`:
- `none` (default): No mitigation
- `fem`: FEM-inspired readout correction from per-qubit F0/F1 calibration rates. Applied at the fragment-counts level with fragment-size-aware mitigators.
- `zne`: Heuristic expectation-value-level correction using `ZeroNoiseExtrapolator.extrapolate()` with synthetic scale points.
- `cdr`: Heuristic expectation-value-level linear correction `E_ideal = a * E_noisy + b` with fixed coefficients unless external training data is provided.

The FEM mitigation operates at the counts level (applied to each fragment's measurement counts before reconstruction). ZNE and CDR operate at the expectation-value level (applied to each reconstructed Pauli expectation value via the `expval_mitigator` callback on `CuttingReconstructor`). This two-level architecture ensures each mitigation method operates at its natural abstraction level.

### Running Tests

```bash
pytest tests/ -v
```

### Programmatic Usage

```python
from pyscf import gto
from vqe_mcpdft import hamiltonian_from_pyscf, CASCIAnsatz, SelfConsistentVQEMCPDFT
from vqe_mcpdft import MCPDFTEnergy, RDMMeasurement, OrbitalRotationCircuit
from vqe_mcpdft.tqp_backend import TQPBackend
from quantum_circuit_cutting.cutting_dispatch import CuttingDispatcher

# Build molecule and qubit Hamiltonian
mol = gto.Mole(atom="C 0 0 0; C 0 0 1.2425", basis="cc-pvtz", spin=0).build()
ham, n_qubits = hamiltonian_from_pyscf(mol, active_space=(8, 8))

# Set up VQE-MC-PDFT components
ansatz = CASCIAnsatz(n_qubits, n_electrons=8, n_layers=2)
tqp = TQPBackend(token="YOUR_TQP_TOKEN")
dispatcher = CuttingDispatcher(max_device_qubits=13, tqp_backend=tqp)

energy_fn_factory = dispatcher.make_energy_fn_factory(n_qubits, 8, ham)
pauli_expval_fn = dispatcher.make_pauli_expval_fn(n_qubits, 8, ham)

# Run self-consistent loop (requires MCPDFTEnergy setup with PySCF grids)
sc = SelfConsistentVQEMCPDFT(
    ansatz=ansatz,
    mcpdft=MCPDFTEnergy(...),
    rdm_measurer=RDMMeasurement(n_qubits),
    orbital_rotator=OrbitalRotationCircuit(n_qubits // 2),
    energy_fn_factory=energy_fn_factory,
    pauli_expval_fn=pauli_expval_fn,
)
```

### Circuit Cutting for Large Active Spaces

```python
from quantum_circuit_cutting import PartitionedVQEMCPDFT, CircuitPartitioner

symmetries = CircuitPartitioner.cr2_symmetry_labels(n_spatial_orbitals=42)
solver = PartitionedVQEMCPDFT(
    n_qubits=84, n_electrons=48, max_fragment_qubits=13,
    use_sampling=True, n_samples=1000,
)
result = solver.run(hamiltonian=ham, orbital_symmetries=symmetries)
```

## Hardware

- **Quantum hardware**: Tencent Tianji-S2 superconducting quantum processor (13 qubits). Calibration data in `data/tianji_s2_calibration.json`.
  - Average T1 = 83.9 us, T2 = 45.4 us
  - Average single-qubit gate error = 6.9 x 10^-4
  - Average CZ gate error = 0.007
  - 10,000 shots per expectation value measurement
- **Development-only utilities**: Test-only simulation helpers exist under `tests/`; manuscript workflows are hardware-only via TQP.

## Key Results

| System | Active Space | Method | Key Result |
|--------|-------------|--------|------------|
| C2 | (8e, 8o) | VQE-MC-PDFT | Req MAE = 0.006 A vs experiment |
| C2 | (8e, 8o) | VQE-MC-PDFT | Excitation energy MAE = 0.10 eV |
| Cr2 | (48e, 42o) | VQE-MC-PDFT + cutting | E_mean = -2086.4371 Ha (84 qubits) |
| Benzene | (6e, 6o) | VQE-MC-PDFT | Excitation MAE = 0.048 eV vs TBE |

## Citation

If you use this repository, cite the accompanying manuscript:

**Multiconfiguration Pair-Density Functional Theory Calculations of Low-lying States of Complex Chemical Systems with Quantum Computers**

For repository-based reproduction, also reference this codebase in the methods or data-availability statement as the implementation used to generate the reported figures and tables.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Note on Reproducibility

All experiment scripts use fixed random seeds (`np.random.seed(42)`) for classical optimizer reproducibility. The reviewer plotting scripts also keep fixed seeds where stochastic horizontal jitter is used only for visualization of dense scatter points in boxplots. Results obtained on quantum hardware (Tianji-S2) may fluctuate within reported error bars due to shot noise and temporal calibration variation.
