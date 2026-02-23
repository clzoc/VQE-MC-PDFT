# VQE-MC-PDFT

**Multiconfiguration Pair-Density Functional Theory Calculations of Ground and Excited States of Complex Chemical Systems with Quantum Computers**

*Zhanou Liu, Yuhao Chen, Yingjin Ma, Xiao He, Yuxin Deng*

Code repository accompanying the PNAS manuscript. This package implements a hybrid quantum-classical framework combining the Variational Quantum Eigensolver (VQE) with Multiconfiguration Pair-Density Functional Theory (MC-PDFT), featuring quantum circuit cutting for scalable active-space simulations and FEM-inspired readout error mitigation.

## Method Overview

VQE-MC-PDFT decouples electron correlation into two components:

1. **Static correlation** is captured by a multiconfigurational wavefunction prepared on a quantum processor via a particle-number-conserving CASCI ansatz.
2. **Dynamic correlation** is recovered classically through the fully-translated PBE (ftPBE) on-top pair-density functional, evaluated from the measured 1- and 2-RDMs.

Self-consistent orbital optimization is achieved by computing the MC-PDFT orbital gradient (generalized Fock matrix), compiling the resulting orbital-rotation generator into a Trotterized quantum circuit, and appending it to the ansatz. The framework is validated on C2, Cr2, and benzene using the Tencent Tianji-S2 superconducting quantum processor.

## Repository Structure

```
Code-for-PNAS/
├── vqe_mcpdft/                    # Core VQE-MC-PDFT implementation
│   ├── hamiltonian.py             #   Fermionic Hamiltonian + Jordan-Wigner mapping (Eqs. 2-5)
│   ├── ansatz.py                  #   CASCI particle-conserving ansatz (Eqs. 6-8)
│   ├── orbital_rotation.py        #   Orbital rotation via Suzuki-Trotter (Eqs. 16-27)
│   ├── mcpdft_energy.py           #   MC-PDFT ftPBE on-top functional + analytic Fock matrix (Eqs. 12-21)
│   ├── rdm.py                     #   Full 1-RDM / 2-RDM Pauli measurement schedules (Eqs. 10-11)
│   ├── vqe_solver.py              #   Inner-loop VQE with Adam optimizer
│   ├── self_consistent.py         #   Outer-loop self-consistent driver (Fig. S1)
│   └── tqp_backend.py             #   Tencent Tianji-S2 SDK interface with calibration-aware qubit selection
├── quantum_circuit_cutting/       # Circuit cutting framework (SI Section S1.2)
│   ├── channel_decomposition.py   #   8-channel identity decomposition (SI Eqs. S19-S27)
│   ├── partition.py               #   Spectral bisection + KL refinement + symmetry-guided partitioning
│   ├── fragment_circuits.py       #   Fragment circuit generation with state prep/measurement
│   ├── reconstruction.py          #   Weighted reconstruction with QWC observable grouping
│   ├── overhead.py                #   Sampling overhead budget and cut strategy selection
│   ├── cutting_dispatch.py        #   Auto-routing dispatcher (direct vs cutting execution)
│   └── cutting_vqe.py             #   Partitioned VQE-MC-PDFT driver (Fig. 6)
├── error_mitigation/              # Error mitigation protocols
│   ├── fem_readout.py             #   FEM-inspired multi-stage iterative readout correction (Eq. 33)
│   ├── zne.py                     #   Zero-Noise Extrapolation via local unitary folding
│   └── clifford_fitting.py        #   Clifford Data Regression error mitigation
├── experiments/                   # Reproduction scripts for all figures and tables
│   ├── c2_ground_state.py         #   Fig. 2: C2 ground-state PEC
│   ├── c2_excited_states.py       #   Fig. 3: C2 excited-state PEC (9 states)
│   ├── cr2_active_space.py        #   Fig. 4: Cr2 active-space scaling
│   ├── cr2_basis_set.py           #   Fig. 5: Cr2 basis-set convergence
│   ├── cr2_1p5A_cutting.py        #   Fig. 6: Cr2 84-qubit circuit cutting (48e,42o)
│   ├── benzene_excitations.py     #   Fig. 7: Benzene pi->pi* excitations
│   └── generate_figures.py        #   Generate all publication figures from data
├── data/                          # Reference data (SI Tables S4-S10)
│   ├── c2_ground_state_pec.csv    #   Table S4: C2 ground-state energies
│   ├── c2_excited_state_pec.csv   #   Table S5: C2 excited-state energies
│   ├── c2_excitation_energies.csv #   Table S6: C2 excitation energies
│   ├── cr2_active_space.csv       #   Table S7: Cr2 active-space comparison
│   ├── cr2_basis_set.csv          #   Table S8: Cr2 basis-set dependence
│   ├── cr2_boxplot_raw.csv        #   Table S9: Cr2 1.50A raw data (50 runs x 4 qubit counts)
│   ├── benzene_excitations.csv    #   Table S10: Benzene excitation energies
│   └── tianji_s2_calibration.json #   Tables S2-S3: Hardware calibration data
├── tests/                         # Unit and integration tests
├── pyproject.toml                 # PEP 621 project metadata
├── requirements.txt               # Exact version pins for reproducibility
├── CITATION.cff                   # Machine-readable citation
├── LICENSE                        # MIT License
└── .zenodo.json                   # Zenodo archival metadata
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
cd Code-for-PNAS
pip install -r requirements.txt   # exact version pins for reproducibility
pip install -e ".[dev]"
```

### Prerequisites

- Python >= 3.9
- NumPy, SciPy, PySCF (>= 2.4), TensorCircuit (>= 0.12), NetworkX, pandas, matplotlib
- For quantum hardware execution: Tencent Quantum Platform SDK token

## Reproducing Results

### Generate Figures from Pre-computed Data

All numerical data from the manuscript (SI Tables S4-S10) is provided in `data/`. To generate publication figures without re-running quantum computations:

```bash
python experiments/generate_figures.py
```

Figures are saved to `figures/`.

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
python experiments/cr2_1p5A_cutting.py --token YOUR_TQP_TOKEN --n-qubits 52

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

```bibtex
@article{liu2025vqemcpdft,
  title   = {Multiconfiguration Pair-Density Functional Theory Calculations
             of Ground and Excited States of Complex Chemical Systems
             with Quantum Computers},
  author  = {Liu, Zhanou and Chen, Yuhao and Ma, Yingjin and He, Xiao and Deng, Yuxin},
  journal = {Proceedings of the National Academy of Sciences},
  year    = {2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'tensorcircuit'` | Install exact versions: `pip install -r requirements.txt` |
| `pyscf.lib.exceptions.BasisNotFoundError` | Ensure PySCF >= 2.4 and the basis set name matches exactly (e.g., `cc-pvtz`, not `cc-pVTZ`) |
| TQP SDK `AuthenticationError` | Verify your token in `token-for-quantum-tencent` and call `apis.set_token(token)` |
| `MemoryError` for large active spaces (e.g., Cr2 (48e,42o)) | Use circuit cutting (`cr2_1p5A_cutting.py`) and reduce shot budgets/group counts during debugging. |
| VQE not converging | Increase `max_outer_iter`, reduce `orbital_step_size`, or try different initial parameters via `ansatz.initial_params(seed=N)` |
| Results differ slightly from paper tables | Hardware results can fluctuate within error bars due to shot noise and device drift. Random seeds (`seed=42`) only control classical optimizer randomness. |

## Note on Reproducibility

All experiment scripts use fixed random seeds (`np.random.seed(42)`) for classical optimizer reproducibility. Results obtained on quantum hardware (Tianji-S2) may fluctuate within reported error bars due to shot noise and temporal calibration variation.

## Technical Optimizations

### Existing Optimizations

The following optimizations are present in the baseline implementation:

- **Symmetry-guided partitioning** (`partition.py`). When MO symmetry labels are available (e.g., D∞h for Cr2), qubits sharing the same irreducible representation are grouped into the same cluster before any graph-based refinement, reducing inter-cluster entangling gates. Reference: Peng et al., PRL 125, 150504 (2020).
- **QWC observable grouping** (`reconstruction.py`). Pauli strings that commute qubit-wise (QWC) are measured in a single circuit execution, reducing the number of distinct measurement bases. Greedy graph-coloring is used for grouping. Reference: Verteletskyi et al., J. Chem. Phys. 152, 124114 (2020).
- **Shared sampling configurations for gradient estimation** (`cutting_vqe.py`). The parameter-shift gradient uses the same set of sampled channel configurations for both E(θ+s) and E(θ−s), ensuring correlated noise cancellation and lower gradient variance.
- **Hardware-only execution policy** (`experiments/`). All experiment scripts require a valid TQP hardware token for execution, ensuring all results are generated on real Tianji-S2 quantum hardware.

### New Optimizations

The following optimizations were added in this revision:

- **Best-first search cut finding** (`partition.py`, `strategy="best_first"`). A priority-queue-based search explores partial qubit-to-cluster assignments ordered by an admissible lower bound on the cut count. Branches whose lower bound exceeds the best complete solution found so far are pruned. When symmetry labels are available, the heuristic biases assignment toward same-symmetry grouping. The spectral-bisection solution seeds the initial upper bound. Applicable to systems where the spectral heuristic yields suboptimal partitions (e.g., irregular interaction graphs). Inspired by Tang et al., "CutQC" (2021) and the Qiskit Circuit Knitting Toolbox `automated_cut_finding` module.
- **Measurement-basis reuse** (`reconstruction.py`, `rdm.py`, `cutting_vqe.py`). A `MeasurementCache` stores fragment measurement results keyed by `(config_idx, cluster_idx, basis_signature)`. When multiple QWC groups require the same measurement rotations on the same fragment, the circuit is executed only once and the cached result is reused. The `RDMMeasurement.grouped_measurement_bases()` method pre-groups Pauli bases by QWC compatibility to facilitate this reuse. Applicable to RDM measurement in the self-consistent MC-PDFT loop, where the number of unique measurement bases is typically smaller than the number of QWC groups. Reference: Peng et al., PRL 125, 150504 (2020).
- **Exponential-backoff hardware retry** (`tqp_backend.py`, `cutting_vqe.py`). `TQPBackend.submit_circuit` and `submit_batch` retry up to 3 times with exponential backoff (2 s, 4 s, 8 s) on transient network errors, queue timeouts, or hardware faults. A `_validate_counts` check rejects empty or all-zero measurement results and triggers a retry. After exhausting retries, the sub-batch returns empty results and logs a warning rather than aborting the VQE iteration, enabling graceful degradation during long-running experiments.
- **General QWC observable construction** (`fragment_circuits.py`, `cutting_dispatch.py`). Fragment measurement rotations now use the "most general observable" across the entire QWC group (via `measurement_basis_key`), not just the first/representative Pauli string. This prevents silent measurement errors when the first Pauli has `I` on a qubit but later terms in the same group require `X` or `Y` rotations.
- **QWC-group-aware energy reconstruction** (`cutting_vqe.py`, `cutting_dispatch.py`). Both `PartitionedVQEMCPDFT` and `CuttingDispatcher` now split Hamiltonian terms into QWC groups and generate separate fragment circuits (with correct measurement bases) for each group. Reconstruction uses only each group's own measurement data. The parameter-shift gradient path uses the same group-aware evaluation.
- **Reconstruction integrity checks** (`reconstruction.py`). `CuttingReconstructor.validate_fragment_results()` checks expected result cardinality (n_configs × n_clusters) and raises `ValueError` on missing fragment keys. Pauli string length is validated against total qubits.
- **RDM stochastic policy** (`rdm.py`). For n_qubits > 13, stochastic RDM mode is enabled by default (overridable). A deterministic `numpy.random.Generator` is threaded through the class for reproducibility. Measurement burden warnings are emitted when projected Pauli term count exceeds threshold.
- **Mitigation pipeline wiring** (`cr2_1p5A_cutting.py`). FEM is wired at counts level from per-qubit readout calibration rates and applied per fragment size. ZNE/CDR are wired at expectation-value level via `expval_mitigator`. Current ZNE/CDR implementations are heuristic and should be interpreted accordingly.
- **Hamiltonian term screening** (`cutting_vqe.py`, `cutting_dispatch.py`). Terms with |coefficient| < 1e-10 are dropped before partitioning. Projected circuit-shot budget is computed and execution aborts if infeasible (>10M circuits in exact mode).
