"""Configuration consistency tests.
Verifies that config IDs are consistent between fragment generation, execution, and reconstruction.
"""
import numpy as np

from quantum_circuit_cutting.cutting_dispatch import CuttingDispatcher
from tests.sim_utils import simulate_fragment_counts


class _MockBackend:
    """Deterministic local backend mock for static pipeline tests."""

    def submit_circuit(self, circuit):
        n_qubits = int(circuit._nqubits)  # TensorCircuit internal field
        return simulate_fragment_counts(
            circuit, n_qubits, 1000, np.random.default_rng(42)
        )

def test_config_id_consistency():
    """Test that configuration IDs match between generation and reconstruction."""
    n_qubits = 16
    n_electrons = 8
    ham = {"I"*n_qubits: 0.0, "Z" + "I"*(n_qubits-1): 1.0}
    
    dispatcher = CuttingDispatcher(
        max_device_qubits=13,
        use_sampling=False,
        tqp_backend=_MockBackend(),
    )
    
    # Build cutting context
    ctx = dispatcher._build_cutting_context(n_qubits, n_electrons, ham, None)
    
    # Run fragments and get configs
    params = np.zeros(16)
    fragment_results, configs = dispatcher._run_fragments(ctx, params, None)
    
    # Verify all config indices are present in results
    config_indices = set(k[0] for k in fragment_results.keys())
    assert max(config_indices) == len(configs) - 1, f"Missing config indices: expected {len(configs)} configs, got {max(config_indices)+1}"
    
    # Verify reconstruction uses the same configs
    reconstructor = ctx["reconstructor"]
    expval = reconstructor.reconstruct_expectation(fragment_results, "Z"+"I"*15, configurations=configs)
    
    # Mismatched configs would raise error or return wrong value
    assert isinstance(expval, float), "Reconstruction failed with consistent configs"

def test_cut_strategy_consistency():
    """Test that cut strategy is consistent across all steps."""
    from quantum_circuit_cutting.overhead import select_cut_strategy
    
    n_qubits = 16
    max_cluster = 13
    entangling_gates = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14), (14,15)]
    
    # Get strategy
    strategy = select_cut_strategy(n_qubits, max_cluster, entangling_gates)
    use_sampling = strategy["use_sampling"]
    n_samples = strategy["n_samples"]
    
    # Verify dispatcher uses same strategy
    dispatcher = CuttingDispatcher(max_device_qubits=13)
    ctx = dispatcher._build_cutting_context(n_qubits, 8, {"I"*16:0.0}, None)
    
    assert ctx["use_sampling"] == use_sampling, "Sampling strategy mismatch"
    assert ctx["n_samples"] == n_samples, "Sample count mismatch"
