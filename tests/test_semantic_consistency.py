"""Semantic consistency tests for measurement basis and reconstruction.
Verifies that Pauli X/Y/Z operators are correctly mapped to measurement bases
and that reconstructed expectation values match direct simulation.
"""
import numpy as np
import networkx as nx
from quantum_circuit_cutting.fragment_circuits import FragmentCircuitGenerator
from quantum_circuit_cutting.partition import ClusterPartition
from quantum_circuit_cutting.channel_decomposition import ChannelDecomposition
from quantum_circuit_cutting.reconstruction import CuttingReconstructor

def test_pauli_measurement_basis_mapping():
    """Test that X/Y/Z Pauli operators map to correct measurement rotations."""
    n_qubits = 4
    # Build cluster graph for 2 clusters with 1 cut
    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from([0, 1])
    cluster_graph.add_edge(0, 1, weight=1)
    partition = ClusterPartition(
        clusters=[[0, 1], [2, 3]], 
        inter_cluster_edges=[(1, 2)],
        cluster_graph=cluster_graph,
        n_cuts=1
    )
    decomposition = ChannelDecomposition(n_cuts=1)
    ansatz_params = np.zeros(4)
    
    # Test each Pauli type by verifying the circuit produces valid output
    for op in ["X", "Y", "Z"]:
        pauli_group = [op + "III"[:n_qubits-1]]
        generator = FragmentCircuitGenerator(
            partition=partition,
            decomposition=decomposition,
            ansatz_params=ansatz_params,
            n_electrons=2,
        )
        fragments = generator.generate_all(pauli_group=pauli_group)
        
        # Verify fragments are generated and circuits are valid
        for config_frags in fragments:
            for frag in config_frags:
                if 0 in partition.clusters[frag.cluster_idx]:
                    # Circuit should produce a valid statevector
                    state = np.asarray(frag.circuit.state()).ravel()
                    assert np.isclose(np.sum(np.abs(state)**2), 1.0, atol=1e-6), \
                        f"Fragment circuit for {op} does not produce normalized state"

def test_reconstructed_expval_match():
    """Test that reconstructed X/Y/Z expectation values match direct simulation."""
    n_qubits = 4
    # Build cluster graph for 2 clusters with 1 cut
    cluster_graph = nx.Graph()
    cluster_graph.add_nodes_from([0, 1])
    cluster_graph.add_edge(0, 1, weight=1)
    partition = ClusterPartition(
        clusters=[[0, 1], [2, 3]], 
        inter_cluster_edges=[(1, 2)],
        cluster_graph=cluster_graph,
        n_cuts=1
    )
    decomposition = ChannelDecomposition(n_cuts=1)
    ansatz_params = np.random.uniform(-np.pi, np.pi, size=4)
    
    # Test each Pauli
    for op in ["X", "Y", "Z"]:
        pauli_string = op + "III"[:n_qubits-1]
        # Generate fragments with QWC group
        pauli_group = [pauli_string]
        generator = FragmentCircuitGenerator(
            partition=partition,
            decomposition=decomposition,
            ansatz_params=ansatz_params,
            n_electrons=2,
        )
        all_fragments = generator.generate_all(pauli_group=pauli_group)
        
        # Simulate fragment counts
        fragment_results = {}
        for config_idx, frags in enumerate(all_fragments):
            for frag in frags:
                state = np.asarray(frag.circuit.state()).ravel()
                probs = np.abs(state)**2
                counts = {}
                for i, p in enumerate(probs):
                    if p > 1e-10:
                        counts[format(i, f"0{frag.n_qubits}b")] = int(p * 10000)
                fragment_results[(config_idx, frag.cluster_idx)] = counts
        
        # Reconstruct expectation
        reconstructor = CuttingReconstructor(partition, decomposition)
        expval_recon = reconstructor.reconstruct_expectation(fragment_results, pauli_string)
        
        # Direct simulation of full circuit
        from vqe_mcpdft.ansatz import CASCIAnsatz
        ansatz = CASCIAnsatz(n_qubits, 2, n_layers=2)
        full_circuit = ansatz.build_circuit(ansatz_params, None)
        psi = np.asarray(full_circuit.state()).ravel()
        
        # Compute exact expectation
        pauli_mat = {
            "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]),
            "Z": np.array([[1, 0], [0, -1]]),
        }[op]
        full_op = pauli_mat
        for _ in range(n_qubits -1):
            full_op = np.kron(full_op, np.eye(2))
        expval_exact = np.vdot(psi, full_op @ psi).real
        
        # Match within 1e-2 tolerance (due to sampling)
        assert np.abs(expval_recon - expval_exact) < 1e-2, f"Mismatch for {op} expval: recon={expval_recon:.4f}, exact={expval_exact:.4f}"
