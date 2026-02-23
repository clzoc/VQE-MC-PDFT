"""Tests for MC-PDFT e_core energy consistency and overlap estimation robustness."""
import numpy as np


class TestMCPDFTEcoreConsistency:
    """Verify that MCPDFTEnergy includes e_core in absolute energy."""

    def test_evaluate_includes_e_core(self):
        """e_mcscf and e_mcpdft must include e_core."""
        from vqe_mcpdft.mcpdft_energy import MCPDFTEnergy
        n_sp = 2
        n_so = 2 * n_sp
        mo = np.eye(n_sp)
        ao = np.ones((5, n_sp))
        w = np.ones(5) * 0.2
        h1e = np.diag([0.1, 0.2])
        h2e = np.zeros((n_sp, n_sp, n_sp, n_sp))

        e_core_val = -10.0
        mcpdft = MCPDFTEnergy(mo, ao, w, h1e, h2e, e_core=e_core_val)

        # Trivial RDMs: identity-like 1-RDM, zero 2-RDM
        rdm1 = np.zeros((n_so, n_so))
        rdm1[0, 0] = 1.0
        rdm1[1, 1] = 1.0
        rdm2 = np.zeros((n_so, n_so, n_so, n_so))

        e_mcpdft, e_mcscf, e_ot = mcpdft.evaluate(rdm1, rdm2)
        # Spin-trace: so indices 0,1 -> spatial orbital 0 (alpha+beta)
        # rdm1_sp[0,0] = 2.0, so e_1body = h1e[0,0]*2.0 = 0.2
        assert e_mcscf < 0, f"e_mcscf={e_mcscf} should be negative (e_core=-10)"
        assert abs(e_mcscf - (e_core_val + 0.2)) < 1e-10

    def test_evaluate_without_e_core_is_zero_offset(self):
        """With e_core=0 (default), e_mcscf is active-space-only."""
        from vqe_mcpdft.mcpdft_energy import MCPDFTEnergy
        n_sp = 2
        n_so = 2 * n_sp
        mo = np.eye(n_sp)
        ao = np.ones((5, n_sp))
        w = np.ones(5) * 0.2
        h1e = np.diag([0.1, 0.2])
        h2e = np.zeros((n_sp, n_sp, n_sp, n_sp))

        mcpdft_no_core = MCPDFTEnergy(mo, ao, w, h1e, h2e)
        mcpdft_with_core = MCPDFTEnergy(mo, ao, w, h1e, h2e, e_core=-5.0)

        rdm1 = np.zeros((n_so, n_so))
        rdm1[0, 0] = 1.0
        rdm1[1, 1] = 1.0
        rdm2 = np.zeros((n_so, n_so, n_so, n_so))

        e1, _, _ = mcpdft_no_core.evaluate(rdm1, rdm2)
        e2, _, _ = mcpdft_with_core.evaluate(rdm1, rdm2)
        assert abs((e2 - e1) - (-5.0)) < 1e-10

    def test_validate_e_core_warns_on_mismatch(self, caplog):
        """validate_e_core should log warning when e_core != Hamiltonian identity."""
        from vqe_mcpdft.mcpdft_energy import MCPDFTEnergy
        import logging
        ham = {"IIII": -10.5, "ZIII": 0.3}
        # No warning when close
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            MCPDFTEnergy.validate_e_core(-10.5, ham, tol=1e-6)
            assert "e_core mismatch" not in caplog.text
        # Should warn when mismatch
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            MCPDFTEnergy.validate_e_core(-999.0, ham, tol=1e-6)
            assert "e_core mismatch" in caplog.text

    def test_e_core_backward_compatible(self):
        """MCPDFTEnergy without e_core arg should still work (default 0)."""
        from vqe_mcpdft.mcpdft_energy import MCPDFTEnergy
        n_sp = 2
        mo = np.eye(n_sp)
        ao = np.ones((3, n_sp))
        w = np.ones(3) / 3
        h1e = np.zeros((n_sp, n_sp))
        h2e = np.zeros((n_sp, n_sp, n_sp, n_sp))
        mcpdft = MCPDFTEnergy(mo, ao, w, h1e, h2e)
        assert mcpdft.e_core == 0.0


class TestOverlapEstimation:
    """Verify symmetric Pauli overlap estimation correctness and bounds."""

    def test_overlap_clamped_to_unit_interval(self):
        """Estimated overlap must be in [0, 1]."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
        from c2_excited_states import _estimate_overlap_symmetric

        # Mock pauli_expval_fn that returns constant values
        def mock_expval_fn(params, orb_angles, pauli_labels):
            return {p: 0.5 for p in pauli_labels}

        params1 = np.zeros(4)
        params2 = np.zeros(4)
        overlap = _estimate_overlap_symmetric(
            mock_expval_fn, params1, params2, None, 4,
            initial_samples=50, max_samples=50,
        )
        assert 0.0 <= overlap <= 1.0

    def test_identical_states_high_overlap(self):
        """Same params should give overlap close to 1 (with mock)."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
        from c2_excited_states import _estimate_overlap_symmetric

        # Mock: same state -> same expectation values -> product = val^2
        # For random Paulis on a state, <P>^2 averages to ~1/2^n for n qubits
        # but with our mock returning 1.0 for all, overlap = 1.0
        def mock_same_state(params, orb_angles, pauli_labels):
            return {p: 1.0 for p in pauli_labels}

        overlap = _estimate_overlap_symmetric(
            mock_same_state, np.zeros(4), np.zeros(4), None, 4,
            initial_samples=50, max_samples=50,
        )
        assert overlap == 1.0

    def test_orthogonal_states_low_overlap(self):
        """Orthogonal states should give overlap near 0."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
        from c2_excited_states import _estimate_overlap_symmetric

        # Mock: one state returns +1, other returns -1 for half the Paulis
        # Product alternates +1 and -1, mean ≈ 0
        call_count = [0]
        def mock_orthogonal(params, orb_angles, pauli_labels):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return {p: (1.0 if i % 2 == 0 else -1.0) for i, p in enumerate(pauli_labels)}
            else:
                return {p: (-1.0 if i % 2 == 0 else 1.0) for i, p in enumerate(pauli_labels)}

        overlap = _estimate_overlap_symmetric(
            mock_orthogonal, np.zeros(4), np.ones(4), None, 4,
            initial_samples=200, max_samples=200,
        )
        # Products alternate -1 and +1, average ≈ 0, so overlap should be near 0
        assert 0.0 <= overlap <= 0.1

    def test_negative_overlap_clamped(self):
        """Negative raw overlap must be clamped to 0."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
        from c2_excited_states import _estimate_overlap_symmetric

        # Mock that produces negative products
        call_count = [0]
        def mock_negative(params, orb_angles, pauli_labels):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return {p: 1.0 for p in pauli_labels}
            else:
                return {p: -1.0 for p in pauli_labels}

        overlap = _estimate_overlap_symmetric(
            mock_negative, np.zeros(4), np.ones(4), None, 4,
            initial_samples=50, max_samples=50,
        )
        assert overlap == 0.0
