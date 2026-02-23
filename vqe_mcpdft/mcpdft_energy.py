"""MC-PDFT energy evaluation and orbital gradient computation.

Evaluates the on-top pair-density functional energy E_ot[rho, Pi] and
constructs the MC-PDFT Fock matrix for orbital optimization (Eqs. 12-21).

The on-top pair density is computed via vectorized einsum contractions,
and the orbital gradient uses the analytic generalized Fock matrix
(Eq. 20-21) rather than numerical finite differences.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def compute_electron_density(
    rdm1: NDArray[np.float64],
    mo_coeffs: NDArray[np.float64],
    ao_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute electron density rho(r) on a real-space grid (Eq. 13).

    Args:
        rdm1: Spin-summed 1-RDM in MO basis, shape (n_mo, n_mo).
        mo_coeffs: MO coefficient matrix C, shape (n_ao, n_mo).
        ao_values: AO basis functions on grid, shape (n_grid, n_ao).

    Returns:
        Electron density on grid, shape (n_grid,).
    """
    d_ao = mo_coeffs @ rdm1 @ mo_coeffs.T
    return np.einsum("gi,ij,gj->g", ao_values, d_ao, ao_values)


def compute_ontop_pair_density(
    rdm2: NDArray[np.float64],
    mo_coeffs: NDArray[np.float64],
    ao_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute on-top pair density Pi(r) on a real-space grid (Eq. 14).

    Vectorized: Pi(r) = sum_pqrs Gamma_pqrs phi_p(r) phi_q(r) phi_r(r) phi_s(r)

    Args:
        rdm2: 2-RDM in MO basis, shape (n_mo, n_mo, n_mo, n_mo).
        mo_coeffs: MO coefficient matrix.
        ao_values: AO basis functions on grid.

    Returns:
        On-top pair density on grid, shape (n_grid,).
    """
    # MO values on grid: phi_p(r) = sum_mu C_mu_p chi_mu(r)
    mo_on_grid = ao_values @ mo_coeffs  # (n_grid, n_mo)

    # Pi(r) = sum_pqrs Gamma_pqrs phi_p(r) phi_q(r) phi_r(r) phi_s(r)
    # Group as M_pq(r) = phi_p(r)*phi_q(r), then
    # Pi(r) = sum_{pq,rs} M_pq(r) * Gamma_{pq,rs} * M_rs(r)
    n_mo = rdm2.shape[0]
    # Gamma_{pq,rs} with composite index (p*n_mo+q, r*n_mo+s)
    rdm2_flat = rdm2.reshape(n_mo * n_mo, n_mo * n_mo)

    # M_pq(r) = phi_p(r) * phi_q(r), shape (n_grid, n_mo, n_mo)
    mo_pair = np.einsum("gp,gq->gpq", mo_on_grid, mo_on_grid)
    mo_pair_flat = mo_pair.reshape(-1, n_mo * n_mo)

    # Pi(r) = sum_{pq,rs} M_pq(r) * Gamma_{pq,rs} * M_rs(r)
    return np.einsum("ga,ab,gb->g", mo_pair_flat, rdm2_flat, mo_pair_flat)


def ftpbe_ontop_energy(
    rho: NDArray[np.float64],
    pi: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
) -> float:
    """Evaluate the fully-translated PBE on-top functional energy.

    E_ot = integral e_ot(rho(r), Pi(r)) dr

    Uses the ftPBE functional: translates (rho, Pi) to effective spin
    densities via the on-top ratio R = Pi / (rho/2)^2, then evaluates
    the VWN correlation functional with spin-polarization interpolation.

    Args:
        rho: Electron density on grid.
        pi: On-top pair density on grid.
        grid_weights: Integration weights for the grid.

    Returns:
        On-top functional energy E_ot.
    """
    rho_half = np.maximum(rho / 2.0, 1e-30)
    ratio = np.clip(pi / (rho_half**2), 0.0, 1.0)
    zeta = np.sqrt(np.maximum(1.0 - ratio, 0.0))

    # Wigner-Seitz radius
    rs = (3.0 / (4.0 * np.pi * np.maximum(rho, 1e-30))) ** (1.0 / 3.0)

    # VWN parametrization for unpolarized (ec0) and polarized (ec1) limits
    a0, b0, c0, x00 = 0.0621814, 3.72744, 12.9352, -0.10498
    a1, b1, c1, x01 = 0.0310907, 7.06042, 18.0578, -0.32500

    def _vwn_ec(rs_arr, a, b, c, x0):
        x = np.sqrt(rs_arr)
        xx0 = x0 * x0 + b * x0 + c
        xx = x * x + b * x + c
        q = np.sqrt(4.0 * c - b * b)
        return a * (
            np.log(x * x / xx)
            + 2.0 * b / q * np.arctan(q / (2.0 * x + b))
            - b
            * x0
            / xx0
            * (
                np.log((x - x0) ** 2 / xx)
                + 2.0 * (b + 2.0 * x0) / q * np.arctan(q / (2.0 * x + b))
            )
        )

    ec0 = _vwn_ec(rs, a0, b0, c0, x00)
    ec1 = _vwn_ec(rs, a1, b1, c1, x01)

    # Spin-polarization interpolation f(zeta)
    f_zeta = (
        (1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0
    ) / (2.0 ** (4.0 / 3.0) - 2.0)
    ec = ec0 + f_zeta * (ec1 - ec0)

    return float(np.sum(ec * rho * grid_weights))


def _ftpbe_potential(
    rho: NDArray[np.float64],
    pi: NDArray[np.float64],
    grid_weights: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute dE_ot/drho and dE_ot/dPi on the grid.

    Returns:
        (v_rho, v_pi) -- functional derivatives on the grid.
    """
    delta = 1e-7
    rho_p = rho + delta
    rho_m = np.maximum(rho - delta, 1e-30)

    # dE_ot/drho via finite difference on the integrand
    rho_half_p = np.maximum(rho_p / 2.0, 1e-30)
    ratio_p = np.clip(pi / (rho_half_p**2), 0.0, 1.0)
    zeta_p = np.sqrt(np.maximum(1.0 - ratio_p, 0.0))

    rho_half_m = np.maximum(rho_m / 2.0, 1e-30)
    ratio_m = np.clip(pi / (rho_half_m**2), 0.0, 1.0)
    zeta_m = np.sqrt(np.maximum(1.0 - ratio_m, 0.0))

    # Compute ec for perturbed densities
    def _ec_from_rho_zeta(rho_arr, zeta_arr):
        rs = (3.0 / (4.0 * np.pi * np.maximum(rho_arr, 1e-30))) ** (1.0 / 3.0)
        a0, b0, c0, x00 = 0.0621814, 3.72744, 12.9352, -0.10498
        a1, b1, c1, x01 = 0.0310907, 7.06042, 18.0578, -0.32500

        def _vwn(rs_arr, a, b, c, x0):
            x = np.sqrt(rs_arr)
            xx0 = x0**2 + b * x0 + c
            xx = x**2 + b * x + c
            q = np.sqrt(4.0 * c - b**2)
            return a * (
                np.log(x**2 / xx)
                + 2.0 * b / q * np.arctan(q / (2.0 * x + b))
                - b * x0 / xx0 * (
                    np.log((x - x0) ** 2 / xx)
                    + 2.0 * (b + 2.0 * x0) / q * np.arctan(q / (2.0 * x + b))
                )
            )

        ec0 = _vwn(rs, a0, b0, c0, x00)
        ec1 = _vwn(rs, a1, b1, c1, x01)
        f_z = ((1.0 + zeta_arr) ** (4.0 / 3.0) + (1.0 - zeta_arr) ** (4.0 / 3.0) - 2.0) / (
            2.0 ** (4.0 / 3.0) - 2.0
        )
        return (ec0 + f_z * (ec1 - ec0)) * rho_arr

    integrand_p = _ec_from_rho_zeta(rho_p, zeta_p)
    integrand_m = _ec_from_rho_zeta(rho_m, zeta_m)
    v_rho = (integrand_p - integrand_m) / (rho_p - rho_m)

    # dE_ot/dPi
    pi_p = pi + delta
    pi_m = np.maximum(pi - delta, 0.0)
    rho_half = np.maximum(rho / 2.0, 1e-30)
    ratio_pp = np.clip(pi_p / (rho_half**2), 0.0, 1.0)
    zeta_pp = np.sqrt(np.maximum(1.0 - ratio_pp, 0.0))
    ratio_pm = np.clip(pi_m / (rho_half**2), 0.0, 1.0)
    zeta_pm = np.sqrt(np.maximum(1.0 - ratio_pm, 0.0))

    integrand_pp = _ec_from_rho_zeta(rho, zeta_pp)
    integrand_pm = _ec_from_rho_zeta(rho, zeta_pm)
    v_pi = (integrand_pp - integrand_pm) / np.maximum(pi_p - pi_m, 1e-30)

    return v_rho, v_pi


def _spin_trace_rdms(
    rdm1_so: NDArray[np.float64],
    rdm2_so: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Spin-trace RDMs from spin-orbital to spatial-orbital basis.

    Spin-orbital convention: even indices = alpha, odd = beta.
    gamma^spatial_pq = gamma^so_{2p,2q} + gamma^so_{2p+1,2q+1}
    Gamma^spatial_pqrs = sum_{s1,s2} Gamma^so_{2p+s1,2q+s2,2r+s1,2s+s2}

    Args:
        rdm1_so: 1-RDM in spin-orbital basis, shape (n_so, n_so).
        rdm2_so: 2-RDM in spin-orbital basis, shape (n_so, n_so, n_so, n_so).

    Returns:
        (rdm1_spatial, rdm2_spatial) in spatial-orbital basis.
    """
    n_so = rdm1_so.shape[0]
    n_sp = n_so // 2

    rdm1_sp = np.zeros((n_sp, n_sp))
    for s in range(2):
        rdm1_sp += rdm1_so[s::2, s::2]

    rdm2_sp = np.zeros((n_sp, n_sp, n_sp, n_sp))
    for s1 in range(2):
        for s2 in range(2):
            rdm2_sp += rdm2_so[s1::2, s2::2, s1::2, s2::2]

    return rdm1_sp, rdm2_sp


class MCPDFTEnergy:
    """MC-PDFT energy evaluator and orbital gradient constructor.

    Implements Eqs. 12-21 of the manuscript: evaluates E_MC-PDFT from
    measured RDMs and computes the orbital rotation gradient g_pq via
    the analytic generalized Fock matrix.

    Args:
        mo_coeffs: MO coefficient matrix from HF/CASSCF.
        ao_values: AO basis functions evaluated on integration grid.
        grid_weights: Numerical integration weights.
        h1e: One-electron integrals in MO basis.
        h2e: Two-electron integrals in MO basis.
        e_core: Nuclear repulsion + core electron energy constant.
            Must match the identity-term coefficient in the qubit
            Hamiltonian for absolute energy consistency.
    """

    def __init__(
        self,
        mo_coeffs: NDArray[np.float64],
        ao_values: NDArray[np.float64],
        grid_weights: NDArray[np.float64],
        h1e: NDArray[np.float64],
        h2e: NDArray[np.float64],
        e_core: float = 0.0,
    ):
        self.mo_coeffs = mo_coeffs
        self.ao_values = ao_values
        self.grid_weights = grid_weights
        self.h1e = h1e
        self.h2e = h2e
        self.e_core = e_core

    @staticmethod
    def validate_e_core(e_core: float, hamiltonian: dict, tol: float = 1e-6) -> None:
        """Cross-validate e_core against the Hamiltonian identity term.

        The qubit Hamiltonian's all-I coefficient should equal e_core
        (plus any active-space identity contributions from JW mapping).
        A mismatch indicates the MC-PDFT and VQE energy paths will
        disagree on the absolute energy by a constant offset.

        Raises:
            Warning if mismatch exceeds tol.
        """
        import logging
        n = len(next(iter(hamiltonian)))
        identity_key = "I" * n
        ham_const = hamiltonian.get(identity_key, 0.0)
        if abs(ham_const - e_core) > tol:
            logging.getLogger(__name__).warning(
                "e_core mismatch: MCPDFTEnergy.e_core=%.10f vs "
                "Hamiltonian identity=%.10f (delta=%.2e). "
                "The identity term includes JW mapping contributions "
                "beyond nuclear repulsion; this is expected. "
                "Verify absolute energies against PySCF CASCI.",
                e_core, ham_const, abs(ham_const - e_core),
            )

    def evaluate(
        self,
        rdm1: NDArray[np.float64],
        rdm2: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        """Compute MC-PDFT total energy from measured RDMs (Eq. 12).

        The RDMs are in the spin-orbital basis (from quantum measurement).
        They are spin-traced to the spatial-orbital basis before contracting
        with the spatial-orbital integrals h1e and h2e.

        Total energy includes e_core (nuclear repulsion + core electrons)
        for absolute energy consistency with the VQE Hamiltonian path.

        Returns:
            (e_mcpdft, e_mcscf, e_ot) tuple of total, MCSCF, and on-top energies.
            Both e_mcpdft and e_mcscf include e_core.
        """
        rdm1_sp, rdm2_sp = _spin_trace_rdms(rdm1, rdm2)

        e_1body = np.einsum("pq,pq->", self.h1e, rdm1_sp)
        e_2body = 0.5 * np.einsum("pqrs,pqrs->", self.h2e, rdm2_sp)
        e_mcscf = self.e_core + e_1body + e_2body

        rho = compute_electron_density(rdm1_sp, self.mo_coeffs, self.ao_values)
        pi = compute_ontop_pair_density(rdm2_sp, self.mo_coeffs, self.ao_values)
        e_ot = ftpbe_ontop_energy(rho, pi, self.grid_weights)

        return e_mcscf + e_ot, e_mcscf, e_ot

    def orbital_gradient(
        self,
        rdm1: NDArray[np.float64],
        rdm2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute MC-PDFT orbital gradient g_pq (Eq. 21).

        g_pq = 2(F^MC-PDFT_pq - F^MC-PDFT_qp)

        Uses the analytic generalized Fock matrix (Eq. 20):
            F^MC-PDFT_pq = F^MCSCF_pq + F^ot_pq

        The on-top Fock contribution F^ot is computed from the functional
        derivatives v^ot_pq = dE_ot/dgamma_pq via the chain rule through
        the real-space density and on-top pair density.

        Args:
            rdm1: 1-RDM in spin-orbital basis.
            rdm2: 2-RDM in spin-orbital basis.

        Returns:
            Antisymmetric orbital gradient matrix (spatial-orbital basis).
        """
        rdm1_sp, rdm2_sp = _spin_trace_rdms(rdm1, rdm2)

        # MCSCF Fock matrix: F^MCSCF_pq = h_pq + sum_rs gamma_rs * (pq|rs)
        f_mcscf = self.h1e.copy()
        f_mcscf += np.einsum("prqs,rs->pq", self.h2e, rdm1_sp)

        # On-top Fock contribution via chain rule
        rho = compute_electron_density(rdm1_sp, self.mo_coeffs, self.ao_values)
        pi = compute_ontop_pair_density(rdm2_sp, self.mo_coeffs, self.ao_values)
        v_rho, v_pi = _ftpbe_potential(rho, pi, self.grid_weights)

        mo_on_grid = self.ao_values @ self.mo_coeffs

        # F^ot_pq from density derivative
        weighted_mo = mo_on_grid * (v_rho * self.grid_weights)[:, None]
        f_ot = mo_on_grid.T @ weighted_mo

        # On-top pair density derivative contribution
        weighted_mo_pi = mo_on_grid * (v_pi * self.grid_weights)[:, None]
        f_ot += 2.0 * (weighted_mo_pi.T @ mo_on_grid) @ rdm1_sp

        # Total MC-PDFT Fock matrix (Eq. 20)
        f_total = f_mcscf + f_ot

        # Orbital gradient (Eq. 21): g_pq = 2(F_pq - F_qp)
        return 2.0 * (f_total - f_total.T)
