"""Identity channel decomposition for quantum circuit cutting.

Implements the 8-channel quasi-probability decomposition of the identity
channel (SI Eqs. S19-S27) used to sever inter-cluster qubit wires.
Each channel consists of a measurement operator O_i on the source qubit
and a state preparation rho_i on the target qubit, with coefficient c_i.

Reference: Peng et al., PRL 125, 150504 (2020).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


class PauliBasis(Enum):
    """Pauli measurement basis for channel decomposition."""

    I = "I"  # noqa: E741
    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass(frozen=True)
class Channel:
    """Single channel in the identity decomposition.

    Attributes:
        index: Channel index (0-7).
        observable: Pauli operator O_i measured on the source qubit.
        prep_state: Label for the prepared state rho_i on the target qubit.
        prep_vector: Statevector of the prepared state.
        coefficient: Quasi-probability coefficient c_i.
    """

    index: int
    observable: PauliBasis
    prep_state: str
    prep_vector: NDArray[np.complex128]
    coefficient: float


def decompose_identity_channel() -> List[Channel]:
    """Return the 8-channel decomposition of the single-qubit identity.

    Implements SI Eqs. S19-S27:
        A = [Tr(A*O_i) * rho_i] summed over i=0..7

    Each channel (O_i, rho_i, c_i) satisfies:
        I = sum_i c_i * Tr(. * O_i) * rho_i

    Returns:
        List of 8 Channel objects.
    """
    s2 = 1.0 / np.sqrt(2.0)

    ket_0 = np.array([1.0, 0.0], dtype=np.complex128)
    ket_1 = np.array([0.0, 1.0], dtype=np.complex128)
    ket_plus = np.array([s2, s2], dtype=np.complex128)
    ket_minus = np.array([s2, -s2], dtype=np.complex128)
    ket_plus_i = np.array([s2, s2 * 1j], dtype=np.complex128)
    ket_minus_i = np.array([s2, -s2 * 1j], dtype=np.complex128)

    return [
        # Eq. S20: O=I, rho=|0><0|, c=+1/2
        Channel(0, PauliBasis.I, "|0>", ket_0, +0.5),
        # Eq. S21: O=I, rho=|1><1|, c=+1/2
        Channel(1, PauliBasis.I, "|1>", ket_1, +0.5),
        # Eq. S22: O=X, rho=|+><+|, c=+1/2
        Channel(2, PauliBasis.X, "|+>", ket_plus, +0.5),
        # Eq. S23: O=X, rho=|-><-|, c=-1/2
        Channel(3, PauliBasis.X, "|->", ket_minus, -0.5),
        # Eq. S24: O=Y, rho=|+i><+i|, c=+1/2
        Channel(4, PauliBasis.Y, "|+i>", ket_plus_i, +0.5),
        # Eq. S25: O=Y, rho=|-i><-i|, c=-1/2
        Channel(5, PauliBasis.Y, "|-i>", ket_minus_i, -0.5),
        # Eq. S26: O=Z, rho=|0><0|, c=+1/2
        Channel(6, PauliBasis.Z, "|0>", ket_0, +0.5),
        # Eq. S27: O=Z, rho=|1><1|, c=-1/2
        Channel(7, PauliBasis.Z, "|1>", ket_1, -0.5),
    ]


class ChannelDecomposition:
    """Manages the quasi-probability decomposition for circuit cutting.

    For K cut wires, the total number of channel configurations is 8^K.
    The sampling overhead (kappa) for a single wire cut is 4.0.

    Note:
        Current implementation uses static wire-cut coefficients only.
        The ``GateQPDDecomposition`` class provides a framework for
        parameter-dependent gate-level QPD but is not yet integrated
        into the cutting pipeline.

    Args:
        n_cuts: Number of inter-cluster wire cuts K.
    """

    def __init__(self, n_cuts: int):
        self.n_cuts = n_cuts
        self.channels = decompose_identity_channel()
        self.n_channels = len(self.channels)

    @property
    def kappa(self) -> float:
        """Sampling overhead: kappa = sum(|c_i|) per cut, raised to K."""
        single_kappa = sum(abs(ch.coefficient) for ch in self.channels)
        return single_kappa**self.n_cuts

    @property
    def n_configurations(self) -> int:
        """Total number of channel configurations: 8^K."""
        return self.n_channels**self.n_cuts

    def enumerate_configurations(
        self,
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """Enumerate all 8^K configurations with their coefficients.

        Returns:
            List of (channel_indices, combined_coefficient) tuples.
            channel_indices is a K-tuple of channel indices (0-7).
        """
        configs: List[Tuple[Tuple[int, ...], float]] = []
        for flat_idx in range(self.n_configurations):
            indices = self._index_to_channels(flat_idx)
            coeff = 1.0
            for ch_idx in indices:
                coeff *= self.channels[ch_idx].coefficient
            configs.append((indices, coeff))
        return configs

    def sample_configurations(
        self, n_samples: int, rng: np.random.Generator | None = None
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """Sample configurations from the quasi-probability distribution.

        For large K, exact enumeration of 8^K is infeasible. This method
        samples configurations proportional to |c_i1 * ... * c_iK| and
        returns the sign-weighted coefficients.

        Args:
            n_samples: Number of samples to draw.
            rng: Random number generator.

        Returns:
            List of (channel_indices, signed_weight) tuples.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Per-cut probability distribution: p_i = |c_i| / kappa_single
        abs_coeffs = np.array([abs(ch.coefficient) for ch in self.channels])
        kappa_single = abs_coeffs.sum()
        probs = abs_coeffs / kappa_single

        samples: List[Tuple[Tuple[int, ...], float]] = []
        for _ in range(n_samples):
            indices = tuple(
                int(rng.choice(self.n_channels, p=probs)) for _ in range(self.n_cuts)
            )
            # Weight = kappa^K / n_samples * sign(product of coefficients)
            sign = 1.0
            for ch_idx in indices:
                sign *= np.sign(self.channels[ch_idx].coefficient)
            weight = (self.kappa / n_samples) * sign
            samples.append((indices, weight))

        return samples

    def get_channel(self, index: int) -> Channel:
        """Get a specific channel by index."""
        return self.channels[index]

    def _index_to_channels(self, flat_idx: int) -> Tuple[int, ...]:
        """Convert flat index to K-tuple of channel indices (base-8)."""
        indices: List[int] = []
        remaining = flat_idx
        for _ in range(self.n_cuts):
            indices.append(remaining % self.n_channels)
            remaining //= self.n_channels
        return tuple(indices)

    def channels_to_index(self, indices: Tuple[int, ...]) -> int:
        """Convert K-tuple of channel indices to flat index."""
        flat = 0
        for k, idx in enumerate(indices):
            flat += idx * (self.n_channels**k)
        return flat


# ---------------------------------------------------------------------------
# Gate-level QPD for particle-conserving (Givens rotation) gates
# ---------------------------------------------------------------------------

def _givens_pauli_coeffs(theta: float) -> List[Tuple[float, str, str]]:
    """Pauli decomposition of the particle-conserving Givens rotation U(theta).

    U(theta) acts on the {|01>,|10>} subspace as a rotation by theta,
    leaving |00> and |11> invariant.  Its Pauli expansion is:

        U = (1/2)(II + ZZ) + (sin theta / 2)(IZ - ZI)
            + (cos theta / 2)(XX + YY)

    Returns list of (coefficient, pauli_A, pauli_B) triples.
    """
    s, c = np.sin(theta), np.cos(theta)
    terms = [
        (0.5,       "I", "I"),
        (0.5,       "Z", "Z"),
        ( s / 2.0,  "I", "Z"),
        (-s / 2.0,  "Z", "I"),
        ( c / 2.0,  "X", "X"),
        ( c / 2.0,  "Y", "Y"),
    ]
    return [(coeff, a, b) for coeff, a, b in terms if abs(coeff) > 1e-15]


class GateQPDDecomposition:
    """QPD decomposition for cross-cluster Givens rotation gates.

    Decomposes the channel E(rho) = U rho U† into a sum of local
    channels using the Pauli expansion of U.  For U = sum_k c_k
    (sigma_k^A x sigma_k^B), the channel is:

        E(rho) = sum_{k,l} c_k c_l (sigma_k^A rho_A sigma_l^A)
                                  x (sigma_k^B rho_B sigma_l^B)

    Each (k,l) term is a local channel pair that can be implemented
    by applying sigma_l before and sigma_k after the gate position
    on each qubit independently.
    """

    def __init__(self, theta: float):
        self.theta = theta
        self.pauli_terms = _givens_pauli_coeffs(theta)
        # Build all (k,l) configurations
        self._configs: List[Tuple[Tuple[int, int], float, str, str, str, str]] = []
        for k, (ck, sk_A, sk_B) in enumerate(self.pauli_terms):
            for l, (cl, sl_A, sl_B) in enumerate(self.pauli_terms):
                coeff = ck * cl
                if abs(coeff) > 1e-15:
                    self._configs.append((
                        (k, l), coeff,
                        sk_A, sl_A,  # source: apply sl_A before, sk_A after
                        sk_B, sl_B,  # target: apply sl_B before, sk_B after
                    ))

    @property
    def n_terms(self) -> int:
        return len(self._configs)

    @property
    def kappa(self) -> float:
        return sum(abs(c) for _, c, *_ in self._configs)

    def enumerate_configurations(self) -> List[Tuple[Tuple[int, int], float]]:
        """Return list of ((k, l), coefficient) for all non-zero terms."""
        return [((k, l), coeff) for (k, l), coeff, *_ in self._configs]

    def get_local_ops(self, config_idx: int) -> Tuple[str, str, str, str]:
        """Return (src_before, src_after, tgt_before, tgt_after) Pauli labels."""
        _, _, sk_A, sl_A, sk_B, sl_B = self._configs[config_idx]
        return sl_A, sk_A, sl_B, sk_B


class PauliRotationQPDDecomposition:
    """Channel decomposition for U(theta)=exp(-i theta P_A⊗P_B).

    Uses the exact two-term Pauli expansion:
        U = c0 * I + c1 * (P_A⊗P_B)
    with c0 = cos(theta), c1 = -i sin(theta).

    The channel E(rho)=U rho U† is expanded as:
        E = sum_{k,l in {0,1}} c_k c_l* L_{k,l}
    where each L_{k,l} is implemented by applying P before/after the gate
    position on each side when l/k equals 1.

    Note:
        Off-diagonal terms carry imaginary coefficients. Downstream
        reconstruction therefore supports complex configuration weights.
    """

    def __init__(self, theta: float):
        self.theta = theta
        c0 = complex(np.cos(theta))
        c1 = complex(-1j * np.sin(theta))
        self._amplitudes = (c0, c1)
        self._configs: List[Tuple[Tuple[int, int], complex]] = []
        for k in (0, 1):
            for l in (0, 1):
                coeff = self._amplitudes[k] * np.conjugate(self._amplitudes[l])
                if abs(coeff) > 1e-15:
                    self._configs.append(((k, l), coeff))

    @property
    def n_terms(self) -> int:
        return len(self._configs)

    def enumerate_configurations(self) -> List[Tuple[Tuple[int, int], complex]]:
        """Return list of ((k, l), coefficient) for all non-zero channel terms."""
        return list(self._configs)

    def get_coefficient(self, config_idx: int) -> complex:
        """Return coefficient for configuration index."""
        return self._configs[config_idx][1]

    def get_local_application_flags(self, config_idx: int) -> Tuple[bool, bool]:
        """Return (apply_before, apply_after) for local operator P.

        For config (k,l), apply P before if l=1 and after if k=1.
        """
        (k, l), _coeff = self._configs[config_idx]
        return bool(l), bool(k)


# Pauli multiplication table: (phase, result)
# NOTE: Also defined in vqe_mcpdft/hamiltonian.py and vqe_mcpdft/rdm.py.
# Kept local to avoid cross-package imports.
_PAULI_MULT = {
    ("I", "I"): (1, "I"), ("I", "X"): (1, "X"), ("I", "Y"): (1, "Y"), ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"), ("X", "X"): (1, "I"), ("X", "Y"): (1j, "Z"), ("X", "Z"): (-1j, "Y"),
    ("Y", "I"): (1, "Y"), ("Y", "X"): (-1j, "Z"), ("Y", "Y"): (1, "I"), ("Y", "Z"): (1j, "X"),
    ("Z", "I"): (1, "Z"), ("Z", "X"): (1j, "Y"), ("Z", "Y"): (-1j, "X"), ("Z", "Z"): (1, "I"),
}


def heisenberg_transform_pauli(
    pauli_str: str,
    gate_qubits: Tuple[int, int],
    gate_theta: float,
) -> List[Tuple[float, str]]:
    """Transform a Pauli string through a Givens rotation in the Heisenberg picture.

    Computes U†(theta) P U(theta) where U acts on gate_qubits and P is
    the full Pauli string.  Returns a list of (coefficient, new_pauli_str)
    pairs whose weighted sum equals the transformed operator.

    Only the two gate qubits are affected; all other positions pass through.
    """
    q0, q1 = gate_qubits
    P_A, P_B = pauli_str[q0], pauli_str[q1]
    rest = list(pauli_str)

    U_terms = _givens_pauli_coeffs(gate_theta)

    # Compute U† (P_A ⊗ P_B) U = sum_{k,l} ck*cl * (sk_A P_A sl_A) ⊗ (sk_B P_B sl_B)
    result: dict = {}
    for ck, sk_A, sk_B in U_terms:
        for cl, sl_A, sl_B in U_terms:
            ph_A1, r_A1 = _PAULI_MULT[(sk_A, P_A)]
            ph_A2, r_A = _PAULI_MULT[(r_A1, sl_A)]
            ph_B1, r_B1 = _PAULI_MULT[(sk_B, P_B)]
            ph_B2, r_B = _PAULI_MULT[(r_B1, sl_B)]
            total = ck * cl * ph_A1 * ph_A2 * ph_B1 * ph_B2
            key = (r_A, r_B)
            result[key] = result.get(key, 0) + total

    out: List[Tuple[float, str]] = []
    for (r_A, r_B), coeff in result.items():
        if abs(coeff) < 1e-14:
            continue
        new_ps = list(rest)
        new_ps[q0] = r_A
        new_ps[q1] = r_B
        out.append((float(coeff.real), "".join(new_ps)))
    return out


def transform_hamiltonian_through_gates(
    hamiltonian: dict,
    cross_cluster_gates: List[Tuple[Tuple[int, int], float]],
) -> dict:
    """Transform a Hamiltonian through cross-cluster gates (Heisenberg picture).

    For each cross-cluster gate U_k(theta_k) acting on (q0, q1), every
    Pauli string P in the Hamiltonian is replaced by U_k† P U_k.  Gates
    are applied in reverse circuit order so that the final Hamiltonian
    corresponds to measuring the original Hamiltonian on the state
    *without* the cross-cluster gates.

    Args:
        hamiltonian: {pauli_label: coefficient} dict.
        cross_cluster_gates: List of ((q0, q1), theta) in circuit order.

    Returns:
        Transformed Hamiltonian dict.
    """
    h = dict(hamiltonian)
    # Apply gates in reverse order (Heisenberg picture)
    for (q0, q1), theta in reversed(cross_cluster_gates):
        new_h: dict = {}
        for ps, coeff in h.items():
            transformed = heisenberg_transform_pauli(ps, (q0, q1), theta)
            for t_coeff, t_ps in transformed:
                new_h[t_ps] = new_h.get(t_ps, 0.0) + coeff * t_coeff
        # Filter negligible terms
        h = {ps: c for ps, c in new_h.items() if abs(c) > 1e-14}
    return h
