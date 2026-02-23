"""Error mitigation protocols for near-term quantum hardware.

Implements FEM-inspired readout error mitigation, Zero-Noise Extrapolation
(ZNE), and Clifford Fitting (CF) for improving VQE accuracy on NISQ devices.
"""

from error_mitigation.fem_readout import FEMReadoutMitigator
from error_mitigation.zne import ZeroNoiseExtrapolator
from error_mitigation.clifford_fitting import CliffordFitter

__all__ = [
    "FEMReadoutMitigator",
    "ZeroNoiseExtrapolator",
    "CliffordFitter",
]
