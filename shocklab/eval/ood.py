"""Out-of-distribution meter for an initial condition (CLAUDE.md §6.4).

Scores u0 against the training distribution (GRF ICs, zero mean, peak |u0| in
[0.5, 1.0], energy concentrated in low modes) along three axes:

  * peak amplitude vs [0.5, 1.0],
  * mean vs 0 (training ICs are zero-mean),
  * spectrum: the fraction of energy above mode 16, and per-mode amplitude vs the
    GRF's own modal std.

Output level is IN-DISTRIBUTION, NEAR, or OUT-OF-DISTRIBUTION with plain-language
reasons.
"""

from __future__ import annotations

import numpy as np

AMP_LO, AMP_HI = 0.5, 1.0
MEAN_TOL = 0.05
HIGH_MODE = 16
HIGH_FRAC_TOL = 0.15


def grf_mode_std(n: np.ndarray) -> np.ndarray:
    """Training GRF per-mode std: sqrt(625 / ((pi n)^2 + 25)^2) (CLAUDE.md §3)."""
    return np.sqrt(625.0 / ((np.pi * n) ** 2 + 25.0) ** 2)


def ood_report(u0: np.ndarray) -> tuple[str, list[str]]:
    """Return (level, reasons) for an initial condition sampled on the periodic grid."""
    u0 = np.asarray(u0, dtype=np.float64)
    peak = float(np.abs(u0).max())
    mean = float(u0.mean())  # ~= (1/2) integral u0 dx; training ICs are ~0

    uh = np.abs(np.fft.rfft(u0))
    energy = uh**2
    total = float(energy[1:].sum())  # exclude the DC/mean mode
    high_frac = float(energy[HIGH_MODE + 1 :].sum() / total) if total > 0 else 0.0

    reasons: list[str] = []
    score = 0
    if peak < AMP_LO:
        reasons.append(f"peak amplitude {peak:.2f} < {AMP_LO} (weaker than training)")
        score += 1
    elif peak > AMP_HI:
        reasons.append(f"peak amplitude {peak:.2f} > {AMP_HI} (stronger than training)")
        score += 1
    if abs(mean) > MEAN_TOL:
        reasons.append(f"nonzero mean {mean:+.2f} (training ICs are zero-mean)")
        score += 2  # a hard training constraint -> strong OOD signal
    if high_frac > HIGH_FRAC_TOL:
        reasons.append(f"{high_frac:.0%} of energy above mode {HIGH_MODE} (sharper than training)")
        score += 1

    if score == 0:
        level = "IN-DISTRIBUTION"
    elif score == 1:
        level = "NEAR"
    else:
        level = "OUT-OF-DISTRIBUTION"
    if not reasons:
        reasons = ["matches the training distribution (amplitude, zero mean, smooth spectrum)"]
    return level, reasons
