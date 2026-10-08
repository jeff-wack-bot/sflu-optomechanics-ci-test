"""
Homodyne readout and quantum noise, from the transfer matrices ``solve_ac``
returns.

Conventions, stated once because every paper picks its own:

**Quadratures.** The field vector is ``[amplitude, phase]`` per mode, i.e.
the two-photon quadratures ``(a1, a2)`` of Caves & Schumaker and KLMTV, with
``a1`` along the carrier. Angles here follow KLMTV: the homodyne quadrature
at angle ``zeta`` is ``b1 cos(zeta) + b2 sin(zeta)``, so ``zeta = pi/2`` reads
the phase quadrature (the conventional GW readout), and a squeeze angle
``lam`` squeezes ``a1 cos(lam) + a2 sin(lam)``, so ``lam = pi/2`` is phase
squeezing. (``MatrixLib.LO(phi)`` uses ``phi = pi/2 - zeta``; this module
does the conversion so that model code never has to.)

**Normalisation.** Carrier fields are in sqrt(W), with power ``|E|^2``. With
that choice a power fluctuation is ``2 sqrt(P) da1``, which is what the
radiation-pressure edges assume, and the **single-sided** vacuum spectral
density of each quadrature is ``hbar omega0 / 2``. Every PSD below is
single-sided, as in KLMTV Eq. 22.

**States.** An input is described by its quadrature covariance in units of
vacuum: the identity for vacuum, ``squeezed(...)`` for squeezed vacuum.
Covariances and readout quadratures may carry a leading frequency axis, so a
frequency-dependent squeeze angle or homodyne angle (an ideal filter cavity)
is just an array.
"""
import numpy as np
import scipy.constants as scc

from .lib import adjoint


def vacuum_psd(lambda_m=1064e-9):
    """Single-sided vacuum PSD of one quadrature, ``hbar omega0 / 2`` [W/Hz]."""
    return scc.hbar * 2 * np.pi * scc.c / lambda_m / 2


def quadrature(mlib, zeta):
    """Readout row vector for homodyne angle ``zeta`` [rad], fundamental mode.

    ``zeta`` may be an ``(N,)`` array; the result is then ``(N, 1, dim)``.
    """
    zeta = np.asarray(zeta, dtype=float)
    row = np.zeros(zeta.shape + (1, mlib.dim))
    row[..., 0, 0] = np.cos(zeta)
    row[..., 0, 1] = np.sin(zeta)
    return row


def squeezed(mlib, dB, angle, antisqueeze_dB=None):
    """Covariance of squeezed vacuum, in units of vacuum.

    Parameters
    ----------
    dB : float
        Squeezing, positive: 10 dB means the squeezed quadrature's variance is
        0.1 of vacuum (KLMTV's ``e^{-2R} = 0.1``).
    angle : float or (N,) array
        The squeezed quadrature, ``a1 cos(angle) + a2 sin(angle)`` [rad].
    antisqueeze_dB : float, optional
        Defaults to ``dB`` (a pure state).

    Higher-order modes, if ``mlib`` has any, are left in vacuum.
    """
    if antisqueeze_dB is None:
        antisqueeze_dB = dB
    angle = np.asarray(angle, dtype=float)
    u = np.stack([np.cos(angle), np.sin(angle)], axis=-1)[..., :, None]
    w = np.stack([-np.sin(angle), np.cos(angle)], axis=-1)[..., :, None]
    V2 = (10**(-dB / 10) * u @ np.swapaxes(u, -1, -2)
          + 10**(antisqueeze_dB / 10) * w @ np.swapaxes(w, -1, -2))
    V = np.broadcast_to(mlib.Id, angle.shape + (mlib.dim, mlib.dim)).copy()
    V[..., :2, :2] = V2
    return V


def noise_budget(T, row, states=None, lambda_m=1064e-9):
    """Readout noise PSD contributed by each optical input.

    Parameters
    ----------
    T : dict
        ``{input: (N, dim, dim) transfer matrix to the readout}``, as returned
        by ``sflu.solve.solve_ac``. Non-optical inputs (shape ``(N, dim, 1)``)
        are skipped: they carry signal or classical drives, not vacuum.
    row : (1, dim) or (N, 1, dim) array
        Readout quadrature, from ``quadrature()``.
    states : dict, optional
        ``{input: covariance}`` for inputs not in vacuum; everything else is
        vacuum.

    Returns
    -------
    dict of input -> ``(N,)`` single-sided PSD [W/Hz] at the readout.
    """
    states = states or {}
    q = vacuum_psd(lambda_m)
    out = {}
    for name, M in T.items():
        M = np.asarray(M)
        if M.shape[-1] != M.shape[-2]:
            continue
        RM = row @ M
        V = states.get(name)
        if V is None:
            P = RM @ adjoint(RM)
        else:
            P = RM @ V @ adjoint(RM)
        out[name] = q * P[..., 0, 0].real
    return out


def response(T, row, signal):
    """Readout response to the ``signal`` input, e.g. ``"ETM.pos.exc"``.

    Returns the ``(N,)`` complex transfer function, readout per unit signal.
    """
    return np.squeeze(row @ T[signal], axis=(-2, -1))


def referred_psd(T, row, signal, states=None, lambda_m=1064e-9):
    """Quantum noise referred to the ``signal`` input.

    For ``signal="ETM.pos.exc"`` this is the displacement-noise PSD
    [m^2/Hz]. Returns ``(total, budget)``, the second being the per-input
    breakdown, likewise referred to the signal.
    """
    budget = noise_budget(T, row, states, lambda_m)
    G2 = np.abs(response(T, row, signal))**2
    budget = {k: v / G2 for k, v in budget.items()}
    return np.sum(list(budget.values()), axis=0), budget


__all__ = [
    "noise_budget",
    "quadrature",
    "referred_psd",
    "response",
    "squeezed",
    "vacuum_psd",
]
