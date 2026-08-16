"""
Quantum noise budget: transfer functions in, PSD and loss breakdown out.

This step is model-independent, and there is now exactly one implementation of
it. Before Stage 3 of ``REFACTOR_PLAN.md`` the same arithmetic existed three
times: inline in ``test_CoupledCav``, again in ``intSqzQuantum`` (73 lines
identical to the first), and a third time inline in ``test_CCwIntFDSqz`` where
the result was only consumed by a commented-out plot line.

The pipeline is:

    injection loss -> [external filter cavity] -> plant -> readout loss
                                                              |
                                        LO projection, |.|^2, per-port sum
                                                              |
                                                  total PSD + loss budget

Models differ only in which plant they hand over, whether an external filter
cavity precedes it, and whether the result is reported as displacement or as
strain.
"""
import numpy as np
from gwinc import const
from gwinc.ifo.noises import dhdl
from wield.bunch import Bunch

from sflu_components.lib import MatsHelper, Vnorm_sq, Vnorm_sqA, adjoint


def accumulate(sfluB, plant, ifo, params, F_Hz, use_SS=True,
               filter_cavity=None):
    """Chain the optical path into a MatsHelper of transfer matrices.

    Parameters
    ----------
    sfluB : Bunch
        Topology bundle: ``sflu`` graph, ``loss_ports``, ``strain_exc``.
    plant : callable
        ``plant(sflu, F_Hz, ifo, params, use_SS) -> dict`` with a
        ``resultsAC`` entry. This is the model-specific piece.
    filter_cavity : callable or None
        ``filter_cavity(F_Hz, ifo, params, use_SS) -> dict`` for an external
        squeezing filter cavity ahead of the interferometer. ``None`` skips it,
        which is what the internal-filter-cavity model wants: its filter cavity
        is inside the plant, not ahead of it.
    """
    mlib = params.mlib
    mats = MatsHelper()
    mats.H['AS'] = mlib.Id

    # injection loss
    L_inj_t = (1 - params.Loss.injection)**0.5
    mats.update_scalar(L_inj_t)
    mats.T['Loss_injection'] = mlib.Id * params.Loss.injection**0.5

    # FIXME: add support for multiple filter cavities
    if filter_cavity is not None and 'Squeezer' in ifo:
        results_FC = filter_cavity(F_Hz, ifo, params, use_SS=use_SS)['resultsAC']
        mats.update_matrix(results_FC["FC1.bk.i.exc"])
        mats.T.update({k: v for k, v in results_FC.items()
                       if k != "FC1.bk.i.exc"})

    results_IFO = plant(
        sflu=sfluB.sflu, F_Hz=F_Hz, ifo=ifo, params=params, use_SS=use_SS,
    )['resultsAC']
    mats.update_matrix(results_IFO["SEM.bk.i.exc"])
    mats.T.update({k: v for k, v in results_IFO.items()
                   if k != "SEM.bk.i.exc"})

    # FIXME add output filter cavities

    # readout loss
    L_read_t = (1 - params.Loss.readout)**0.5
    mats.update_scalar(L_read_t)
    mats.T['Loss_readout'] = mlib.diag(params.Loss.readout**0.5)
    return mats


def quantum_budget(sfluB, mats, ifo, params, F_Hz, strain=False,
                   alias_ASport=False):
    """Project onto the LO and accumulate the per-port noise budget.

    Parameters
    ----------
    strain : bool
        Convert displacement PSD to strain PSD via ``dhdl``. The
        internal-squeezing entry point reports strain; the plotting examples
        divide by the arm length themselves and want displacement.
    alias_ASport : bool
        Reproduce a long-standing aliasing bug in ``intSqzQuantum``, where
        ``total = ASport`` followed by an in-place ``total += lossB`` silently
        overwrote the reported AS-port term with the running total. Preserved
        so this refactor changes no number; see the NOTE below.

    Returns
    -------
    Bunch with ``total``, ``ASport``, ``LB``, ``d_sense``, ``PSDdisplacement``.
    """
    mlib = params.mlib
    k_ = 2 * np.pi / ifo.Laser.Wavelength

    sqzV = 10**(-params.SQZ_DB / 10.)
    asqzV = 10**(params.ANTISQZ_DB / 10.)
    SQZ_angle_rad = params.alpha
    HD_angle_rad = params.LO_angle

    # local oscillator, as a row vector to project the output onto
    if params.follow_fringe:
        # NOTE: mats.L is only populated by MatsHelper.update_LO(), which
        # nothing calls, so this branch raises. Kept as-is: making it work is
        # a modelling decision, not a refactor.
        LOa = adjoint(mats.L['AS'] @ mlib.LO(HD_angle_rad))
        LOa = LOa / (Vnorm_sq(LOa)**0.5).reshape(-1, 1, 1)
    else:
        LOa = adjoint(mlib.LO(HD_angle_rad))

    # scalar s in (E22): strain drive to readout
    d_sense = np.sum([cc * mats.T[exc]
                      for exc, cc in sfluB.strain_exc.items()], axis=0)
    d_sense = np.squeeze(LOa @ d_sense)

    omega = k_ * const.c
    Qnoise = const.hbar * omega / 2
    # \hbar\omega/2 * G * L_A^2 in (E23)
    PSDdisplacement = Qnoise / abs(d_sense)**2

    # every non-signal input port contributes vacuum noise at the readout
    ASbudget = {}
    for key in mats.T:
        if key in sfluB.strain_exc:
            continue
        ASbudget[key] = Vnorm_sqA(LOa @ mats.T[key])

    LOdotAS = LOa @ mats.H['AS']
    ASquantumAll = Vnorm_sqA(
        LOdotAS @ mlib.Mrotation(SQZ_angle_rad) @ mlib.SQZ(sqzV, asqzV)
    )

    scale = 1.0
    if strain:
        dhdl_sqr, _sinc_sqr = dhdl(F_Hz, ifo.Infrastructure.Length)
        scale = dhdl_sqr

    ASport = ASquantumAll * PSDdisplacement * scale
    total = ASport.copy()

    LB = {}
    if alias_ASport:
        # NOTE: bug-compatibility. The original bound LB['ASport'] to the same
        # array as `total`, so the in-place accumulation below overwrote it and
        # callers received total under the name ASport. Verified against the
        # captured baseline: LB['ASport'] is bit-identical to total.
        LB['ASport'] = total

    for port_name, port_list in sfluB.loss_ports.items():
        lossB = PSDdisplacement * _sum_losses(ASbudget, port_list) * scale
        LB[port_name] = lossB
        total += lossB

    return Bunch(
        total=total,
        ASport=ASport,
        LB=LB,
        d_sense=d_sense,
        PSDdisplacement=PSDdisplacement,
        ASbudget=ASbudget,
    )


def _sum_losses(noises, loss_ports):
    return np.sum([noises[exc_pt] for exc_pt in loss_ports], axis=0)
