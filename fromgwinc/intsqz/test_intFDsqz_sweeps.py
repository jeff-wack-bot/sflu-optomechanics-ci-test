"""
Parameter sweeps and signal response comparisons for internal FD squeezing.

Contains tests that sweep filter cavity parameters (pole, detuning),
homodyne readout angle, and compare signal response across topologies.
"""

import copy
import numpy as np
from os import path

from wield.bunch import Bunch
from wield.utilities.mpl import mplfigB
import gwinc

from sflu_components.lib import (
    adjoint,
)
from .lib import MatsHelper, Vnorm_sq, Vnorm_sqA
from . import optics

from gwinc.struct import Struct
from gwinc import const

from .common import standardize_params, arm_gouyRT

from .test_CCwIntFDSqz import (
    sflu_CCwIntFDSqz,
    _compute_intFDsqz_budget,
    intFDsqzQuantum,
)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def test_intFDsqz_param_sweep(fpath_join, tpath_join, plotTF, pprint):
    """Sweep internal FC pole and detuning for frequency-dependent squeezing.

    Sweeps the internal filter cavity pole (half-linewidth) from 10 to 20 Hz
    at two fixed detuning values (10 Hz and 20 Hz).  The homodyne angle is
    set to -20 degrees offset from pi/2 (near maximum sensitivity).

    For the travelling-wave cavity the coupling mirror transmission Ti
    is computed from::

        pole_Hz = c * (Ti + Lrt) / (8 * pi * L)

    where L is the one-way cavity length (round-trip = 2L) and the BS
    loss per round trip is Ti + Lrt (no separate end mirror).
    """
    use_SS = True
    F_Hz = np.geomspace(10, 30e3, 1000)
    L_arm = 4000

    # reference budgets
    budgetApl = gwinc.load_budget('Aplus', freq=F_Hz)
    aplB = budgetApl.run()
    aplQB = aplB.Quantum

    # base configuration
    budget = gwinc.load_budget(fpath_join('AhatTestIntFC.yaml'))
    ifo_base = budget.ifo
    ifo_base.Optics.INTSQ_loss = 1000e-6

    # homodyne angle: pi/2 - 20 deg (near maximum sensitivity)
    hd_opt_offset_deg = 0
    hd_opt_rad = np.pi / 2 + hd_opt_offset_deg * np.pi / 180
    pprint(f"Homodyne angle: pi/2 + {hd_opt_offset_deg:.1f} deg "
           f"= {hd_opt_rad * 180 / np.pi:.1f} deg")

    # fixed cavity parameters
    IFC_L = ifo_base.intSqueezer.FilterCavity.L
    IFC_Lrt = ifo_base.intSqueezer.FilterCavity.Lrt

    def Ti_from_pole(pole_Hz):
        """Compute BS coupling transmission for a desired pole frequency."""
        return 8 * np.pi * IFC_L * pole_Hz / const.c - IFC_Lrt

    # ---- sweep grid ----
    pole_Hz_values = np.linspace(20, 60, 4)
    fdetune_Hz_values = np.linspace(0, -60, 4)

    # ======================================================================
    # Sweep: FC pole at each fixed detuning (at optimal homodyne angle)
    # ======================================================================
    all_results = {'freq': F_Hz}

    for fdet in fdetune_Hz_values:
        pprint(f"=== Sweep: FC pole 10--20 Hz, detuning = {fdet} Hz ===")

        results = {}
        for pole_Hz in pole_Hz_values:
            ifo = copy.deepcopy(ifo_base)
            ifo.intSqueezer.FilterCavity.Ti = Ti_from_pole(pole_Hz)
            ifo.intSqueezer.FilterCavity.fdetune = fdet
            ifo.Optics.Quadrature.dc = hd_opt_rad
            total, LB, _ = intFDsqzQuantum(ifo, F_Hz, use_SS=use_SS)
            key = f'pole{pole_Hz:.0f}_fd{abs(fdet)}'
            results[key] = total
            all_results[key] = total
            pprint(f"  pole={pole_Hz:.0f} Hz, Ti={ifo.intSqueezer.FilterCavity.Ti:.3e}: "
                   f"min ASD = {np.min(total**0.5 / L_arm):.2e}")

        # plot for this detuning value
        axB = mplfigB(size_in=[6.5, 4])
        cmap_pole = plt.cm.viridis
        norm_pole = mcolors.Normalize(
            vmin=pole_Hz_values[0], vmax=pole_Hz_values[-1],
        )
        sm_pole = mcm.ScalarMappable(cmap=cmap_pole, norm=norm_pole)
        sm_pole.set_array([])

        for idx, pole_Hz in enumerate(pole_Hz_values):
            key = f'pole{pole_Hz:.0f}_fd{abs(fdet)}'
            axB.ax0.loglog(
                F_Hz, results[key]**0.5 / L_arm,
                color=cmap_pole(norm_pole(pole_Hz)), lw=1.5,
            )
        axB.ax0.loglog(aplQB.freq, aplQB.asd, color='black', lw=2, ls='--', label='A+ quantum')
        axB.ax0.loglog(aplQB.freq, aplQB.asd / 2, color='black', lw=1, ls=':', label='A+ quantum/2')
        axB.ax0.set_ylim(1e-25, 3e-23)
        axB.ax0.set_xlabel('Frequency [Hz]')
        axB.ax0.set_ylabel('Strain ASD [1/$\\sqrt{\\mathrm{Hz}}$]')
        axB.ax0.set_title(f'IntFDSqz: FC pole sweep, $f_{{\\mathrm{{det}}}}$ = {fdet} Hz, '
                          f'$\\Delta\\zeta$ = {hd_opt_offset_deg:.1f}$^\\circ$')
        axB.ax0.legend(loc='lower left', framealpha=1, fontsize=7)
        cbar = axB.fig.colorbar(sm_pole, ax=axB.ax0)
        cbar.set_label('FC pole [Hz]')
        axB.save(tpath_join(f'sweep_pole_fdet{abs(fdet):.0f}'))

    # ======================================================================
    # Save all results to CSV
    # ======================================================================
    arr = np.array(list(all_results.values())).T
    np.savetxt(
        tpath_join('param_sweep.csv'),
        arr,
        delimiter=",",
        header=', '.join(all_results.keys()),
    )
    pprint(f"Saved {len(all_results)} columns to param_sweep.csv")


def test_intFDsqz_homodyne_sweep(fpath_join, tpath_join, plotTF, pprint):
    """Sweep homodyne readout angle for internal FD squeezing.

    Sweeps the homodyne angle (Optics.Quadrature.dc) around the nominal
    value of pi/2 (phase quadrature).  The offset from pi/2 sets the
    variational readout angle that provides the second half of the
    frequency-dependent rotation.
    """
    use_SS = True
    F_Hz = np.geomspace(10, 30e3, 1000)
    L_arm = 4000

    # reference budgets
    budgetApl = gwinc.load_budget('Aplus', freq=F_Hz)
    aplB = budgetApl.run()
    aplQB = aplB.Quantum

    # base configuration
    budget = gwinc.load_budget(fpath_join('AhatTestIntFC.yaml'))
    ifo_base = budget.ifo
    ifo_base.Optics.INTSQ_loss = 1000e-6

    # ---- sweep grid ----
    # homodyne angle offsets from pi/2 in degrees
    hd_offset_deg_values = np.linspace(-90, 90, 37)

    pprint("=== Sweep: homodyne angle offset from pi/2 ===")

    results = {}
    all_results = {'freq': F_Hz}

    for hd_off_deg in hd_offset_deg_values:
        ifo = copy.deepcopy(ifo_base)
        hd_rad = np.pi / 2 + hd_off_deg * np.pi / 180
        ifo.Optics.Quadrature.dc = hd_rad
        total, LB, _ = intFDsqzQuantum(ifo, F_Hz, use_SS=use_SS)
        key = f'hd{hd_off_deg:+.1f}deg'
        results[key] = total
        all_results[key] = total
        pprint(f"  HD offset = {hd_off_deg:+.1f} deg: "
               f"min ASD = {np.min(total**0.5 / L_arm):.2e}")

    # ---- plot with colorbar ----
    axB = mplfigB(size_in=[6.5, 4])
    cmap = plt.cm.coolwarm
    norm = mcolors.Normalize(
        vmin=hd_offset_deg_values[0], vmax=hd_offset_deg_values[-1],
    )
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for idx, hd_off_deg in enumerate(hd_offset_deg_values):
        key = f'hd{hd_off_deg:+.1f}deg'
        axB.ax0.loglog(
            F_Hz, results[key]**0.5 / L_arm,
            color=cmap(norm(hd_off_deg)), lw=1.5,
        )
    axB.ax0.loglog(aplQB.freq, aplQB.asd, color='black', lw=2, ls='--', label='A+ quantum')
    axB.ax0.loglog(aplQB.freq, aplQB.asd / 2, color='black', lw=1, ls=':', label='A+ quantum/2')
    axB.ax0.set_ylim(1e-25, 3e-23)
    axB.ax0.set_xlabel('Frequency [Hz]')
    axB.ax0.set_ylabel('Strain ASD [1/$\\sqrt{\\mathrm{Hz}}$]')
    axB.ax0.set_title('IntFDSqz: homodyne angle sweep ($\\zeta = \\pi/2 + \\Delta\\zeta$)')
    axB.ax0.legend(loc='lower left', framealpha=1, fontsize=7)
    cbar = axB.fig.colorbar(sm, ax=axB.ax0)
    cbar.set_label('$\\Delta\\zeta$ [deg]')
    axB.save(tpath_join('sweep_homodyne_angle'))

    # ---- save CSV ----
    arr = np.array(list(all_results.values())).T
    np.savetxt(
        tpath_join('homodyne_sweep.csv'),
        arr,
        delimiter=",",
        header=', '.join(all_results.keys()),
    )
    pprint(f"Saved {len(all_results)} columns to homodyne_sweep.csv")


def _compute_d_sense_CC(F_Hz, ifo, use_SS=True):
    """Compute signal response d_sense for coupled cavity (no internal FC).

    Uses the same topology as test_CCwIntSqz (CoupledCavity).
    """
    from . import test_CCwIntSqz
    from . import FilterCavity

    sfluB = test_CCwIntSqz.sflu_CoupledCav()
    sflu = sfluB.sflu
    sflu.reduce_auto()
    params = standardize_params(ifo)
    mlib = params.mlib
    mats = MatsHelper()
    mats.H['AS'] = mlib.Id

    L_inj_t = (1 - params.Loss.injection)**0.5
    mats.update_scalar(L_inj_t)
    mats.T['Loss_injection'] = mlib.Id * params.Loss.injection**0.5

    # external filter cavity (if present in the config)
    if 'Squeezer' in ifo:
        ret_FC = FilterCavity.FilterCavity(F_Hz, ifo, params, use_SS=use_SS)
        results_FC = ret_FC['resultsAC']
        mats.update_matrix(results_FC["FC1.bk.i.exc"])
        mats.T.update({k: v for k, v in results_FC.items() if k != "FC1.bk.i.exc"})

    ret_IFO = test_CCwIntSqz.CoupledCavity(
        sflu=sflu, F_Hz=F_Hz, ifo=ifo, params=params, use_SS=use_SS,
    )
    results_IFO = ret_IFO['resultsAC']
    mats.update_matrix(results_IFO["SEM.bk.i.exc"])
    mats.T.update({k: v for k, v in results_IFO.items() if k != "SEM.bk.i.exc"})

    L_read_t = (1 - params.Loss.readout)**0.5
    mats.update_scalar(L_read_t)
    mats.T['Loss_readout'] = mlib.diag(params.Loss.readout**0.5)

    params = standardize_params(ifo)
    mlib = params.mlib
    HD_angle_rad = params.LO_angle
    LOa = adjoint(mlib.LO(HD_angle_rad))

    d_sense = np.sum([cc * mats.T[exc] for exc, cc in sfluB.strain_exc.items()], axis=0)
    d_sense = np.squeeze(LOa @ d_sense)
    return d_sense


def test_signal_response_comparison(fpath_join, tpath_join, plotTF, pprint):
    """Compare signal response |d_sense| for three configurations.

    1. Coupled cavity with external FD squeezing (standard A+-like)
    2. Coupled cavity with internal squeezing (no internal FC)
    3. Coupled cavity with internal FD squeezing (internal FC)

    The signal response is the transfer function from differential arm
    strain to the homodyne readout output.  It encodes how the cavity
    structure modifies the signal.
    """
    use_SS = True
    F_Hz = np.geomspace(10, 30e3, 1000)

    # load configs
    budget_intFC = gwinc.load_budget(fpath_join('AhatTestIntFC.yaml'))
    ifo_intFC = budget_intFC.ifo
    ifo_intFC.Optics.INTSQ_loss = 1000e-6

    budget_intSqz = gwinc.load_budget(fpath_join('AhatTest.yaml'))
    ifo_intSqz = budget_intSqz.ifo
    ifo_intSqz.Optics.INTSQ_loss = 1000e-6

    # 1. Coupled cavity with external FD squeezing (no internal squeezer)
    pprint("Computing signal response: coupled cavity (ext FD sqz)...")
    d_sense_CC = _compute_d_sense_CC(F_Hz, ifo_intSqz, use_SS=use_SS)

    # 2. Internal squeezing (no internal FC) — same topology as (1)
    #    but the SQZ edges modify the signal path
    #    Actually uses the same CoupledCavity topology, signal response
    #    is identical to case 1 since SQZ edges don't change d_sense.
    #    The difference is only in the noise.
    #    So we skip this and note it in the plot.

    # 3. Internal FD squeezing (with internal TWC FC)
    pprint("Computing signal response: internal FD sqz (TWC)...")
    sfluB_intFC = sflu_CCwIntFDSqz()
    _, _, _, d_sense_intFC = _compute_intFDsqz_budget(sfluB_intFC, F_Hz, ifo_intFC, use_SS)

    # plot magnitude of signal response
    axB = mplfigB(size_in=[6.5, 4])
    axB.ax0.loglog(F_Hz, abs(d_sense_CC), label='Coupled cavity (ext FD sqz)', lw=2, color='black')
    axB.ax0.loglog(F_Hz, abs(d_sense_intFC), label='Coupled cavity + internal TWC FC', lw=2, color='teal')

    axB.ax0.set_xlabel('Frequency [Hz]')
    axB.ax0.set_ylabel('$|d_{\\mathrm{sense}}|$ [1/m]')
    axB.ax0.set_title('Signal response: strain to homodyne readout')
    axB.ax0.legend(loc='best', framealpha=1, fontsize=8)
    axB.ax0.grid(True, alpha=0.3)
    axB.save(tpath_join('signal_response_comparison'))

    # also plot the phase
    axP = mplfigB(size_in=[6.5, 4])
    axP.ax0.semilogx(F_Hz, np.angle(d_sense_CC) * 180 / np.pi,
                      label='Coupled cavity (ext FD sqz)', lw=2, color='black')
    axP.ax0.semilogx(F_Hz, np.angle(d_sense_intFC) * 180 / np.pi,
                      label='Coupled cavity + internal TWC FC', lw=2, color='teal')

    axP.ax0.set_xlabel('Frequency [Hz]')
    axP.ax0.set_ylabel('Phase of $d_{\\mathrm{sense}}$ [deg]')
    axP.ax0.set_title('Signal response phase')
    axP.ax0.legend(loc='best', framealpha=1, fontsize=8)
    axP.ax0.grid(True, alpha=0.3)
    axP.save(tpath_join('signal_response_phase'))

    pprint("Signal response comparison complete")
