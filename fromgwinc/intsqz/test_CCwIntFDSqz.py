"""
Examples: coupled cavity with internal frequency-dependent squeezing.

The model lives in :mod:`sflu.models.int_fd_sqz`; this file loads parameters,
runs it, and plots.
"""
import numpy as np
import gwinc
from wield.control.SFLU import SFLU, nx2tikz
from wield.utilities.mpl import mplfigB

from sflu.models import sflu_CCwIntFDSqz, CoupledCavityIntFC
from sflu.models.budget import accumulate, quantum_budget
from sflu.params import standardize_params

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def test_CCwIntFDSqz(fpath_join, tpath_join, plotTF, pprint):
    """Test coupled cavity with internal frequency-dependent squeezing.

    The internal filter cavity performs half the frequency-dependent
    rotation of the squeezing angle. Combined with appropriate homodyne
    angle (variational readout), this achieves broadband quantum noise
    reduction.
    """
    use_SS = True
    F_Hz = np.geomspace(10, 30e3, 1000)

    # load reference budgets
    budgetApl = gwinc.load_budget('Aplus', freq=F_Hz)

    # reference curves
    aplB = budgetApl.run()
    aplQB = aplB.Quantum
    aplClassicalPSD = np.sum([aplB[k].psd for k in aplB.keys() if k != 'Quantum'], axis=0)
    aplClassicalASD = np.sqrt(aplClassicalPSD)

    # load internal FD squeezing config
    ifo = gwinc.load_budget(fpath_join('AhatTestIntFC.yaml')).ifo
    ifo.Optics.INTSQ_loss = 1000e-6
    print(ifo.Optics)

    sfluB = sflu_CCwIntFDSqz()
    sfluB.sflu.reduce_auto()
    params = standardize_params(ifo)
    mats = accumulate(
        sfluB, plant=CoupledCavityIntFC, ifo=ifo, params=params, F_Hz=F_Hz,
        use_SS=use_SS, filter_cavity=None,
    )
    out = quantum_budget(sfluB, mats, ifo, params, F_Hz=F_Hz, strain=False)
    total, ASport, LB = out.total, out.ASport, out.LB

    L_arm = 4000

    # --- total noise comparison plot ---
    axB = mplfigB()
    axB.ax0.set_ylim(1e-25, 3e-23)
    axB.ax0.loglog(F_Hz, total**0.5 / L_arm, label='Internal Frequency Dependent Squeezing', lw=3, color='teal')
    #axB.ax0.loglog(F_Hz, total_noFC**0.5 / L_arm, label='IntSqz (no FC)', lw=2, ls='--', color='orange')
    axB.ax0.loglog(aplQB.freq, aplQB.asd, label='A+ quantum', lw=2, color='black')
    axB.ax0.loglog(aplB.freq, aplClassicalASD, label='A+ classical (total)', lw=2, ls=':', color='red')
    for k in ('CoatingBrownian', 'SuspensionThermal'):
        axB.ax0.loglog(aplB.freq, aplB[k].asd, label=f'A+ {k}', lw=1.5, ls='--', alpha=0.8)
    #axB.ax0.loglog(aplQB.freq, aplQB.asd/2, label='A+ quantum/2', lw=1, ls='--', color='black')
    axB.ax0.legend(loc='lower left', framealpha=1, fontsize=7)
    axB.ax0.set_xlabel('Frequency [Hz]')
    axB.ax0.set_ylabel('Strain ASD [1/$\\sqrt{\\mathrm{Hz}}$]')
    axB.save(tpath_join('intFDsqz_cmp'))

    # --- A+ only version of the comparison plot ---
    axA = mplfigB()
    axA.ax0.set_ylim(1e-25, 3e-23)
    axA.ax0.loglog(aplQB.freq, aplQB.asd, label='A+ quantum', lw=2, color='black')
    axA.ax0.legend(loc='lower left', framealpha=1, fontsize=7)
    axA.ax0.set_xlabel('Frequency [Hz]')
    axA.ax0.set_ylabel('Strain ASD [1/$\\sqrt{\\mathrm{Hz}}$]')
    axA.save(tpath_join('intFDsqz_cmp_Aplus_only'))

    # --- loss budget plot ---
    axL = mplfigB()
    axL.ax0.set_ylim(1e-25, 3e-23)
    axL.ax0.loglog(F_Hz, total**0.5 / L_arm, label='Total', lw=3, color='teal')
    axL.ax0.loglog(F_Hz, ASport**0.5 / L_arm, label='AS port', lw=2, color='gray')
    for lpN in LB:
        axL.ax0.loglog(F_Hz, LB[lpN]**0.5 / L_arm, label=lpN, lw=1.5, ls='--')
    axL.ax0.legend(loc='lower left', framealpha=1, fontsize=7)
    axL.ax0.set_xlabel('Frequency [Hz]')
    axL.ax0.set_ylabel('Strain ASD [1/$\\sqrt{\\mathrm{Hz}}$]')
    axL.save(tpath_join('intFDsqz_loss_budget'))

    pprint("IntFDSqz test completed")
    pprint(f"Loss port keys: {list(LB.keys())}")

    return



def test_build_CCwIntFDSqz(tpath_join):
    """Test that the SFLU graph builds and serializes correctly."""
    sfluB = sflu_CCwIntFDSqz()
    sflu = sfluB.sflu
    yamlstr = sflu.convert_self2yamlstr()
    with open(tpath_join('CoupledCavINTFDSQZ.yaml'), 'w') as F:
        F.write(yamlstr)
    sflu2 = SFLU.SFLU.convert_yamlstr2self(yamlstr)



def test_plot_graph_CCwIntFDSqz(tpath_join):
    """Plot the signal flow graph for the internal FD squeezing topology.

    Generates two graphs:
    - G1: full unreduced graph showing all nodes and edges
    - G2: reduced graph after SFLU auto-reduction
    """
    sfluB = sflu_CCwIntFDSqz()
    sflu = sfluB.sflu
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-8, rX=+8, Y=3, dY=-3)
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos_io(lX=-8, rX=+8, Y=3, dY=-3)
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname=tpath_join("intFDsqz_graph.pdf"),
        scale="10pt",
    )
