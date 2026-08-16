"""
Examples: coupled cavity with internal squeezing.

The model itself lives in :mod:`sflu.models.coupled_cavity`. This file only
loads parameters, runs it, and plots -- which is all a ``test_*.py`` should do.
"""
import numpy as np
import gwinc
from wield.control.SFLU import SFLU, nx2tikz
from wield.utilities.mpl import mplfigB

from sflu.models import sflu_CoupledCav, CoupledCavity, intSqzQuantum
from sflu.models import filter_cavity
from sflu.models.budget import accumulate, quantum_budget
from sflu.params import standardize_params

import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def test_CoupledCav(fpath_join, tpath_join, plotTF, pprint):
    """Compare the internal-squeezing budget against A+ and CE reference curves."""
    F_Hz = np.geomspace(10, 30e3, 1000)

    budgetApl = gwinc.load_budget('Aplus', freq=F_Hz)
    budgetAplWide = gwinc.load_budget(fpath_join('AplWide' + '.yaml'), freq=F_Hz)
    budgetCE2 = gwinc.load_budget('CE2silica', freq=np.geomspace(30, 10e3, 1000))

    ifo = gwinc.load_budget(fpath_join('AhatTest' + '.yaml')).ifo
    ifo.Optics.INTSQ_loss = 1000e-6
    print(ifo.Optics)

    # topology -> plant -> budget
    sfluB = sflu_CoupledCav()
    sfluB.sflu.reduce_auto()
    params = standardize_params(ifo)
    mats = accumulate(
        sfluB, plant=CoupledCavity, ifo=ifo, params=params, F_Hz=F_Hz,
        use_SS=True, filter_cavity=filter_cavity.FilterCavity,
    )
    out = quantum_budget(sfluB, mats, ifo, params, F_Hz=F_Hz, strain=False)
    total, ASport, LB = out.total, out.ASport, out.LB

    select = (F_Hz > 1e3) & (F_Hz < 1e4)
    print("MINRATIO", np.min(ASport[select] / LB['INTSQZ'][select]))

    axB = mplfigB()
    axB.ax0.set_ylim(1e-25, 3e-23)
    axB.ax0.loglog(F_Hz, (total)**0.5 / 4000, label='LIGOintSQZ', lw=3, color='orange')
    aplB = budgetApl.run()
    aplWB = budgetAplWide.run()
    ce2B = budgetCE2.run()
    aplQB = aplB.Quantum
    axB.ax0.loglog(aplQB.freq, aplQB.asd, label = 'ALIGOquantum', lw=2, color='black')
    axB.ax0.loglog(aplQB.freq, aplWB.Quantum.asd, label = 'ALIGOquantumWide', lw=2, color='black')
    axB.ax0.loglog(aplQB.freq, aplQB.asd/2, label = 'ALIGOquantum/2', lw=1, ls='--', color='black')
    axB.ax0.loglog(ce2B.freq, ce2B.asd, label = 'CE2quantum', lw=2, color='blue')
    axB.ax0.loglog(aplB.freq, aplB.CoatingBrownian.asd, label='A+CTN', color='red')
    axB.ax0.legend(loc='lower left', framealpha=1)
    axB.save(tpath_join('cmp'))

    fig = aplB.Quantum.plot()
    fig.savefig(tpath_join('budget.pdf'))
    return



def test_CoupledCav_variants(fpath_join, tpath_join, plotTF, pprint):
    use_SS = True
    F_Hz = np.geomspace(10, 15e3, 1000)

    #tpath = path.split(__file__)[0]
    #ifo = Struct.from_file(path.join(tpath, '../Aplus/ifo.yaml'))
    budgetApl = gwinc.load_budget('Aplus', freq=F_Hz)
    aplB = budgetApl.run()
    CLPSD = aplB.psd - aplB.Quantum.psd


    #budget = gwinc.load_budget(fpath_join('Ahat25' + '.yaml'))
    #budget = gwinc.load_budget(fpath_join('AhatPL25' + '.yaml'))

    #ifo.Optics.INTSQ_loss = 2000e-6
    # print(ifo.Optics)

    budgetD = {
        'freq': F_Hz,
        'AplCl': CLPSD,
    }

    def plot_loss_series(axB, ifo, *, name, **kw):
        ifo.Optics.INTSQ_loss = 500e-6
        totalL, LB = intSqzQuantum(ifo, freq=F_Hz)
        ifo.Optics.INTSQ_loss = 1000e-6
        totalM, LB = intSqzQuantum(ifo, freq=F_Hz)
        ifo.Optics.INTSQ_loss = 2000e-6
        totalH, LB = intSqzQuantum(ifo, freq=F_Hz)
        kw1 = dict(kw)
        kw1.pop('alpha', None)
        axB.ax0.loglog(F_Hz, (totalM + CLPSD)**0.5, **kw1)
        kw1['ls'] = '--'
        kw1.pop('label', None)
        kw1['lw'] = 2
        kw1['zorder'] = 5
        axB.ax0.loglog(F_Hz, (totalM)**0.5, **kw1)
        kwf = dict(kw)
        kwf.pop('ls', None)
        kwf.pop('lw', None)
        kwf.pop('label', None)
        kwf['lw'] = 0.5
        axB.ax0.fill_between(F_Hz, (totalL + CLPSD)**0.5, (totalH + CLPSD)**0.5, **kwf)
        budgetD[name + '_500ppm'] = totalL
        budgetD[name + '_1000ppm'] = totalM
        budgetD[name + '_2000ppm'] = totalH
        #axB.ax0.loglog(F_Hz, (totalL)**0.5 / 4000, **kw)
        #axB.ax0.loglog(F_Hz, (totalH)**0.5 / 4000, **kw)

    def plot_other_budget(axB, bname, *, name, dashed = False, **kw):
        budgetApl = gwinc.load_budget(bname, freq=F_Hz)
        aplB = budgetApl.run()
        axB.ax0.loglog(aplB.freq, aplB.asd, **kw)
        kw['ls'] = '--'
        kw.pop('label', None)
        if dashed:
            axB.ax0.loglog(aplB.freq, aplB.Quantum.asd, **kw)
        budgetD[name] = aplB.Quantum.psd

    if True:
        axBpl = mplfigB(size_in=[6.5, 3])
        budget = gwinc.load_budget(fpath_join('Ahat17' + '.yaml'))
        ifo = budget.ifo
        plot_loss_series(axBpl, ifo, name='Ahat17', label='$\\widehat{\\mathrm{A}}$ (405kW, $G_\\mathrm{int}$=17dB)', lw=2.5, color='#FF6C0C', alpha=0.3, zorder=100)

        budget = gwinc.load_budget(fpath_join('Ahat22' + '.yaml'))
        ifo = budget.ifo
        plot_loss_series(axBpl, ifo, name='Ahat22', label='$\\widehat{\\mathrm{A}}$ (405kW, $G_\\mathrm{int}$=22dB)', lw=2.5, color='#005851', alpha=0.3, zorder=90)

        # testing
        if False:
            budget = gwinc.load_budget(fpath_join('AplusTest' + '.yaml'))
            ifo = budget.ifo
            ifo.Optics.MM_INTSQZ = 0
            #ifo.intSqueezer = Struct()
            #ifo.intSqueezer.AmplitudedB = 0
            plot_loss_series(axBpl, ifo, name='Apluss', label='Apluss', lw=2.5, color='magenta', alpha=0.3, zorder=90)

        plot_other_budget(axBpl, 'Aplus', name='Apl', label='A+ (750kW)', lw=2, color='black', dashed=True)

        plot_other_budget(axBpl, fpath_join('AplWide' + '.yaml'), name='AplWB10', label='A+ wideband', lw=2, color='black', alpha=0.5)
        plot_other_budget(axBpl, fpath_join('AplWide05' + '.yaml'), name='AplWB05', lw=2, color='black', alpha=0.5)

        #budgetCE2 = gwinc.load_budget('CE2silica', freq=F_Hz)
        #ce2B = budgetCE2.run()
        #axB.ax0.loglog(ce2B.freq, ce2B.asd, label = 'CE', lw=2, color='blue')

        axBpl.ax0.set_ylim(5e-25, 3e-23)
        axBpl.ax0.set_xlim(min(F_Hz), max(F_Hz))
        axBpl.ax0.loglog(aplB.freq, CLPSD**0.5, label='A+ classical noise', color='#7A303F', dashes=[2, 1, 3, 1])
        axBpl.ax0.legend(loc='upper center', framealpha=1, fontsize=8, ncols=1)
        axBpl.ax0.set_xlabel('Frequency [Hz]')
        axBpl.ax0.set_ylabel('Strain ASD [1/$\\sqrt{\\mathrm{Hz}}$]')
        axBpl.save(tpath_join('A+cmp'))


    if True:
        budgetAsh = gwinc.load_budget(fpath_join('Asharp' + '.yaml'), freq=F_Hz)
        ashB = budgetApl.run()
        CLPSD = ashB.psd - ashB.Quantum.psd
        budgetD['AShCl'] = CLPSD

        axBsh = mplfigB(size_in=[6.5, 3])
        budget = gwinc.load_budget(fpath_join('AhatSh17' + '.yaml'))
        ifo = budget.ifo
        plot_loss_series(axBsh, ifo, name='AhatSh17', label='$\\widehat{\\mathrm{A}}^\\sharp$ (860kW, $G_\\mathrm{int}$=17dB)', lw=2.5, color='#FF6C0C', alpha=0.3, zorder=100)

        budget = gwinc.load_budget(fpath_join('AhatSh22' + '.yaml'))
        ifo = budget.ifo
        plot_loss_series(axBsh, ifo, name='AhatSh22', label='$\\widehat{\\mathrm{A}}^\\sharp$ (860kW, $G_\\mathrm{int}$=22dB)', lw=2.5, color='#005851', alpha=0.3, zorder=90)

        plot_other_budget(axBsh, fpath_join('Asharp' + '.yaml'), name='ASh', label='A$^\\sharp$ (1.5MW)', lw=2, color='black', dashed=True)

        plot_other_budget(axBsh, fpath_join('Asharp_wideband' + '.yaml'), name='AShWB05', label='A$^\\sharp$ wideband', lw=2, color='black', alpha=0.5)
        #plot_other_budget(axBsh, fpath_join('AplWide05' + '.yaml'), lw=2, color='black', alpha=0.5)


        axBsh.ax0.set_ylim(2e-25, 4e-23)
        axBsh.ax0.loglog(aplB.freq, CLPSD**0.5, label='A$^\\sharp$ classical noise', color='#7A303F', dashes=[2, 1, 3, 1])

        budgetCE2 = gwinc.load_budget('CE2silica', freq=F_Hz)
        ce2B = budgetCE2.run()
        axBsh.ax0.loglog(ce2B.freq, ce2B.asd, label='Cosmic Explorer', lw=2, color='blue', alpha = 0.3)
        budgetD['CE'] = ce2B.psd
        budgetD['CEQu'] = ce2B.Quantum.psd

        axBsh.ax0.legend(loc='upper center', framealpha=1, fontsize=8, ncols=2)
        axBsh.ax0.set_xlim(min(F_Hz), max(F_Hz))
        axBsh.ax0.set_xlabel('Frequency [Hz]')
        axBsh.ax0.set_ylabel('Strain ASD [1/$\\sqrt{\\mathrm{Hz}}$]')
        axBsh.save(tpath_join('A#cmp'))
    print(', '.join(budgetD.keys()))
    arr = np.array(list(budgetD.values())).T
    print(arr.shape)
    np.savetxt(
        tpath_join('budgets.csv'),
        arr,
        delimiter=",",
        header=', '.join(budgetD.keys()),
    )
    return



def plot_graph_CoupledCav(tpath_join):
    sfluB = sflu_CoupledCav()
    sflu = sfluB.sflu
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-8, rX=+8, Y=3, dY=-3),
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos_io(lX=-8, rX=+8, Y=3, dY=-3),
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname=tpath_join("testG.pdf"),
        scale="10pt",
    )



def test_build_CoupledCav(tpath_join):
    sfluB = sflu_CoupledCav()
    sflu = sfluB.sflu
    yamlstr = sflu.convert_self2yamlstr()
    with open(tpath_join('CoupledCavINTSQZ.yaml'), 'w') as F:
        F.write(yamlstr)
    sflu = SFLU.SFLU.convert_yamlstr2self(yamlstr)
