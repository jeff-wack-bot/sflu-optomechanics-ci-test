import numpy as np
from gwinc.struct import Struct
import scipy.constants as scc
import matlib
import matplotlib.pyplot as plt
import pytest


pi2i = 2j * np.pi

params = Struct(
    Ti = 0.014,
    Te = 0,
    Larm_m = 40e3,
    Pin_W = 10,
    Plo_W = 1e8,
    Phase_rad = 0*np.pi/180,
    lambda_m = 1064e-9,
    detune_rad = 1*np.pi/180,
    M_kg = 10,
    f0_Hz = 1,
    Q = 100,
)

F_Hz = np.logspace(-1, 4, 3000)

@pytest.fixture
@matlib.optickle_model('opt_OS', params)
def opt_OS():
    import qlance.optickle as qopt
    from qlance.filters import resRoots
    eng = matlib.start_matlab_engine()

    opt = qopt.Optickle(eng, 'opt', lambda0=params.lambda_m)

    opt.addMirror('IX', Thr=params.Ti)
    opt.addMirror('EX', Thr=params.Te)
    opt.addLink('IX', 'fr', 'EX', 'fr', params.Larm_m)
    opt.addLink('EX', 'fr', 'IX', 'fr', params.Larm_m)

    p = np.array(resRoots(2*np.pi*params.f0_Hz, params.Q, Hz=False))
    opt.setMechTF('EX', [], p, 1/params.M_kg)

    detune_m = params.lambda_m * params.detune_rad / (2*np.pi)
    opt.setPosOffset('EX', detune_m)

    opt.addSource('Laser', np.sqrt(params.Pin_W)*np.exp(1j*params.Phase_rad))
    opt.addLink('Laser', 'out', 'IX', 'bk', 0)

    opt.addProbeIn('EX_DC', 'EX', 'fr', 0, 0)
    opt.addHomodyneReadout('REFL', LOpower=params.Plo_W)
    opt.addLink('IX', 'bk', 'REFL_BS', 'fr', 0)

    opt.run(F_Hz)
    return opt


def test_optickle_OS(opt_OS, tpath_join, pprint, plotTF, makegrid):
    from qlance.filters import ZPKFilter, resRoots
    pprint('Arm power: {:0.0f} W'.format(opt_OS.getSigDC('EX_DC')))
    tf = opt_OS.getTF('REFL_DIFF', 'EX')
    qnoise_W_rtHz = opt_OS.getQuantumNoise('REFL_DIFF')
    qnoise_W_m = qnoise_W_rtHz / np.abs(tf)

    fig = opt_OS.plotTF('REFL_DIFF', 'EX')
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('trans.pdf'))

    p = np.array(resRoots(params.f0_Hz, params.Q))
    plant = ZPKFilter([], p, 1/params.M_kg)
    mech_mod = opt_OS.getMechMod('EX', 'EX')

    fig = opt_OS.plotMechTF('EX', 'EX', label='RP plant')
    plant.plotFilter(F_Hz, *fig.axes, ls='--', label='Free plant')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [m/N]')
    fig.savefig(tpath_join('plant.pdf'))

    fig, ax = plt.subplots()
    ax.loglog(F_Hz, qnoise_W_m)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'Quantum noise [W/Hz$^{1/2}$]')
    makegrid(ax, F_Hz)
    fig.savefig(tpath_join('qnoise.pdf'))


par_mirr = Struct(
    Thr = 0.5,
    Pfr_W = 100,
    Pbk_W = 100,
    Plo_W = 1e8,
    phase_rad = 10*np.pi/180,
    lambda_m = 1064e-9,
    M_kg = 1e-6,
    f0_Hz = 1,
    Q = 100,
)


@pytest.fixture
@matlib.optickle_model('opt_mirr', par_mirr)
def opt_mirr():
    import qlance.optickle as qopt
    from qlance.filters import resRoots
    eng = matlib.start_matlab_engine()

    opt = qopt.Optickle(eng, 'opt', lambda0=par_mirr.lambda_m)
    opt.addMirror('mirr', Thr=par_mirr.Thr)

    p = np.array(resRoots(2*np.pi*par_mirr.f0_Hz, par_mirr.Q, Hz=False))
    opt.setMechTF('mirr', [], p, 1/par_mirr.M_kg)

    opt.addSource('Laser_fr', np.sqrt(par_mirr.Pfr_W))
    opt.addSource('Laser_bk', np.sqrt(par_mirr.Pbk_W)*np.exp(1j*par_mirr.phase_rad))
    opt.addLink('Laser_fr', 'out', 'mirr', 'fr', 0)
    opt.addLink('Laser_bk', 'out', 'mirr', 'bk', 0)

    opt.addHomodyneReadout('FR', LOpower=par_mirr.Plo_W)
    opt.addLink('mirr', 'fr', 'FR_BS', 'fr', 0)

    opt.run(F_Hz)
    return opt


def test_optickle_mirr(opt_mirr, tpath_join, pprint, makegrid):
    T = par_mirr.Thr
    R = 1 - T
    P1 = par_mirr.Pfr_W
    P2 = par_mirr.Pbk_W
    Kopt = 16*np.pi/scc.c * np.sqrt(R*T*P1*P2) * np.sin(par_mirr.phase_rad)
    fopt_Hz = Kopt / par_mirr.M_kg / (2*np.pi)
    pprint(Kopt)
    pprint(fopt_Hz)
    from qlance.filters import ZPKFilter, resRoots
    p = np.array(resRoots(par_mirr.f0_Hz, par_mirr.Q))
    plant = ZPKFilter([], p, 1/par_mirr.M_kg)

    tf = opt_mirr.getTF('FR_DIFF', 'mirr')
    qnoise_W_rtHz = opt_mirr.getQuantumNoise('FR_DIFF')
    qnoise_m_rtHz = qnoise_W_rtHz / np.abs(tf)

    fig = opt_mirr.plotTF('FR_DIFF', 'mirr')
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('refl.pdf'))

    fig = opt_mirr.plotMechTF('mirr', 'mirr', label='RP plant')
    plant.plotFilter(F_Hz, *fig.axes, ls='--', label='Free plant')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [m/N]')
    fig.savefig(tpath_join('plant.pdf'))

    fig, ax = plt.subplots()
    ax.loglog(F_Hz, qnoise_m_rtHz)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'Quantum noise [m/Hz$^{1/2}$]')
    makegrid(ax, F_Hz)
    fig.savefig(tpath_join('qnoise.pdf'))
