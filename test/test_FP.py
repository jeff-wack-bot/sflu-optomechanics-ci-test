import numpy as np
from wavestate.control.SFLU import SFLU, optics, nx2tikz
from gwinc.noise.quantum_lib import mats_planewave as mlib
from gwinc.struct import Struct
import scipy.constants as scc
import matlib
import matplotlib.pyplot as plt
from copy import deepcopy
import pytest

pi2i = 2j * np.pi

reduce_list = [
    'EX.fr.o',
    'IX.fr.i',
    'IX.fr.o',
    'EX.fr.i',
]

f_Hz = np.logspace(-1, 4, 3000)

params = Struct(
    Ti     = 0.014,
    Te     = 0,  # 5e-6,
    Larm_m = 40e3,
    Pin_W  = 1,
    f1_Hz  = 11e3,
    g1_rad = 0.1,
    Plo_W  = 1,
    lambda_m = 1064e-9,
)


@pytest.fixture
def sflu_FP():
    ifo = optics.GraphElement()
    ifo.subgraph_add(
        'IX', optics.BasisMirror(),
        translation_xy=(-10, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        'EX', optics.BasisMirror(),
        translation_xy=(10, 0),
        rotation_deg=0,
    )
    ifo.edges.update({
        ('EX.fr.i', 'IX.fr.o'): 'tau',
        ('IX.fr.i', 'EX.fr.o'): 'tau',
    })

    ifo['EX'].edges['fr.o', 'fr.o.exc'] = '1'
    ifo['IX'].edges['bk.o.tp', 'bk.o'] = '1'
    ifo['EX'].locations['fr.o.exc'] = (-5, -15)
    ifo['IX'].locations['bk.o.tp'] = (5, 15)

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        reduce_list=reduce_list,
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


@pytest.fixture
@matlib.optickle_model('opt_FP', params)
def opt_FP():
    import qlance.optickle as qopt
    eng = matlib.start_matlab_engine()

    vRF = np.array([-params.f1_Hz, 0, params.f1_Hz])
    opt = qopt.Optickle(eng, 'opt', vRF, lambda0=params.lambda_m)

    opt.addMirror('IX', Thr=params.Ti)
    opt.addMirror('EX', Thr=params.Te)
    opt.addLink('IX', 'fr', 'EX', 'fr', params.Larm_m)
    opt.addLink('EX', 'fr', 'IX', 'fr', params.Larm_m)

    opt.addSource('Laser', np.sqrt(params.Pin_W) * (vRF == 0))
    opt.addRFmodulator('Mod', params.f1_Hz, params.g1_rad*1j)
    opt.addLink('Laser', 'out', 'Mod', 'in', 0)
    opt.addLink('Mod', 'out', 'IX', 'bk', 0)

    opt.addProbeIn('EX_DC', 'EX', 'fr', 0, 0)
    opt.addHomodyneReadout('REFL', LOpower=params.Plo_W)
    opt.addLink('IX', 'bk', 'REFL_BS', 'fr', 0)

    opt.run(f_Hz, noise=False)
    return opt


@pytest.fixture
def sflu_FP_results(sflu_FP):
    sflu_FP.reduce(*reduce_list)
    ti = mlib.diag(np.sqrt(params.Ti))
    te = mlib.diag(np.sqrt(params.Te))
    ri = mlib.diag(np.sqrt(1 - params.Ti))
    re = mlib.diag(np.sqrt(1 - params.Te))

    edge_map = {
        '1': mlib.Id,
        'IX.bk.t': ti,
        'IX.fr.t': ti,
        'IX.bk.r': ri,
        'IX.fr.r': -ri,
        'EX.bk.t': te,
        'EX.fr.t': te,
        'EX.bk.r': re,
        'EX.fr.r': -re,
        'tau': mlib.diag(np.exp(-pi2i*f_Hz*params.Larm_m/scc.c)),
    }

    comp = sflu_FP.computer(eye=mlib.Id)
    comp.compute(edge_map=edge_map)
    results = comp.inverse_row(
        {'IX.bk.o.tp': None},
        {'EX.fr.o.exc'},
    )

    Parm_W = 4 / params.Ti * params.Pin_W
    REFL = deepcopy(results['EX.fr.o.exc'])[:, 1, 1]
    REFL *= 4*np.pi/params.lambda_m * np.sqrt(Parm_W)
    REFL *= np.sqrt(params.Plo_W)

    return results, REFL, Parm_W


def test_sflu_FP(sflu_FP_results, tpath_join, plot_tf, plotTF, pprint):
    results, REFL, Parm_W = sflu_FP_results

    fig = plot_tf(f_Hz, results['EX.fr.o.exc'])
    fig.savefig(tpath_join('REFL_2p.pdf'))

    pprint(Parm_W)

    fig = plotTF(f_Hz, REFL)
    fig.savefig(tpath_join('REFL.pdf'))


def test_cmp_FP(sflu_FP_results, opt_FP, tpath_join, pprint, plotTF, makegrid):
    results, sflu_tf, Parm_W = sflu_FP_results
    optickle_tf = opt_FP.getTF('REFL_DIFF', 'EX') / 2  # 2 for the homodyne
    pprint('SFLU arm power: {:0.1f} W'.format(Parm_W))
    pprint('Optickle arm power: {:0.1f} W'.format(opt_FP.getSigDC('EX_DC')))

    # fig = opt_FP.plotTF('REFL_DIFF', 'EX', label='Optickle')
    fig = plotTF(f_Hz, optickle_tf, label='Optickle')
    plotTF(f_Hz, sflu_tf, *fig.axes, ls='--', label='SFLU')
    fig.axes[0].legend()
    fig.savefig(tpath_join('REFL_compare.pdf'))

    fig, ax = plt.subplots()
    ax.semilogx(f_Hz, np.abs(optickle_tf/sflu_tf))
    makegrid(ax, f_Hz)
    ax.set_ylim(0.1, 5)
    fig.savefig(tpath_join('ratio.pdf'))


def test_optickle_FP(opt_FP, tpath_join, pprint):
    pprint(opt_FP.getSigDC('REFL_DIFF'))
    pprint(opt_FP.getSigDC('EX_DC'))
    fig = opt_FP.plotTF('REFL_DIFF', 'EX')
    fig.savefig(tpath_join('REFL.pdf'))


def plot_graph(sflu_FP, tpath_join):
    sflu = sflu_FP
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-12, rX=+12, Y=0, dY=-5)
    sflu.reduce(*reduce_list)
    sflu.graph_reduce_auto_pos(lX=-15, rX=+15, Y=-5, dY=-5)
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname = tpath_join('testG.pdf'),
        scale='10pt',
    )
