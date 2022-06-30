import numpy as np
from wavestate.control.SFLU import SFLU, optics, nx2tikz
import components as cmp
from gwinc.struct import Struct
from gwinc.noise.quantum_lib import adjoint, Vnorm_sq
import matlib
import scipy.constants as scc
from copy import deepcopy
import pytest

pi2i = 2j * np.pi

params = Struct(
    Ti = 0.014,
    Te = 0,
    Larm_m = 40e3,
    Pin_W = 10,
    Plo_W = 500e-3,
    lambda_m = 1064e-9,
    detune_rad = 2*np.pi/180,
)

F_Hz = np.logspace(-1, 4, 3000)

reduce_list = [
    'EX.fr.o',
    'IX.fr.i',
    'IX.fr.o',
    'EX.fr.i',
]


@pytest.fixture
@matlib.optickle_model('opt_FP', params)
def opt_FP():
    import qlance.optickle as qopt
    eng = matlib.start_matlab_engine()

    opt = qopt.Optickle(eng, 'opt', lambda0=params.lambda_m)

    opt.addMirror('IX', Thr=params.Ti)
    opt.addMirror('EX', Thr=params.Te)
    opt.addLink('IX', 'fr', 'EX', 'fr', params.Larm_m)
    opt.addLink('EX', 'fr', 'IX', 'fr', params.Larm_m)

    detune_m = params.lambda_m * params.detune_rad / (2*np.pi)
    opt.setPosOffset('EX', detune_m)

    opt.addSource('Laser', np.sqrt(params.Pin_W))
    opt.addLink('Laser', 'out', 'IX', 'bk', 0)

    opt.addProbeIn('EX_DC', 'EX', 'fr', 0, 0)
    opt.addHomodyneReadout('REFL', LOpower=params.Plo_W)
    opt.addLink('IX', 'bk', 'REFL_BS', 'fr', 0)

    opt.run(F_Hz, noise=False)
    return opt


@pytest.fixture
def matrix_FP():
    mlib = cmp.mats_planewave
    ti = mlib.diag(np.sqrt(params.Ti))
    ri = mlib.diag(np.sqrt(1 - params.Ti))
    te = mlib.diag(np.sqrt(params.Te))
    re = mlib.diag(np.sqrt(1 - params.Te))

    delay = mlib.diag(np.exp(-pi2i * F_Hz * params.Larm_m / scc.c))
    Lmat = delay @ mlib.Mrotation(params.detune_rad)
    rt = re @ Lmat @ ri @ Lmat
    cl = mlib.Minv(mlib.Id - rt)
    refl = ri - ti @ Lmat @ re @ cl @ Lmat @ ti
    trans = ti @ Lmat @ cl

    return trans


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
    ifo['EX'].edges['fr.o', 'fr.o.pos'] = 'EX.X'
    ifo['EX'].edges['fr.i.tp', 'fr.i'] = '1'
    ifo['IX'].edges['bk.o.tp', 'bk.o'] = '1'
    ifo['IX'].edges['bk.i', 'bk.i.exc'] = '1'
    ifo['EX'].locations['fr.o.exc'] = (-5, -12)
    ifo['EX'].locations['fr.i.tp'] = (-5, 12)
    ifo['EX'].locations['fr.o.pos'] = (2, -10)
    ifo['IX'].locations['bk.o.tp'] = (5, 12)
    ifo['IX'].locations['bk.i.exc'] = (5, -12)

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        reduce_list=reduce_list,
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


@pytest.fixture
def sflu_FP_results(sflu_FP):
    sflu_FP.reduce(*reduce_list)

    EX = cmp.MirrorEdge('EX', Thr=params.Te)
    IX = cmp.MirrorEdge('IX', Thr=params.Ti)
    ArmLink = cmp.LinkEdge(
        'tau', L_m=params.Larm_m,
        detune_rad=params.detune_rad)
    mlib = EX.mlib

    edgesDC = {'1': mlib.Id}
    # edgesDC.update({'EX.X': mlib.Id})
    edgesDC.update(EX.edgesDC())
    edgesDC.update(IX.edgesDC())
    edgesDC.update(ArmLink.edgesDC())

    compDC = sflu_FP.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        {'EX.fr.i.tp'},
        {'IX.bk.i.exc': np.sqrt(params.Pin_W) * mlib.LO(np.pi/2)},
    )

    to_W = 4*np.pi/params.lambda_m
    edgesAC = {'1': mlib.Id}
    # edgesAC.update({'EX.X': to_W * mlib.Mrotation(np.pi/2) @ resultsDC['EX.fr.i.tp']})
    edgesAC.update(EX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(IX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(ArmLink.edgesAC(F_Hz))

    compAC = sflu_FP.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC = compAC.inverse_row(
        {'IX.bk.o.tp': None},
        {
            'EX.fr.o.exc',
            'EX.fr.o.pos',
        },
    )

    return resultsDC, resultsAC


def test_sflu_FP(sflu_FP_results, tpath_join, plot_tf, plotTF, pprint):
    resultsDC, resultsAC = sflu_FP_results
    pprint(resultsDC.keys())
    pprint(resultsAC.keys())
    pprint(resultsDC['EX.fr.i.tp'].shape)
    pprint(resultsAC['EX.fr.o.exc'].shape)
    pprint(resultsAC['EX.fr.o.pos'].shape)

    fig = plot_tf(F_Hz, resultsAC['EX.fr.o.exc'])
    fig.savefig(tpath_join('REFL_2p.pdf'))

    mlib = cmp.mats_planewave
    LOa = np.sqrt(params.Plo_W) * adjoint(mlib.LO(0))
    tf = LOa @ resultsAC['EX.fr.o.pos']
    tf = tf[..., 0, 0]

    # Parm_W = 4 / params.Ti * params.Pin_W
    # pprint(Parm_W)
    # to_W = 4*np.pi/params.lambda_m
    # fieldsDC = resultsDC['EX.fr.i.tp']
    # pprint(Vnorm_sq(adjoint(fieldsDC)))
    # tf = to_W * LOa @ deepcopy(resultsAC['EX.fr.o.exc']) @ mlib.Mrotation(np.pi/2) @ fieldsDC
    # tf = tf[..., 0, 0]

    fig = plotTF(F_Hz, tf)
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('REFL.pdf'))


def test_cmp_FP(
        sflu_FP_results, opt_FP, matrix_FP, tpath_join, pprint, plotTF,
        plot_tf_error):
    # Parm_W = 4 / params.Ti * params.Pin_W
    resultsDC, resultsAC = sflu_FP_results
    fieldsDC = resultsDC['EX.fr.i.tp']
    Parm_W = Vnorm_sq(adjoint(fieldsDC))
    pprint('SFLU arm power: {:0.1f} W'.format(Parm_W))
    pprint('Optickle arm power: {:0.1f} W'.format(opt_FP.getSigDC('EX_DC')))

    mlib = cmp.mats_planewave
    LOa = np.sqrt(params.Plo_W) * adjoint(mlib.LO(0))
    to_W = 4*np.pi/params.lambda_m

    sflu_tf = LOa @ resultsAC['EX.fr.o.pos']
    sflu_tf = sflu_tf[..., 0, 0]
    matrix_tf = to_W * LOa @ matrix_FP @ mlib.Mrotation(np.pi/2) @ fieldsDC
    matrix_tf = matrix_tf[..., 0, 0]
    optickle_tf = -opt_FP.getTF('REFL_DIFF', 'EX') / 2

    fig = plotTF(F_Hz, optickle_tf, label='Optickle')
    plotTF(F_Hz, sflu_tf, *fig.axes, ls='--', label='SFLU')
    plotTF(F_Hz, matrix_tf, *fig.axes, ls=':', c='C3', label='Matrix')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('tf_compare.pdf'))

    fig = plot_tf_error(F_Hz, optickle_tf, sflu_tf)
    fig.axes[0].set_ylabel('Magnitued [W/m]')
    fig.tight_layout()
    fig.savefig(tpath_join('error.pdf'))


def test_optickle_FP(opt_FP, tpath_join):
    fig = opt_FP.plotTF('REFL_DIFF', 'EX')
    fig.savefig(tpath_join('REFL.pdf'))


def plot_FP_graph(sflu_FP, tpath_join):
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
