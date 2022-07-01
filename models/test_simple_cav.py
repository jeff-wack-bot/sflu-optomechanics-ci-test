import numpy as np
from wavestate.control.SFLU import SFLU, optics, nx2tikz
import components as cmp
from gwinc.struct import Struct
from gwinc.noise.quantum_lib import adjoint, Vnorm_sq
import matlib
import scipy.constants as scc
from copy import deepcopy
import matplotlib.pyplot as plt
import pytest

pi2i = 2j * np.pi

F_Hz = np.logspace(-1, 4, 3000)


params = Struct(
    Pin_W = 10,
    Plo_W = 500e-3,
    lambda_m = 1064e-9,
    detune_rad = 1*np.pi/180,
    M_kg = 1,
    Thr = 0.014,
    Lcav_m = 40e3,
)


reduce_list = [
    'EX.fr.o',
    'IX.fr.i',
    'IX.fr.o',
    'EX.fr.i',
    'IX.bk.i',
    'IX.bk.o',
    # 'EX.pos',
    # 'EX.fr.F',
]


@pytest.fixture
def sflu_simple_cav():
    ifo = optics.GraphElement()

    ############################################################
    # basis mirror: IX
    ############################################################
    ifo.subgraph_add(
        'IX', optics.BasisMirror(),
        translation_xy=(-10, 0),
        rotation_deg=180,
    )
    ifo.locations.update({
        'IX.bk.i.exc': (-20, +5),
        'IX.bk.o.tp': (-20, -5),
    })
    ifo.edges.update({
        ('IX.bk.i', 'IX.bk.i.exc'): '1',
        ('IX.bk.o.tp', 'IX.bk.o'): '1',
    })

    ifo.node_angle['IX.bk.o.tp'] = +45

    ############################################################
    # simple mirror: EX
    ############################################################

    ifo.locations.update({
        'EX.fr.i': (8, +5),
        'EX.fr.o': (8, -5),
        'EX.pos': (14, 0),
        'EX.fr.F': (14, 4),
    })
    ifo.locations.update({
        'EX.fr.i.tp': (8, 8),
        'EX.fr.o.exc': (8, -8),
        'EX.pos.exc': (19, 0),
        })

    ifo.edges.update({
        ('EX.fr.o', 'EX.fr.i'): 'EX.fr.r',
        ('EX.fr.o', 'EX.pos'): 'EX.px',
        # ('EX.fr.o', 'EX.pos.exc'): 'EX.px',
        ('EX.fr.F', 'EX.fr.i'): 'EX.fr.Fq',
        ('EX.pos', 'EX.fr.F'): 'EX.fr.chi',
    })

    ifo.edges.update({
        ('EX.fr.i.tp', 'EX.fr.i'): '1',
        ('EX.fr.o', 'EX.fr.o.exc'): '1',
        ('EX.pos', 'EX.pos.exc'): '1s',
    })

    ifo.node_angle['EX.pos.exc'] = 45

    ############################################################
    # cavity
    ############################################################
    ifo.edges.update({
        ("EX.fr.i", "IX.fr.o"): 'tau',
        ("IX.fr.i", "EX.fr.o"): 'tau',
    })

    ############################################################
    # building
    ############################################################
    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        reduce_list=reduce_list,
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def test_sflu_simple_cav(sflu_simple_cav, tpath_join, pprint, plotTF):
    sflu = sflu_simple_cav
    sflu.reduce(*reduce_list)

    npts = len(F_Hz)

    chi = -1/(params.M_kg * (2*np.pi*F_Hz)**2)
    chi = chi.reshape((npts, 1, 1))

    IX = cmp.MirrorEdge('IX', Thr=params.Thr)
    # EX = cmp.MirrorEdge('EX', Thr=0)
    ArmLink = cmp.LinkEdge(
        'tau', L_m=params.Lcav_m,
        detune_rad=params.detune_rad,
    )
    mlib = IX.mlib

    Id = {
        '1': mlib.Id,
        '1s': mlib.Id_s,
        '1v': mlib.Id_v,
    }

    edgesDC = deepcopy(Id)
    edgesDC.update(IX.edgesDC())
    # edgesDC.update(EX.edgesDC())
    edgesDC.update(ArmLink.edgesDC())

    edgesDC['EX.fr.r'] = mlib.diag(-1)
    # edgesDC['EX.px'] = mlib.diag(0)
    # edgesDC['EX.fr.Fq'] = mlib.diag(0)
    # edgesDC['EX.fr.chi'] = mlib.diag(0)
    edgesDC['EX.px'] = np.zeros((2, 1))
    edgesDC['EX.fr.Fq'] = np.zeros((1, 2))
    edgesDC['EX.fr.chi'] = np.zeros((1, 1))

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        {'EX.fr.i.tp'},
        {'IX.bk.i.exc': np.sqrt(params.Pin_W) * mlib.LO(np.pi/2)},
    )
    pprint(resultsDC['EX.fr.i.tp'])

    edgesAC = deepcopy(Id)
    edgesAC.update(IX.edgesAC(F_Hz, resultsDC))
    # edgesAC.update(EX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(ArmLink.edgesAC(F_Hz))

    fieldsDC = resultsDC['EX.fr.i.tp']
    px = 4*np.pi/params.lambda_m * mlib.Mrotation(np.pi/2) @ fieldsDC
    pprint('px', px.shape)

    Fq = 4/scc.c * adjoint(fieldsDC)
    pprint('Fq', Fq.shape)

    edgesAC['EX.fr.r'] = mlib.diag(-1)
    edgesAC['EX.px'] = px
    edgesAC['EX.fr.Fq'] = Fq
    edgesAC['EX.fr.chi'] = chi

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC = compAC.inverse_row(
        {'IX.bk.o.tp': None},
        {
            'EX.fr.o.exc',
            'EX.pos.exc',
        },
    )
    pprint(resultsAC['EX.fr.o.exc'].shape)

    LOa = np.sqrt(params.Plo_W) * adjoint(mlib.LO(0))
    tf = LOa @ resultsAC['EX.fr.o.exc'] @ px
    pprint('tf', tf.shape)
    tf = tf[..., 0, 0]

    tf2 = LOa @ resultsAC['EX.pos.exc']
    pprint('tf2', tf2.shape)
    tf2 = tf2[..., 0, 0]

    fig = plotTF(F_Hz, tf)
    plotTF(F_Hz, tf2, *fig.axes, ls='--')
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('tf.pdf'))


@pytest.fixture
@matlib.optickle_model('opt_simple_cav', params)
def opt_simple_cav():
    import qlance.optickle as qopt
    eng = matlib.start_matlab_engine()

    opt = qopt.Optickle(eng, 'opt', lambda0=params.lambda_m)
    opt.addMirror('IX', Thr=params.Thr)
    opt.addMirror('EX', Thr=0)
    opt.addLink('IX', 'fr', 'EX', 'fr', params.Lcav_m)
    opt.addLink('EX', 'fr', 'IX', 'fr', params.Lcav_m)
    opt.setMechTF('EX', [], [0, 0], 1/params.M_kg)

    detune_m = params.lambda_m*params.detune_rad / (2*np.pi)
    opt.setPosOffset('EX', detune_m)

    opt.addSource('Laser', np.sqrt(params.Pin_W))
    opt.addLink('Laser', 'out', 'IX', 'bk', 0)

    opt.addProbeIn('EX_DC', 'EX', 'fr', 0, 0)
    opt.addHomodyneReadout('REFL', LOpower=params.Plo_W)
    opt.addLink('IX', 'bk', 'REFL_BS', 'fr', 0)

    opt.run(F_Hz)
    return opt


def test_optickle_simple_cav(opt_simple_cav, tpath_join, pprint, plotTF):
    opt = opt_simple_cav
    tf = -opt.getTF('REFL_DIFF', 'EX') / 2
    fig = plotTF(F_Hz, tf)
    fig.axes[0].set_ylabel('Magnitued [W/m]')
    fig.savefig(tpath_join('tf.pdf'))


def plot_simple_cav_graph(sflu_simple_cav, tpath_join):
    sflu = sflu_simple_cav
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
