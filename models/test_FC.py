import numpy as np
from wield.control.SFLU import SFLU, optics, nx2tikz
from gwinc.noise.quantum_lib import mats_planewave as mlib
import scipy.constants as scc
import pytest

pi2i = 2j * np.pi

locs = {
    'itm.bk.i': (-20, +5),
    'itm.bk.o': (-20, -5),
    'itm.fr.i': (-10, -5),
    'itm.fr.o': (-10, +5),
    'etm.fr.i': (+10, +5),
    'etm.fr.o': (+10, -5),
    'etm.bk.o': (+20, +5),
    'etm.bk.i': (+20, -5),
    'etm.fr.o.exc': (+10, -8),
    'itm.bk.o.tp': (-30, -5),
}

FC_matrix = {
    ('itm.fr.o', 'itm.bk.i'): 'itm.t',
    ('itm.bk.o', 'itm.fr.i'): 'itm.t',
    ('itm.bk.o', 'itm.bk.i'): 'itm.r',
    ('itm.fr.o', 'itm.fr.i'): '-itm.r',

    ('etm.fr.o', 'etm.bk.i'): 'etm.t',
    ('etm.bk.o', 'etm.fr.i'): 'etm.t',
    ('etm.bk.o', 'etm.bk.i'): 'etm.r',
    ('etm.fr.o', 'etm.fr.i'): '-etm.r',

    ('etm.fr.i', 'itm.fr.o'): 'tau',
    ('itm.fr.i', 'etm.fr.o'): 'tau',

    ('etm.fr.o', 'etm.fr.o.exc'): '1',
    ('itm.bk.o.tp', 'itm.bk.o'): '1',
}

reduce_list = [
    'etm.fr.o',
    'itm.fr.i',
    'itm.fr.o',
    'etm.fr.i',
]


@pytest.fixture
def sflu_FP():
    ifo = optics.GraphElement()
    ifo.subgraph_add(
        'itm', optics.BasisMirror(),
        translation_xy=(-10, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        'etm', optics.BasisMirror(),
        translation_xy=(10, 0),
        rotation_deg=0,
    )
    ifo.edges.update({
        ('etm.fr.i', 'itm.fr.o'): 'tau',
        ('itm.fr.i', 'etm.fr.o'): 'tau',
    })

    ifo['etm'].edges['fr.o', 'fr.o.exc'] = '1'
    ifo['itm'].edges['bk.o.tp', 'bk.o'] = '1'
    ifo['etm'].locations['fr.o.exc'] = (-5, -15)
    ifo['itm'].locations['bk.o.tp'] = (5, 15)

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        reduce_list=reduce_list,
        graph=True,
    )
    ifo.update_sflu(sflu)
    return ifo, sflu


def plot_graph_optics(sflu_FP, tpath_join):
    ifo, sflu = sflu_FP
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


def test_FC_optics(sflu_FP, tpath_join, plot_tf):
    ifo, sflu = sflu_FP
    sflu.reduce(*reduce_list)

    f_Hz = np.logspace(0, 4, 1000)

    Ti = 0.014
    Te = 5e-6
    Larm_m = 4e3

    ti = mlib.diag(np.sqrt(Ti))
    te = mlib.diag(np.sqrt(Te))
    ri = mlib.diag(np.sqrt(1 - Ti))
    re = mlib.diag(np.sqrt(1 - Te))

    edge_map = {
        '1': mlib.Id,
        'itm.fr.t': ti,
        'itm.bk.t': ti,
        'itm.fr.r': -ri,
        'itm.bk.r': ri,
        'etm.fr.t': te,
        'etm.bk.t': te,
        'etm.fr.r': -re,
        'etm.bk.r': re,
        'tau': mlib.diag(np.exp(-pi2i*f_Hz*Larm_m/scc.c)),
    }

    comp = sflu.computer(eye=mlib.Id)
    comp.compute(edge_map=edge_map)
    results = comp.inverse_row(
        {'itm.bk.o.tp': None},
        {'etm.fr.o.exc'},
    )

    fig = plot_tf(f_Hz, results['etm.fr.o.exc'])
    fig.savefig(tpath_join('REFL.pdf'))


def test_FC(tpath_join, plot_tf, pprint):
    sflu = SFLU.SFLU(
        FC_matrix
    )
    sflu.reduce(*reduce_list)

    f_Hz = np.logspace(0, 4, 1000)

    Ti = 0.014
    Te = 5e-6
    Larm_m = 4e3

    ti = np.sqrt(Ti)
    te = np.sqrt(Te)
    ri = np.sqrt(1 - Ti)
    re = np.sqrt(1 - Te)

    edge_map = {
        '1': mlib.Id,
        'itm.t': mlib.diag(ti),
        'itm.r': mlib.diag(ri),
        'etm.t': mlib.diag(te),
        'etm.r': mlib.diag(re),
        'tau': mlib.diag(np.exp(-pi2i*f_Hz*Larm_m/scc.c)),
    }
    comp = sflu.computer(eye=mlib.Id)
    comp.compute(edge_map=edge_map)

    results = comp.inverse_row(
        {'itm.bk.o.tp': None},
        {'etm.fr.o.exc'},
    )

    fig = plot_tf(f_Hz, results['etm.fr.o.exc'])
    fig.savefig(tpath_join('refl_pp.pdf'))


def plot_graph(tpath_join):
    sflu = SFLU.SFLU(FC_matrix, graph=True)
    sflu.graph_nodes_pos(locs, match=True)

    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-12, rX=+12, Y=0, dY=-5)
    sflu.reduce(*reduce_list)
    sflu.graph_reduce_auto_pos(lX=-15, rX=+15, Y=-5, dY=-5)
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname = tpath_join('testG.pdf'),
        texname = tpath_join('testG.pdf'),
        scale = '10pt',
    )
