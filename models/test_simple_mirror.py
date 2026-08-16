import numpy as np
from wield.control.SFLU import SFLU, optics, nx2tikz
import components as cmp
from gwinc.struct import Struct
from sflu_components.quantum_lib import adjoint, Vnorm_sq
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
    'pos',
    # 'fr.F.i',
    # 'fr.F.o',
    # 'fr.o',
    # 'fr.i'
]


@pytest.fixture
def sflu_simple_mirror():
    ifo = optics.GraphElement()

    ifo.locations.update({
        'fr.i': (-8, +10),
        'fr.o': (-8, -10),
        'pos': (0, 0),
        'fr.F.i': (0, +6),
        'fr.F.o': (0, -6),
    })

    ifo.locations.update({
        'fr.i.exc': (-12, +10),
        'fr.i.tp': (-4, 14),
        'fr.o.tp': (-12, -10),
        'fr.F.i.exc': (0, +12),
        # 'fr.F.o.exc': (0, -12),
        'pos.exc': (3, -3),
        'pos.tp': (3, 3),
    })

    ifo.edges.update({
        ('fr.o', 'fr.i'): 'fr.r',
        ('fr.F.i', 'fr.i'): 'fr.Fq.i',
        ('pos', 'fr.F.i'): 'chi',
        ('pos', 'fr.F.o'): 'chi',
        ('fr.F.o', 'fr.o'): 'fr.Fq.o',
        ('fr.o', 'pos'): 'fr.qx',
    })

    ifo.edges.update({
        ('fr.i', 'fr.i.exc'): '1',
        ('fr.i.tp', 'fr.i'): '1',
        ('fr.o.tp', 'fr.o'): '1',
        ('fr.F.i', 'fr.F.i.exc'): '1a',
        # ('fr.F.o', 'fr.F.o.exc'): '1a',
        ('pos', 'pos.exc'): '1s',
        ('pos.tp', 'pos'): '1s',
    })

    ifo.node_angle['pos'] = 135
    ifo.node_angle['fr.o.tp'] = 135
    ifo.node_angle['fr.i.exc'] = 135
    ifo.node_angle['fr.i'] = 100
    # ifo.edge_handedness['fr.o', 'pos'] = 'r'

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        reduce_list=reduce_list,
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def test_sflu_simple_mirror(sflu_simple_mirror, tpath_join, pprint):
    sflu = sflu_simple_mirror
    sflu.reduce(*reduce_list)

    r = np.sqrt(1 - params.Thr)

    mlib = cmp.mats_planewave

    Id = {
        '1': mlib.Id,
        '1s': mlib.Id_s,
        '1a': mlib.Id_a,
    }

    Fq = np.zeros((1, 2))
    qx = np.zeros((2, 1))
    chi = np.zeros((1, 1))
    # prod = qx @ chi @ Fq
    # pprint(prod.shape)
    # prod2 = chi @ Fq
    # pprint(prod2.shape)
    # prod3 = chi @ mlib.Id_a
    # pprint(prod3.shape)

    edgesDC = deepcopy(Id)
    # edgesDC.update({
    #     'fr.r': mlib.diag(-r),
    #     'fr.Fq.i': np.zeros((1, 2)),
    #     'fr.Fq.o': np.zeros((1, 2)),
    #     'fr.qx': np.zeros((2, 1)),
    #     'chi': np.zeros((1, 1)),
    # })
    edgesDC.update({
        'fr.r': mlib.diag(-r),
        'fr.Fq.i': Fq,
        'fr.Fq.o': Fq,
        'fr.qx': qx,
        'chi': chi,
    })

    prod1 = edgesDC['1'] @ edgesDC['fr.qx'] @ edgesDC['chi'] @ edgesDC['fr.Fq.i'] @ edgesDC['1']
    pprint('prod1', prod1.shape)
    prod2 = edgesDC['chi'] @ edgesDC['fr.Fq.o']
    pprint('prod2', prod2.shape)
    prod3 = edgesDC['chi'] @ edgesDC['1a']
    pprint('prod3', prod3.shape)
    prod4 = edgesDC['1'] @ edgesDC['fr.qx'] @ edgesDC['1s']
    pprint('prod4', prod4.shape)

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    pprint(edgesDC.keys())
    resultsDC = compDC.inverse_col(
        {'fr.i.tp'},
        {'bk.i.exc': np.sqrt(params.Pin_W) * mlib.LO(np.pi/2)},
    )
    pprint(resultsDC.keys())


reduce_list_HRMirrorRPReduced = [
    'EX.fr.i',
    'IX.fr.i',
    'EX.fr.o',
    'IX.fr.o',
    'IX.bk.i',
    'IX.bk.o',
]

@pytest.fixture
def sflu_HRMirrorRPReduced():
    ifo = optics.GraphElement()
    ifo.subgraph_add(
        'IX', optics.BasisMirror(),
        translation_xy=(-10, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        'EX', cmp.HRMirrorRPReduced(),
        translation_xy=(+10, 0),
        rotation_deg=0,
    )
    ifo.edges.update({
        ('EX.fr.i', 'IX.fr.o'): 'tau',
        ('IX.fr.i', 'EX.fr.o'): 'tau',
    })

    ifo['IX'].locations.update({
        'bk.i.exc': (+6, -5),
        'bk.o.tp': (+6, +5),
    })
    ifo['IX'].edges.update({
        ('bk.i', 'bk.i.exc'): '1',
        ('bk.o.tp', 'bk.o'): '1',
    })

    ifo.node_angle['IX.bk.i.exc'] = 135
    ifo.node_angle['IX.bk.o.tp'] = -135

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        reduce_list=reduce_list_HRMirrorRPReduced,
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


params = Struct(
    Ti = 0.2,
    Larm_m = 40e3,
    M_kg = 10,
    Pin_W = 100,
    Plo_W = 1,
    detune_rad = -3*np.pi/180,
    lambda_m = 1064e-9,
)


def test_sflu_HRMirrorRPReduced(
        sflu_HRMirrorRPReduced, tpath_join, pprint, plotTF,
        opt_simple_cav,
):
    sflu = sflu_HRMirrorRPReduced
    opt = opt_simple_cav
    sflu.reduce(*reduce_list_HRMirrorRPReduced)

    EX = cmp.HRMirrorRPReducedEdge('EX', Thr=0, M_kg=params.M_kg)
    IX = cmp.MirrorEdge('IX', Thr=params.Ti)
    ArmLink = cmp.LinkEdge(
        'tau', L_m=params.Larm_m,
        detune_rad=params.detune_rad,
    )
    mlib = EX.mlib

    edgesDC = {'1': mlib.Id}
    edgesDC.update(EX.edgesDC())
    edgesDC.update(IX.edgesDC())
    edgesDC.update(ArmLink.edgesDC())

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        {
            'EX.fr.i.tp',
            'EX.fr.o.tp',
        },
        {'IX.bk.i.exc': np.sqrt(params.Pin_W) * mlib.LO(np.pi/2)}
    )
    pprint(resultsDC['EX.fr.i.tp'])
    pprint(resultsDC['EX.fr.o.tp'])

    edgesAC = {'1': mlib.Id}
    edgesAC.update(EX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(IX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(ArmLink.edgesAC(F_Hz))

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC_opt = compAC.inverse_row(
        {'IX.bk.o.tp': None},
        {'EX.pos.exc'},
    )
    resultsAC_mech = compAC.inverse_row(
        {'EX.pos.tp': None},
        {'EX.fr.F.i.exc'},
        # {'EX.pos.exc'},
    )
    pprint(resultsAC_opt['EX.pos.exc'].shape)
    pprint(resultsAC_mech['EX.fr.F.i.exc'].shape)
    # pprint(resultsAC_mech['EX.pos.exc'].shape)

    LOa = np.sqrt(params.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt['EX.pos.exc']
    opt_tf = opt_tf[..., 0, 0]
    mech_tf = resultsAC_mech['EX.fr.F.i.exc']
    # mech_tf = resultsAC_mech['EX.pos.exc']
    opt_tf_optickle = opt.getTF('REFL_DIFF', 'EX') / 2
    mechmod = opt.getMechMod('EX', 'EX')

    fig = plotTF(F_Hz, opt_tf, label='SFLU')
    plotTF(F_Hz, opt_tf_optickle, *fig.axes, ls='--', label='Optickle')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('optical_tf.pdf'))

    # fig = plotTF(F_Hz, mech_tf[:, 0, 0], label='SFLU')
    fig = plotTF(F_Hz, mech_tf, label='SFLU')
    opt.plotMechTF('EX', 'EX', *fig.axes, ls='--', label='Optickle')
    plotTF(F_Hz, mechmod, *fig.axes, ls=':', label='mechmod')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [m/N]')
    fig.savefig(tpath_join('mechanical_tf.pdf'))


@pytest.fixture
@matlib.optickle_model('opt_simple_cav', params)
def opt_simple_cav():
    import qlance.optickle as qopt
    eng = matlib.start_matlab_engine()

    opt = qopt.Optickle(eng, 'opt', lambda0=params.lambda_m)
    opt.addMirror('IX', Thr=params.Ti)
    opt.addMirror('EX', Thr=0)
    opt.addLink('IX', 'fr', 'EX', 'fr', params.Larm_m)
    opt.addLink('EX', 'fr', 'IX', 'fr', params.Larm_m)
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


def plot_simple_mirror_graph(sflu_simple_mirror, tpath_join):
    sflu = sflu_simple_mirror
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-12, rX=+12, Y=0, dY=-5)
    sflu.reduce(*reduce_list)
    sflu.graph_reduce_auto_pos(lX=-15, rX=+15, Y=-5, dY=-5)
    G2 = sflu.G.copy()

    node_labels = {
        'fr.i':   r'$fr_{\mathrm{in}}$',
        'fr.o':   r'$fr_{\mathrm{out}}$',
        'fr.F.i': r'$F_i$',
        'fr.F.o': r'$F_o$',
        'pos':    r'$x$',
    }
    edge_labels = {
        ('fr.i', 'fr.o'):   r'$-r$',
        ('fr.i', 'fr.F.i'): r'$F_q^{i}$',
        ('fr.o', 'fr.F.o'): r'$F_q^{o}$',
        ('fr.F.i', 'pos'):  r'$\chi$',
        ('fr.F.o', 'pos'):  r'$\chi$',
        ('pos', 'fr.o'):    r'$q_x$',
    }

    def is_aux(n):
        s = str(n)
        return s.endswith('.tp') or s.endswith('.exc')

    for G in (G1, G2):
        for n in [n for n in G.nodes if is_aux(n)]:
            G.remove_node(n)
        for n, lbl in node_labels.items():
            if n in G.nodes:
                G.nodes[n]['label'] = lbl
        for (u, v), lbl in edge_labels.items():
            if G.has_edge(u, v):
                G.edges[u, v]['label'] = lbl

    nx2tikz.dump_pdf(
        [G1, G2],
        fname = tpath_join('testG.pdf'),
        scale='10pt',
    )


def plot_doubled_simple_mirror_graph(tpath_join):
    """Pedagogical 2-quadrature view of the perfect-reflectivity optomechanical
    mirror. Each optical node (fr.in, fr.out) is doubled: a Q1 (amplitude) copy
    and a Q2 (phase) copy stacked with a small diagonal offset. Mechanical
    nodes (force, position) are scalar and remain singular.

    Layout: optical IO on the left (input on top, output on bottom);
    mechanical chain extends to the right.

    The radiation-pressure loop visibly *crosses quadratures*:
        fr.in_q1 --F_q--> F_in --chi--> x --q_x--> fr.out_q2
    i.e. an amplitude-quadrature fluctuation drives motion, which radiates
    out as a phase-quadrature fluctuation. Q2 has no F_q edge because, with
    the carrier chosen along Q1, phase fluctuations are decoupled from
    radiation-pressure force at linear order.
    """
    import networkx as nx

    G = nx.DiGraph()

    dx, dy = 0.6, 0.6  # diagonal stack offset (Q2 sits up-and-right of Q1)
    IO_DIST = 6.0      # length of dangling IO edges (matches fig 1)

    # Optical nodes: doubled (Q1 = front, Q2 = back/offset).
    # Layout is rotated 90° vs the usual: input on top, output on bottom,
    # both on the LEFT of the figure.
    G.add_node('fr.in.q1',  pos=(-8, +6), shape='nodeS')
    G.add_node('fr.in.q2',  pos=(-8 + dx, +6 + dy), shape='nodeS', angle=45)
    G.add_node('fr.out.q1', pos=(-8, -6), shape='nodeS')
    G.add_node('fr.out.q2', pos=(-8 + dx, -6 + dy), shape='nodeS')

    # Dangling IO endpoints (hidden coordinates, one per quadrature copy)
    # extending to the LEFT of the mirror surface nodes.
    G.add_node('fr.in.q1.exc',  pos=(-8 - IO_DIST,        +6),      shape='coordinate')
    G.add_node('fr.in.q2.exc',  pos=(-8 + dx - IO_DIST,   +6 + dy), shape='coordinate')
    G.add_node('fr.out.q1.tp',  pos=(-8 - IO_DIST,        -6),      shape='coordinate')
    G.add_node('fr.out.q2.tp',  pos=(-8 + dx - IO_DIST,   -6 + dy), shape='coordinate')
    # Identify each input quadrature on its dangling edge (output side
    # inherits identity via the within-quadrature -r reflection, so the
    # q_1 / q_2 labels appear only once each).
    G.add_edge('fr.in.q1.exc',  'fr.in.q1', label=r'$q_1$', handed='r')
    G.add_edge('fr.in.q2.exc',  'fr.in.q2', label=r'$q_2$')
    G.add_edge('fr.out.q1', 'fr.out.q1.tp')
    G.add_edge('fr.out.q2', 'fr.out.q2.tp')

    # Mechanical chain: a single force node F and the position node x.
    # All radiation pressure is bookkept on the *incoming* (top) side;
    # this lets the back-action loop read as one clean arc from the carrier
    # (top, Q1) down through x and out to the orthogonal quadrature (bottom, Q2).
    G.add_node('F',   pos=(0, +2), shape='nodeS', label=r'$F$', angle=45)
    G.add_node('pos', pos=(0, -2), shape='nodeS', label=r'$x$')

    # Reflection: within-quadrature on each copy. The label is shown only
    # on the Q1 edge to avoid duplication; the Q2 edge is drawn unlabeled.
    G.add_edge('fr.in.q1', 'fr.out.q1', label=r'$-r$', handed='r')
    G.add_edge('fr.in.q2', 'fr.out.q2')

    # Radiation-pressure force: applied entirely at the input (top) node.
    # Only the amplitude quadrature (Q1) drives F — the Q2 edge is omitted
    # because F_q · (Q2 fluctuation) = 0 in the carrier-aligned basis.
    G.add_edge('fr.in.q1', 'F', label=r'$F^{q_1}$', bend=-20)

    # Mechanical susceptibility (force -> displacement)
    G.add_edge('F', 'pos', label=r'$\chi$')

    # Position back-action: motion radiates into the *phase* quadrature (Q2)
    # of the outgoing field — closes the arc back to the bottom-left stack.
    G.add_edge('pos', 'fr.out.q2', label=r'$q_2^{x}$', bend=-20)

    nx2tikz.dump_pdf(
        [G],
        fname=tpath_join('testG.pdf'),
        scale='10pt',
    )


def plot_HRMirrorRPReduced_graph(sflu_HRMirrorRPReduced, tpath_join):
    sflu = sflu_HRMirrorRPReduced
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-12, rX=+12, Y=0, dY=-5)
    sflu.reduce(*reduce_list_HRMirrorRPReduced)
    sflu.graph_reduce_auto_pos(lX=-15, rX=+15, Y=-5, dY=-5)
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname = tpath_join('testG.pdf'),
        scale='10pt',
    )
