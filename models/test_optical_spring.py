"""
Tests of SFLU radiation pressure effects with an optical spring model consisting
of a single FP cavity. The ETM is treated as a mirror with only an HR surface
for simplicity since that is all that is needed for this simple case.

Two methods are compared
* Hardcoding the Gaussian elimination on the radiation pressure loop into the
mirror definition. These are the HRMirrorRPReduced models.
* Using the full radiation pressure graph with no manual reduction. These are
the HRMirrorRP models.
"""
import numpy as np
from wield.control.SFLU import SFLU, optics, nx2tikz, SFLUcompute
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
    Ti = 0.2,
    Larm_m = 40e3,
    M_kg = 10,
    Pin_W = 100,
    Plo_W = 1,
    detune_rad = -3*np.pi/180,
    lambda_m = 1064e-9,
)

################################################################################
# Manual Gaussian elimination
################################################################################


@pytest.fixture
def sflu_HRMirrorRPReduced():
    """
    SFLU model of an FP cavity using a BasisMirror for the ITM and a mirror with
    only an HR surface for the ETM but which include radiation pressure effects
    calculated by Gaussian elimination hardcoded in the definition
    """
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
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


@pytest.fixture
def sflu_HRMirrorRPReduced_results(sflu_HRMirrorRPReduced, pprint):
    """
    Do the actual calculation for the manual Gaussian elimination of radiation
    pressure effects.

    This is made a fixture so that it can be used both to compare with Optickle and
    be used in SFLU only tests not needing Optickle or qlance to be installed
    """
    sflu = sflu_HRMirrorRPReduced
    sflu.reduce_auto()

    EX = cmp.HRMirrorRPReducedEdge('EX', Thr=0, M_kg=params.M_kg)
    IX = cmp.MirrorEdge('IX', Thr=params.Ti)
    ArmLink = cmp.LinkEdge(
        'tau', L_m=params.Larm_m,
        detune_rad=params.detune_rad,
    )
    mlib = EX.mlib

    ##################################################
    # DC calculation
    ##################################################
    edgesDC = {'1': mlib.Id}
    edgesDC.update(EX.edgesDC())
    edgesDC.update(IX.edgesDC())
    edgesDC.update(ArmLink.edgesDC())

    pprint('\nDC')

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        {
            'EX.fr.i.tp',
            'EX.fr.o.tp',
        },
        {'IX.bk.i.exc': np.sqrt(params.Pin_W) * mlib.LO(np.pi/2)}
    )
    # pprint(resultsDC['EX.fr.i.tp'])
    # pprint(resultsDC['EX.fr.o.tp'])

    ##################################################
    # AC calculation
    ##################################################
    edgesAC = {'1': mlib.Id}
    edgesAC.update(EX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(IX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(ArmLink.edgesAC(F_Hz))

    pprint('\nAC')

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
    # pprint(resultsAC_opt['EX.pos.exc'].shape)
    # pprint(resultsAC_mech['EX.fr.F.i.exc'].shape)
    # # pprint(resultsAC_mech['EX.pos.exc'].shape)

    return resultsDC, resultsAC_opt, resultsAC_mech


def test_sflu_HRMirrorRPReduced(
        sflu_HRMirrorRPReduced_results, tpath_join, pprint, plotTF):
    """
    Plot the optical response to ETM mirror motion and the radiation pressure
    modified mechanical response of the ETM.
    """
    resultsDC, resultsAC_opt, resultsAC_mech = sflu_HRMirrorRPReduced_results
    mlib = cmp.mats_planewave

    LOa = np.sqrt(params.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt['EX.pos.exc']
    opt_tf = opt_tf[..., 0, 0]
    mech_tf = resultsAC_mech['EX.fr.F.i.exc']

    fig = plotTF(F_Hz, opt_tf, label='SFLU')
    fig.axes[0].legend()
    fig.axes[0].set_title('Phase response to mirror motion')
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('optical_tf.pdf'))

    fig = plotTF(F_Hz, mech_tf, label='SFLU')
    fig.axes[0].set_title('Radiation pressure modified mechanical susceptibility')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [m/N]')
    fig.savefig(tpath_join('mechanical_tf.pdf'))


def test_HRMirrorRPReduced_compare_optickle(
        sflu_HRMirrorRPReduced_results, tpath_join, pprint, plotTF, opt_FP):
    """
    Make the same plots as test_sflu_HRMirrorRPReduced comparing with Optickle as well
    """
    resultsDC, resultsAC_opt, resultsAC_mech = sflu_HRMirrorRPReduced_results
    mlib = cmp.mats_planewave
    opt = opt_FP

    LOa = np.sqrt(params.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt['EX.pos.exc']
    opt_tf = opt_tf[..., 0, 0]
    mech_tf = resultsAC_mech['EX.fr.F.i.exc']
    opt_tf_optickle = opt.getTF('REFL_DIFF', 'EX') / 2  # factor of 2 for BHD BS
    # mechmod = opt.getMechMod('EX', 'EX')

    fig = plotTF(F_Hz, opt_tf, label='SFLU')
    plotTF(F_Hz, opt_tf_optickle, *fig.axes, ls='--', label='Optickle')
    fig.axes[0].legend()
    fig.axes[0].set_title('Phase response to mirror motion')
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('optical_tf.pdf'))

    fig = plotTF(F_Hz, mech_tf, label='SFLU')
    opt.plotMechTF('EX', 'EX', *fig.axes, ls='--', label='Optickle')
    # plotTF(F_Hz, mechmod, *fig.axes, ls=':', label='mechmod')
    fig.axes[0].set_title('Radiation pressure modified mechanical susceptibility')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [m/N]')
    fig.savefig(tpath_join('mechanical_tf.pdf'))


################################################################################
# Full graph
################################################################################


@pytest.fixture
def sflu_HRMirrorRP():
    """
    SFLU model of an FP cavity using a BasisMirror for the ITM and a mirror with
    only an HR surface for the ETM but which includes radiation pressure effects
    using a full graph
    """
    ifo = optics.GraphElement()

    ############################################################
    # basis mirror: IX
    ############################################################
    ifo.subgraph_add(
        'IX', optics.BasisMirror(),
        translation_xy=(-10, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        'EX', cmp.HRMirrorRP(),
        translation_xy=(+10, 0),
        rotation_deg=0,
    )
    ifo.edges.update({
        ("EX.fr.i", "IX.fr.o"): 'tau',
        ("IX.fr.i", "EX.fr.o"): 'tau',
    })
    ifo.locations.update({
        'IX.bk.i.exc': (-20, +5),
        'IX.bk.o.tp': (-20, -5),
    })
    ifo.edges.update({
        ('IX.bk.i', 'IX.bk.i.exc'): '1',
        ('IX.bk.o.tp', 'IX.bk.o'): '1',
    })

    ifo.node_angle['IX.bk.o.tp'] = +45

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def test_sflu_HRMirrorRP(sflu_HRMirrorRP, tpath_join, pprint, plotTF):
    """
    Solve the SFLU model defined in sflu_HRMirrorRP
    """
    sflu = sflu_HRMirrorRP
    sflu.reduce_auto()

    IX = cmp.MirrorEdge('IX', Thr=params.Ti)
    EX = cmp.HRMirrorRPEdge('EX', Thr=0, M_kg=params.M_kg)
    ArmLink = cmp.LinkEdge(
        'tau', L_m=params.Larm_m,
        detune_rad=params.detune_rad,
    )
    mlib = IX.mlib

    Id = {
        '1': mlib.Id,     # 2x2 matrix identity
        '1s': mlib.Id_s,  # 1x1 scalar identity
        '1v': mlib.Id_v,  # 2x1 vector identity
        '1a': mlib.Id_a,  # 1x2 adjoint identity
    }

    ##################################################
    # DC calculation
    ##################################################

    edgesDC = deepcopy(Id)
    edgesDC.update(IX.edgesDC())
    edgesDC.update(EX.edgesDC())
    edgesDC.update(ArmLink.edgesDC())

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        {
            'EX.fr.i.tp',
            'EX.fr.o.tp',
        },
        {'IX.bk.i.exc': np.sqrt(params.Pin_W) * mlib.LO(np.pi/2)},
    )

    ##################################################
    # AC calculation
    ##################################################

    edgesAC = deepcopy(Id)
    edgesAC.update(IX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(EX.edgesAC(F_Hz, resultsDC))
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
    )

    LOa = np.sqrt(params.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt['EX.pos.exc']
    opt_tf = opt_tf[..., 0, 0]
    mech_tf = resultsAC_mech['EX.fr.F.i.exc']
    mech_tf = mech_tf[..., 0, 0]

    fig = plotTF(F_Hz, opt_tf, label='SFLU')
    fig.axes[0].legend()
    fig.axes[0].set_title('Phase response to mirror motion')
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('optical_tf.pdf'))

    fig = plotTF(F_Hz, mech_tf, label='SFLU')
    fig.axes[0].set_title('Radiation pressure modified mechanical susceptibility')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [m/N]')
    fig.savefig(tpath_join('mechanical_tf.pdf'))


################################################################################

def plot_HRMirrorRPReduced_graph(sflu_HRMirrorRPReduced, tpath_join):
    sflu = sflu_HRMirrorRPReduced
    with open(tpath_join('opcodes.yaml'), 'w') as fh:
        fh.write(sflu.convert_self2yamlstr())
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-12, rX=+12, Y=0, dY=-5)
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos(lX=-15, rX=+15, Y=-5, dY=-5)
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname = tpath_join('testG.pdf'),
        scale='10pt',
    )


def plot_HRMirrorRP_graph(sflu_HRMirrorRP, tpath_join):
    sflu = sflu_HRMirrorRP
    with open(tpath_join('opcodes.yaml'), 'w') as fh:
        fh.write(sflu.convert_self2yamlstr())
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-12, rX=+12, Y=0, dY=-5)
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos(lX=-15, rX=+15, Y=-5, dY=-5)
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname = tpath_join('testG.pdf'),
        scale='10pt',
    )


def test_load_save_HRMirrorRPReduced(sflu_HRMirrorRPReduced, tpath_join):
    sflu = sflu_HRMirrorRPReduced
    with open(tpath_join('graph.yaml'), 'w') as fh:
        fh.write(sflu.convert_self2yamlstr())
    # can't reduce before saving graph, otherwise load won't work
    # but need to reduce to get the oplist
    sflu.reduce_auto()
    comp = sflu.computer()
    with open(tpath_join('opcode.yaml'), 'w') as fh:
        fh.write(comp.convert_oplistE2yamlstr())

    with open(tpath_join('graph.yaml'), 'r') as fh:
        ystr = fh.read()
    sflu2 = SFLU.SFLU.convert_yamlstr2self(ystr)
    with open(tpath_join('opcode.yaml'), 'r') as fh:
        ystr = fh.read()
    oplistE = SFLUcompute.SFLUCompute.convert_yamlstr2oplistE(ystr)
    sflu2.reduce_auto()
    comp2 = sflu2.computer()
    assert(sorted(comp2.oplistE) == sorted(oplistE))


################################################################################
# optickle model for comparison
################################################################################

@pytest.fixture
@matlib.optickle_model('opt_FP', params)
def opt_FP():
    """
    Optickle model for a FP cavity
    """
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
