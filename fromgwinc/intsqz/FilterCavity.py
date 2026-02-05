"""
SFLU filter cavity model
"""
import numpy as np
import os

from wield.control.SFLU import SFLU

from . import optics
from gwinc.struct import Struct
from gwinc import const


def sflu_FilterCavity():
    fpath, _ = os.path.split(__file__)
    qfile = os.path.join(fpath, 'FilterCavity.yaml')
    with open(qfile, 'r') as F:
        s = F.read()
    return SFLU.SFLU.convert_yamlstr2self(s)


def FilterCavity(
    F_Hz,
    ifo,
    params,
    use_SS=True,
):
    ######################################################################
    # Extract parameters
    ######################################################################

    fc = ifo.Squeezer.FilterCavity
    mlib = params.mlib

    Ti = fc.Ti
    Te = fc.Te
    Lfc_m = fc.L
    loss_rt = fc.Lrt
    gouy_rad = fc.get('Gouy_rad', 33 * np.pi/180)
    if mlib.nhom > 1:
        gouy_rad *= params.mode_order

    FSR_Hz = const.c / (2 * Lfc_m)
    lambda_m = ifo.Laser.Wavelength
    detune_Hz = fc.fdetune
    # increased frequency means a shorter length
    detune_m  = -detune_Hz / FSR_Hz * lambda_m / 2
    detune_rad = -2 * np.pi / lambda_m * detune_m

    ######################################################################
    # Build SFLU model
    ######################################################################

    sflu = sflu_FilterCavity()
    sflu.reduce_auto()
    edge_objs = Struct()

    edge_objs.FC1 = optics.MirrorEdge(
        name = 'FC1',
        Thr  = Ti,
        Lhr  = 0,  # loss_rt / 2,
        mlib = mlib,
    )
    edge_objs.FC2 = optics.MirrorEdge(
        name = 'FC2',
        Thr  = Te,
        Lhr  = loss_rt,
        mlib = mlib,
    )
    edge_objs.L_FC = optics.LinkEdge(
        name       = 'FC.L',
        L_m        = Lfc_m,
        detune_rad = detune_rad,
        gouy_rad   = gouy_rad,
        mlib       = mlib,
    )

    ######################################################################
    # AC calculation (DC not needed for filter cavity)
    ######################################################################

    edge_map = {
        "1": mlib.Id,
        "FC.to": params.MM.SQZ_FC,
        "FC.fr": mlib.Minv(params.MM.SQZ_FC),
    }

    inverse_row_Rmap = {
        "FC1.bk.o.tp": None
    }
    inverse_row_Cset = {
        "FC1.bk.i.exc",
        # "FC1.frL.i",
        "FC2.frL.i",
    }

    if use_SS:
        for edge_obj in edge_objs.values():
            edge_map.update(edge_obj.edgesACSS(F_Hz=F_Hz))
        compAC = sflu.SScomputer(eye=mlib.Id)
        compAC.SScompletion(edge_map)

        resultsAC = compAC.inverse_row_fresponse(
            Rmap=inverse_row_Rmap,
            Cset=inverse_row_Cset,
            F_Hz=F_Hz,
        )
        # print({k: a.shape for k, a in resultsAC.items()})

    if (not use_SS):
        for edge_obj in edge_objs.values():
            edge_map.update(edge_obj.edgesAC(F_Hz=F_Hz))

        compAC = sflu.computer(eye=mlib.Id)
        compAC.compute(edge_map=edge_map)
        resultsAC = compAC.inverse_row(
            Rmap=inverse_row_Rmap,
            Cset=inverse_row_Cset,
        )
    return dict(locals())
