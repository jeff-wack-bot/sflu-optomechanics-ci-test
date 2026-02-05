"""
SFLU DRFPMI model
"""
import numpy as np
import os
from copy import deepcopy

from wield.control.SFLU import SFLU

from . import optics
from . import lib
from ..struct import Struct
from .. import nb
from ..suspension import precomp_suspension
from .common import arm_gouyRT, standardize_params
from . import FilterCavity


def sflu_DRFPMI():
    fpath, _ = os.path.split(__file__)
    qfile = os.path.join(fpath, 'DRFPMI.yaml')
    with open(qfile, 'r') as F:
        s = F.read()
    return SFLU.SFLU.convert_yamlstr2self(s)


def plant_debug(F_Hz, ifo, FC_use=False):
    sustf = precomp_suspension(F_Hz, ifo)
    params = standardize_params(ifo)
    if FC_use:
        ret = DRFPMIwFilterCavity(F_Hz, ifo, sustf, params)
    else:
        ret = DRFPMI(F_Hz, ifo, sustf, params)
    return ret


# ports used to calculate each loss
# FIXME: add support for multiple filter cavities

loss_ports = dict(
    Arm = [
        "EX.bk.i",
        "EY.bk.i",
        "EX.frL.i",
        "EY.frL.i",
        "IX.frL.i",
        "IY.frL.i",
    ],
    SEC = ["SEM.frL.i"],
    FilterCavity = [
        # "FC1.frL.i",
        "FC2.frL.i"
    ],
)

strain_exc = {
    "EX.pos.exc": 1/2,
    "EY.pos.exc": -1/2,
}


def DRFPMI(
    F_Hz,
    ifo,
    sustf,
    params,
    use_SS=True,
):
    ######################################################################
    # Extract parameters
    ######################################################################

    mlib = params.mlib
    lambda_m = ifo.Laser.Wavelength
    Ti = ifo.Optics.ITM.Transmittance
    Te = ifo.Optics.ETM.Transmittance
    Ts = ifo.Optics.SRM.Transmittance
    Tp = ifo.Optics.PRM.Transmittance

    ARM_gouy_rad = arm_gouyRT(
        ifo.Optics.Curvature.ITM,
        params.Length_m.ARM,
        ifo.Optics.Curvature.ETM,
    ) / 2
    SEC_gouy_rad = ifo.Optics.SRM.get('SRCGouy_rad', 19 * np.pi/180)
    PRC_gouy_rad = ifo.Optics.PRM.get('PRCGouy_rad', 25 * np.pi/180)
    if mlib.nhom > 0:
        ARM_gouy_rad *= params.mode_order
        SEC_gouy_rad *= params.mode_order
        PRC_gouy_rad *= params.mode_order

    # convert to one-way and shift for RSE convention used here
    SEC_detune_rad = ifo.Optics.SRM.Tunephase / 2 + np.pi/2

    # use the exact DC fields if True
    # need exact for state space calculations
    exact = True

    # # use RP on both ITM and ETM for 'BOTH' and only ETM for 'ETM'
    # rp = 'BOTH'

    # def suscept_m_N(F_Hz, tm):
    #     if rp == 'BOTH':
    #         return sustf.tst_suscept
    #     elif rp == 'ETM':
    #         if tm == 'ETM':
    #             return 2 * sustf.tst_suscept
    #         elif tm == 'ITM':
    #             return np.zeros_like(sustf.tst_suscept)

    ######################################################################
    # Build SFLU model
    ######################################################################

    sflu = sflu_DRFPMI()
    sflu.reduce_auto()
    edge_objs = Struct()

    ####################
    # optics
    ####################

    # TODO, currently a HACK using free space test mass susceptibility
    # should use AAA on the calculated suscept or turn that into an SS model

    from wield.control import SISO
    suscept_ss = SISO.zpk([], [0, 0], 1/ifo.Suspension.Stage[0].Mass).asSS
    suscept_func = lambda F_Hz: suscept_ss.fresponse(f=F_Hz).tf

    for opt_name in ['EX', 'EY']:
        edge_objs[opt_name] = optics.RPMirrorEdge(
            name     = opt_name,
            Thr      = Te,
            Lhr      = params.Loss.arm_rt / 2,
            # suscept  = lambda F_Hz: suscept_m_N(F_Hz, 'ETM'),
            suscept  = suscept_func,
            suscept_ss = suscept_ss,
            lambda_m = lambda_m,
            mlib     = mlib,
        )
    for opt_name in ['IX', 'IY']:
        edge_objs[opt_name] = optics.RPMirrorEdge(
            name     = opt_name,
            Thr      = Ti,
            Lhr      = params.Loss.arm_rt / 2,
            # suscept  = lambda F_Hz: suscept_m_N(F_Hz, 'ITM'),
            suscept  = suscept_func,
            suscept_ss = suscept_ss,
            lambda_m = lambda_m,
            mlib     = mlib,
        )
    edge_objs.BS = optics.BSEdge(
        name = 'BS',
        Thr  = 0.5,
        mlib = mlib,
    )
    # FIXME: add PRC loss
    edge_objs.PRM = optics.MirrorEdge(
        name = 'PRM',
        Thr  = Tp,
        mlib = mlib,
    )
    edge_objs.SEM = optics.MirrorEdge(
        name = 'SEM',
        Thr  = Ts,
        Lhr  = params.Loss.SEC_rt,
        mlib = mlib,
    )

    ####################
    # links
    ####################

    def get_MM_fr(MM):
        if params.is_OPD:
            return MM
        else:
            return mlib.Minv(MM)

    for link_name in ['XARM', 'YARM']:
        edge_objs[link_name] = optics.LinkEdge(
            name     = link_name + '.L',
            L_m      = params.Length_m.ARM,
            gouy_rad = ARM_gouy_rad,
            mlib     = mlib,
        )
    edge_objs.L_BSX_to = optics.LinkEdge(
        name  = 'BSX.L.to',
        L_m   = params.Length_m.BSX,
        MM_to = params.MM.SEC_XARM,
        mlib  = mlib,
    )
    edge_objs.L_BSX_fr= optics.LinkEdge(
        name  = 'BSX.L.fr',
        L_m   = params.Length_m.BSX,
        MM_fr = get_MM_fr(params.MM.SEC_XARM),
        mlib  = mlib,
    )
    edge_objs.L_BSY_to = optics.LinkEdge(
        name  = 'BSY.L.to',
        L_m   = params.Length_m.BSY,
        MM_to = params.MM.SEC_YARM,
        mlib  = mlib,
    )
    edge_objs.L_BSY_fr = optics.LinkEdge(
        name  = 'BSY.L.fr',
        L_m   = params.Length_m.BSY,
        MM_fr = get_MM_fr(params.MM.SEC_YARM),
        mlib  = mlib,
    )
    edge_objs.L_SEC_to = optics.LinkEdge(
        name       = 'SEC.L.to',
        L_m        = params.Length_m.SEM,
        detune_rad = SEC_detune_rad,
        gouy_rad   = SEC_gouy_rad,
        mlib       = mlib,
    )
    edge_objs.L_SEC_fr = optics.LinkEdge(
        name       = 'SEC.L.fr',
        L_m        = params.Length_m.SEM,
        detune_rad = SEC_detune_rad,
        gouy_rad   = SEC_gouy_rad,
        mlib       = mlib,
    )
    edge_objs.L_PRC_to = optics.LinkEdge(
        name     = 'PRC.L.to',
        L_m      = params.Length_m.PRM,
        gouy_rad = PRC_gouy_rad,
        MM_to    = params.MM.SEC_PRC,
        mlib     = mlib,
    )
    edge_objs.L_PRC_fr = optics.LinkEdge(
        name     = 'PRC.L.fr',
        L_m      = params.Length_m.PRM,
        gouy_rad = PRC_gouy_rad,
        MM_fr    = mlib.Minv(params.MM.SEC_PRC),
        mlib     = mlib,
    )

    cSEC = mlib.Mrotation(np.pi/2)
    edge_map = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "SEC.to": cSEC @ params.MM.SQZ_SEC,
        "SEC.fr": mlib.Minv(params.MM.IFO_OMC) @ cSEC,
    }

    ######################################################################
    # DC calculation
    ######################################################################

    if exact:
        edgesDC = deepcopy(edge_map)
        for edge_obj in edge_objs.values():
            edgesDC.update(edge_obj.edgesDC())

        # DC test points
        tp_dc = {
            '{:}.{:}.{:}.tp'.format(opt, side, port)
            for opt in ['IX', 'EX', 'IY', 'EY']
            for side in ['fr', 'bk']
            for port in ['o', 'i']
        }
        tp_dc.remove("EX.bk.i.tp")
        tp_dc.remove("EY.bk.i.tp")

        if params.arm_power_fixed:
            Ein_rtW = mlib.LO(np.pi/2)
        else:
            Ein_rtW = np.sqrt(params.Pin_W) * mlib.LO(np.pi/2)

        compDC = sflu.computer(eye=mlib.Id)
        compDC.compute(edge_map=edgesDC)
        resultsDC = compDC.inverse_col(
            tp_dc,
            {
                "PRM.bk.i.exc": Ein_rtW,
            },
        )

        # set arm power if necessary
        if params.arm_power_fixed:
            Pxarm_W = lib.Vnorm_sq(resultsDC["EX.fr.i.tp"])
            Pyarm_W = lib.Vnorm_sq(resultsDC["EY.fr.i.tp"])
            avg_arm_power = (Pxarm_W + Pyarm_W) / 2
            power_correction = np.sqrt(params.Parm_W / avg_arm_power)
            for k, v in resultsDC.items():
                resultsDC[k] = v * power_correction
    else:
        Earm_rtW = np.sqrt(params.Parm_W) * mlib.LO(np.pi/2)
        fri = Earm_rtW
        fro = -edge_objs.EX.r @ Earm_rtW
        bko = edge_objs.EX.t @ Earm_rtW
        # negative for Y b/c of phase change from the BS
        resultsDC = {
            "EX.fr.i.tp": fri,
            "EY.fr.i.tp": -fri,
            "EX.fr.o.tp": fro,
            "EY.fr.o.tp": -fro,
            "EX.bk.o.tp": bko,
            "EY.bk.o.tp": -bko,
        }

    ######################################################################
    # AC calculation
    ######################################################################

    inverse_row_Rmap = {
        "SEM.bk.o.tp": None,
    }
    inverse_row_Cset = {
        "SEM.bk.i.exc",
        "EX.pos.exc",
        "EY.pos.exc",
        "EX.bk.i",
        "EY.bk.i",
        "EX.frL.i",
        "EY.frL.i",
        "IX.frL.i",
        "IY.frL.i",
        "SEM.frL.i",
    }

    edgesAC = deepcopy(edge_map)
    if use_SS:
        for edge_obj in edge_objs.values():
            edge_map.update(edge_obj.edgesACSS(F_Hz=F_Hz, resultsDC=resultsDC))
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
            edge_map.update(edge_obj.edgesAC(F_Hz=F_Hz, resultsDC=resultsDC))

        compAC = sflu.computer(eye=mlib.Id)
        compAC.compute(edge_map=edge_map)
        resultsAC = compAC.inverse_row(
            Rmap=inverse_row_Rmap,
            Cset=inverse_row_Cset,
        )

    return dict(locals())


def DRFPMIwFilterCavity(
    F_Hz,
    ifo,
    sustf,
    params,
    use_SS=True,
):
    mlib = params.mlib
    mats = lib.MatsHelper()
    mats.H['AS'] = mlib.Id

    # injection loss
    L_inj_t = (1 - params.Loss.injection)**0.5
    mats.update_scalar(L_inj_t)
    mats.T['Loss_injection'] = mlib.Id * params.Loss.injection**0.5

    # FIXME: add support for multiple filter cavities
    # FIXME: better deal with no filter cavity
    # filter cavity
    if 'Squeezer' in ifo:
        ret_FC = FilterCavity.FilterCavity(F_Hz, ifo, params, use_SS=use_SS)
        results_FC = ret_FC['resultsAC']
        mats.update_matrix(results_FC["FC1.bk.i.exc"])
        mats.T.update({k: v for k, v in results_FC.items() if k != "FC1.bk.i.exc"})

    # IFO
    ret_IFO = DRFPMI(F_Hz, ifo, sustf, params, use_SS=use_SS)
    results_IFO = ret_IFO['resultsAC']
    mats.update_matrix(results_IFO["SEM.bk.i.exc"])
    mats.T.update({k: v for k, v in results_IFO.items() if k != "SEM.bk.i.exc"})

    # FIXME add output filter cavities

    # readout loss
    L_read_t = (1 - params.Loss.readout)**0.5
    mats.update_scalar(L_read_t)
    mats.T['Loss_readout'] = mlib.diag(params.Loss.readout**0.5)

    return dict(locals())


@nb.precomp(sustf=precomp_suspension)
def precomp_optomechanical_plant(F_Hz, ifo, sustf, **kw):
    params = standardize_params(ifo)
    ret = DRFPMIwFilterCavity(F_Hz, ifo, sustf, params, **kw)
    ret['loss_ports'] = loss_ports
    ret['strain_exc'] = strain_exc
    return ret
