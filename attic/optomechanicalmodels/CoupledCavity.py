"""
SFLU coupled cavity model of a DRFPMI
"""
import numpy as np
import os

from wield.control.SFLU import SFLU

from . import optics
from . import lib
from ..struct import Struct
from . import FilterCavity
from .. import nb
from ..suspension import precomp_suspension
from .common import arm_gouyRT, standardize_params


def sflu_CoupledCavity():
    fpath, _ = os.path.split(__file__)
    qfile = os.path.join(fpath, 'CoupledCavity.yaml')
    with open(qfile, 'r') as F:
        s = F.read()
    return SFLU.SFLU.convert_yamlstr2self(s)


# ports used to calculate each loss
# FIXME: add support for multiple filter cavities

loss_ports = dict(
    Arm = [
        "ETM.bk.i",
        "ETM.frL.i",
        "ITM.frL.i",
    ],
    SEC = ["SEM.frL.i"],
    FilterCavity = [
        # "FC1.frL.i",
        "FC2.frL.i"
    ],
)

strain_exc = {"ETM.pos.exc": 1/np.sqrt(2)}

def CoupledCavity(
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

    ARM_gouy_rad = arm_gouyRT(
        ifo.Optics.Curvature.ITM,
        params.Length_m.ARM,
        ifo.Optics.Curvature.ETM,
    ) / 2
    SEC_gouy_rad = ifo.Optics.SRM.get('SRCGouy_rad', 19 * np.pi/180)
    if mlib.nhom > 0:
        ARM_gouy_rad *= params.mode_order
        SEC_gouy_rad *= params.mode_order

    # convert to one-way and shift for RSE convention used here
    SEC_detune_rad = ifo.Optics.SRM.Tunephase / 2 + np.pi/2

    def suscept_m_N(F_Hz):
        """
        Factor of 2 because all of the susceptibility is put on the ETM here
        """
        return 2 * sustf.tst_suscept

    ######################################################################
    # Build SFLU model
    ######################################################################

    sflu = sflu_CoupledCavity()
    sflu.reduce_auto()
    edge_objs = Struct()

    ####################
    # optics
    ####################

    # TODO, currently a HACK using free space test mass susceptibility
    # should use AAA on the calculated suscept or turn that into an SS model

    from wield.control import SISO
    # factor of 2 because all the susceptibility is on the ETM in this approximation
    suscept_ss = SISO.zpk([], [0, 0], 2 / ifo.Suspension.Stage[0].Mass).asSS
    suscept_func = lambda F_Hz: suscept_ss.fresponse(f=F_Hz).tf

    edge_objs.ETM = optics.RPMirrorEdge(
        name     = 'ETM',
        Thr      = Te,
        Lhr      = params.Loss.arm_rt / 2,
        # suscept  = suscept_m_N,
        suscept = suscept_func,
        suscept_ss = suscept_ss,
        lambda_m = lambda_m,
        mlib     = mlib,
    )
    edge_objs.ITM = optics.MirrorEdge(
        name    = 'ITM',
        Thr     = Ti,
        Lhr     = params.Loss.arm_rt / 2,
        mlib    = mlib,
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

    edge_objs.ARM = optics.LinkEdge(
        name     = 'ARM.L',
        L_m      = params.Length_m.ARM,
        gouy_rad = ARM_gouy_rad,
        mlib     = mlib,
    )
    edge_objs.L_SEC_to = optics.LinkEdge(
        name       = 'SEC.L.to',
        L_m        = params.Length_m.SEM,
        detune_rad = SEC_detune_rad,
        gouy_rad   = SEC_gouy_rad,
        MM_to      = params.MM.SEC_ARM,
        mlib       = mlib,
    )
    edge_objs.L_SEC_fr = optics.LinkEdge(
        name       = 'SEC.L.fr',
        L_m        = params.Length_m.SEM,
        detune_rad = SEC_detune_rad,
        gouy_rad   = SEC_gouy_rad,
        MM_fr      = get_MM_fr(params.MM.SEC_ARM),
        mlib       = mlib,
    )

    cSEC = mlib.Mrotation(np.pi/2)
    edge_map = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "SEC.to": cSEC @ params.MM.SQZ_SEC,
        "SEC.fr": mlib.Minv(params.MM.IFO_OMC) @ cSEC,
    }

    ######################################################################
    # AC calculation (DC not used for coupled cavity approximation)
    ######################################################################

    # set the power at the front of the ETM to be the arm power in the
    # amplitude quadrature
    Earm_rtW = np.sqrt(params.Parm_W) * mlib.LO(np.pi/2)
    resultsDC = {
        "ETM.fr.i.tp": Earm_rtW,
        "ETM.fr.o.tp": -edge_objs.ETM.r @ Earm_rtW,
        "ETM.bk.o.tp": edge_objs.ETM.t @ Earm_rtW,
    }

    inverse_row_Rmap = {
        "SEM.bk.o.tp": None,
    }
    inverse_row_Cset = {
        "SEM.bk.i.exc",
        "ETM.pos.exc",
        "ETM.bk.i",
        "ETM.frL.i",
        "ITM.frL.i",
        "SEM.frL.i",

    }

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


def CoupledCavitywFilterCavity(
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
    ret_IFO = CoupledCavity(F_Hz, ifo, sustf, params, use_SS=use_SS)
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
    ret = CoupledCavitywFilterCavity(F_Hz, ifo, sustf, params, **kw)
    ret['loss_ports'] = loss_ports
    ret['strain_exc'] = strain_exc
    return ret
