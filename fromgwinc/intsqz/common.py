import numpy as np
from gwinc.struct import Struct
from .lib import MatrixLib, matrix_stack
from gwinc.ifo.noises import ifo_power


def arm_gouyRT(ROCi_m, L_m, ROCe_m):
    """Accumulated round-trip Gouy phase of a FP cavity

    Needs to be multiplied by (2p + l) for the Gouy phase of the LGpl mode
    and by (n + m) for the Gouy phase of the HGnm mode
    """
    L = matrix_stack([[1, L_m], [0, 1]])
    M2 = matrix_stack([[1., 0.], [-2. / ROCe_m, 1]])
    M1 = matrix_stack([[1., 0.], [-2. / ROCi_m, 1]])
    M = L @ M2 @ L @ M1
    A = M[0, 0]
    B = M[0, 1]
    D = M[1, 1]
    return np.sign(B) * np.arccos((A + D)/2)



def standardize_params(ifo):
    """Determine squeezer type, if any, and extract common parameters

    Returns a struct with the following attributes:
    SQZ_DB: squeezing in dB
    ANTISQZ_DB: anti-squeezing in dB
    alpha: freq Indep Squeeze angle [rad]
    lambda_in: loss to squeezing before injection [Power]
    LO_RMS: quadrature noise [rad]
    eta_IFO: homodyne angle [rad]
    """
    params = Struct()

    ##################################################
    # squeezer parameters
    ##################################################

    if 'Squeezer' not in ifo:
        sqzType = 'None'
    elif ifo.Squeezer.AmplitudedB == 0:
        sqzType = 'None'
    else:
        sqzType = ifo.Squeezer.get('Type', 'Freq Independent')

    params.sqzType = sqzType
    if sqzType == 'None':
        params.SQZ_DB     = 0
        params.ANTISQZ_DB = 0
        params.alpha      = 0
        params.lambda_in  = 0
        params.LO_RMS     = 0
        params.SQZ_RMS    = 0
    else:
        params.SQZ_DB     = ifo.Squeezer.AmplitudedB
        params.ANTISQZ_DB = ifo.Squeezer.get('AntiAmplitudedB', params.SQZ_DB)
        params.alpha      = ifo.Squeezer.SQZAngle
        params.lambda_in  = ifo.Squeezer.InjectionLoss
        params.LO_RMS     = ifo.Squeezer.get('LOAngleRMS', 0)
        params.SQZ_RMS    = ifo.Squeezer.get('SQZAngleRMS', 0)

    ##################################################
    # Homodyne Readout phase
    ##################################################

    LO_angle_orig = ifo.Optics.get('Quadrature', Struct()).get('dc', None)

    ifoRead = ifo.get('Squeezer', Struct()).get('Readout', None)
    if ifoRead is None:
        LO_angle = LO_angle_orig
        if LO_angle_orig is None:
            raise Exception("must add Quadrature.dc or Readout...")

    elif ifoRead.Type == 'DC':
        LO_angle = (
            np.sign(ifoRead.fringe_side)
            * np.arccos((ifoRead.defect_PWR_W / ifoRead.readout_PWR_W)**.5)
        )

    elif ifoRead.Type == 'Homodyne':
        LO_angle = ifoRead.Angle
    else:
        raise Exception("Unknown Readout Type")

    if LO_angle_orig is not None and LO_angle_orig != LO_angle:
        raise Exception("Quadrature.dc inconsistent with Readout LO_angle")

    if ifoRead is not None:
        params.follow_fringe = ifoRead.get('follow_fringe', False)
    else:
        #this is not a sane default, but it is the default of GWINC
        params.follow_fringe = False

    params.LO_angle = LO_angle - np.pi/2

    ##################################################
    # Lengths
    ##################################################

    Lsec_m = ifo.Optics.SRM.CavityLength
    Lprc_m = ifo.Infrastructure.get('LengthPRC', 0)
    Lasy_m = 0
    Lavg_m = 0
    Lbsx_m = Lavg_m + Lasy_m / 2
    Lbsy_m = Lavg_m - Lasy_m / 2
    Ls_m = Lsec_m - Lavg_m
    Lp_m = Lprc_m - Lavg_m

    params.Length_m = Struct(
        ARM = ifo.Infrastructure.Length,
        BSX = Lbsx_m,
        BSY = Lbsy_m,
        SEM = Ls_m,
        PRM = Lp_m,
    )

    ###################################################
    # HOMs and mismatch
    ###################################################

    # may be better to only allow mismatch to be specified in ModeMismatch
    # and also standardize L_mm and psi_mm with other MM

    # FIXME: Need to generalize for multiple filter cavities

    MM_L = Struct(
        IFO_OMC   = ifo.Optics.get('MM_IFO_OMC', 0),
        SEC_ARM   = ifo.Optics.get('MM_ARM_SRC', 0),
        XARM_YARM = ifo.Optics.get('MM_XARM_YARM', 0),
        SEC_PRC   = ifo.Optics.get('MM_SRC_PRC', 0),
        SQZ_OMC   = ifo.get('Squeezer.MM_SQZ_OMC', 0),
        SQZ_FC    = ifo.get('Squeezer.FilterCavity.L_mm', 0),
    )
    MM_phi = Struct(
        IFO_OMC   = ifo.Optics.get('MM_IFO_OMCphi', 0),
        SEC_ARM   = ifo.Optics.get('MM_ARM_SRCphi', 0),
        XARM_YARM = ifo.Optics.get('MM_XARM_YARMphi', 0),
        SEC_PRC   = ifo.Optics.get('MM_SRC_PRCphi', 0),
        SQZ_OMC   = ifo.get('Squeezer.MM_SQZ_OMCphi', 0),
        SQZ_FC    = ifo.get('Squeezer.FilterCavity.psi_mm', 0),
    )

    # start with no HOMs and increase as non-zero mismatch losses are encountered
    nhom = 0
    params.mode_order = ifo.Optics.get('mode_order', 2)
    params.is_OPD = ifo.Optics.get('is_OPD', False)
    # if mismatch losses are specified outside of ModeMismatch,
    # they must be for a single HOM
    for MM, loss in MM_L.items():
        if not np.isscalar(loss):
            raise ValueError(
                'Can specify at most one HOM mismatch outside ModeMismatch '
                'but {:s} specifies multiple HOMs'.format(MM)
            )
        if loss > 0:
            nhom = 1
        print(MM, loss)

    if 'ModeMismatch' in ifo:
        # if nhom > 0 at this point one of the above losses has been specified
        if nhom > 0:
            raise ValueError(
                'Cannot specify mismatch in both ModeMismatch and elsewhere'
            )

        MM = ifo.ModeMismatch
        params.mode_order = np.array(MM.mode_order)
        if params.mode_order.ndim == 0:
            nhom = 1
        else:
            nhom = len(params.mode_order)

        def get_MM(key):
            MM_L[key] = MM.get(key + '_L', np.zeros(nhom))
            MM_phi[key] = MM.get(key + '_phi', np.zeros(nhom))
            # FIXME: logic to check for correct dimensions
        get_MM('IFO_OMC')
        get_MM('SEC_ARM')
        get_MM('XARM_YARM')
        get_MM('SEC_PRC')
        get_MM('SQZ_OMC')
        get_MM('SQZ_FC')
        # note that this will overwrite settings specified in Optics
        # should have some logic to deal with this
        params.is_OPD = MM.get('is_OPD', False)

    params.nhom=nhom
    mlib = MatrixLib(nhom=nhom)

    params.mlib = mlib

    params.MM = Struct()
    params.MM.SEC_ARM = mlib.MrotationMM(MM_L.SEC_ARM, MM_phi.SEC_ARM)
    params.MM.SEC_XARM = params.MM.SEC_ARM
    MM_XARM_YARM = mlib.MrotationMM(MM_L.XARM_YARM, MM_phi.XARM_YARM)
    params.MM.SEC_YARM = params.MM.SEC_XARM @ MM_XARM_YARM
    params.MM.SEC_PRC = mlib.MrotationMM(MM_L.SEC_PRC, MM_phi.SEC_PRC)
    params.MM.IFO_OMC = mlib.MrotationMM(MM_L.IFO_OMC, MM_phi.IFO_OMC)
    # If True, this is really the direct mismatch between the SQZ and OMC
    # If False, this is actually the direct mismatch between the SQZ and SEC
    # False is a questionable default. should it even be an option?
    if hasattr(ifo, 'Squeezer') and ifo.Squeezer.get('direct_mm_sqz_ifo', False):
        MM_SQZ_OMC = mlib.MrotationMM(MM_L.SQZ_OMC, MM_phi.SQZ_OMC)
        params.MM.SQZ_SEC = mlib.Minv(MM_SQZ_OMC) @ params.MM.IFO_OMC
    else:
        params.MM.SQZ_SEC = mlib.MrotationMM(MM_L.SQZ_OMC, MM_phi.SQZ_OMC)
    params.MM.SQZ_FC = mlib.MrotationMM(MM_L.SQZ_FC, MM_phi.SQZ_FC)

    ###################################################
    # Loss
    ###################################################

    params.Loss = Struct()
    arm_loss0 = ifo.Optics.Loss
    params.Loss.arm_rt = 2 * np.hstack((
        arm_loss0,
        ifo.Optics.get('ArmLossHOM', arm_loss0 * np.ones(nhom))
        ))
    params.Loss.injection = params.lambda_in
    params.Loss.readout = 1 - ifo.Optics.PhotoDetectorEfficiency

    bsloss = ifo.Optics.BSLoss
    # this mismatch is historical only used for SEC loss specification
    mismatch = (1 - ifo.Optics.coupling) + ifo.TCS.SRCloss
    loss_sec0 = 1 - (1 - mismatch) * (1 - bsloss)
    params.Loss.SEC_rt = np.hstack((
        loss_sec0,
        ifo.Optics.get('SECLossHOM', loss_sec0 * np.ones(nhom))
    ))

    ###################################################
    # Power
    ###################################################

    # seems better to get rid of arm_power_fixed eventually and just have
    # fixed if ArmPower is specified and free if Power is specified
    if ifo.Laser.get('ArmPower', None) is not None:
        if ifo.Laser.get('Power', None) is not None:
            raise ValueError(
                'Cannot specify both Laser.Power and Laser.ArmPower')
        params.Parm_W = ifo.Laser.ArmPower
        params.arm_power_fixed = True
    else:
        if ifo.Laser.get('ArmPower', None) is not None:
            raise ValueError(
                'Cannot specify both Laser.Power and Laser.ArmPower')
        params.arm_power_fixed = ifo.Laser.get('arm_power_fixed', True)
        params.Pin_W = ifo.Laser.Power
        power = ifo_power(ifo)
        params.Parm_W = power.parm
    print(f"ARM POWER {power.parm}, BS POWER: {power.pbs}, fixed arm_power {params.arm_power_fixed}")

    # optomechanical plant
    plant_type = ifo.Optics.get('Type', 'DRFPMI')
    if plant_type == 'SignalRecycled':
        plant_type = 'DRFPMI'
    params.plant_type = plant_type

    return params
