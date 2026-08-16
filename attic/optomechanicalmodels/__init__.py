import importlib
from copy import deepcopy


PLANT_TYPES = [
    'DRFPMI',
    'CoupledCavity',
]


def precomp_optomechanical_plant(freq, ifo, **kwargs):
    plant_type = ifo.Optics.get('plant_type', 'DRFPMI')
    if plant_type in PLANT_TYPES:
        modname = '.' + plant_type
        package = 'gwinc.optomechanicalmodels'
    else:
        modname = plant_type
        package = None
    plant_module = importlib.import_module(modname, package)
    pc_func = getattr(plant_module, 'precomp_optomechanical_plant')
    _precomp = dict()
    #pc = _precomp_recurse_mapping(pc_func, freq=freq, ifo=ifo, _precomp=_precomp)
    #pc.update(kwargs)
    #ret = pc_func(freq, ifo, **pc)
    #return ret


def plant_debug(F_Hz, ifo, plant_type='DRFPMI', FC_use=False, **pc_kwargs):
    # from ..suspension import precomp_suspension
    # from .common import standardize_params
    # sustf = precomp_suspension(F_Hz, ifo)
    # params = standardize_params(ifo)
    # if plant_type == 'DRFPMI':
    #     from . import DRFPMI
    #     if FC_use:
    #         ret = DRFPMI.DRFPMIwFilterCavity(F_Hz, ifo, sustf, params)
    #     else:
    #         ret = DRFPMI.DRFPMI(F_Hz, ifo, sustf, params)
    # elif plant_type == 'CoupledCavity':
    #     from . import CoupledCavity
    #     if FC_use:
    #         ret = CoupledCavity.CoupledCavitywFilterCavity(F_Hz, ifo, sustf, params)
    #     else:
    #         ret = CoupledCavity.CoupledCavity(F_Hz, ifo, sustf, params)
    # ret['params'] = params
    ifo = deepcopy(ifo)
    ifo.Optics.plant_type = plant_type
    if FC_use is False:
        del ifo.Squeezer
    ret = precomp_optomechanical_plant(F_Hz, ifo, **pc_kwargs)
    return ret
