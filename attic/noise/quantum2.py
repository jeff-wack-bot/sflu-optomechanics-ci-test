'''Functions to calculate quantum noise

'''
import numpy as np

from ..struct import Struct
from ..optomechanicalmodels import precomp_optomechanical_plant
from ..optomechanicalmodels.common import standardize_params
from .. import const
from .. import nb

from .quantum_lib import (
    Vnorm_sq,
    adjoint,
)

pi2j = 2j * np.pi
NaN = float('NaN')






