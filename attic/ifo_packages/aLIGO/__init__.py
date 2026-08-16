from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE

from gwinc.noise.quantum2 import (
    Quantum,
    QuantumRelShotNoise,
    QuantumRelGamma,
)

class aLIGO(nb.Budget):

    name = 'Advanced LIGO'

    noises = [
        Quantum,
        Seismic,
        Newtonian,
        SuspensionThermal,
        CoatingBrownian,
        CoatingThermoOptic,
        SubstrateBrownian,
        SubstrateThermoElastic,
        ExcessGas,
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
