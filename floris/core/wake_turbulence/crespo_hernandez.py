
from typing import Any, Dict

import numexpr as ne
import numpy as np
from attrs import define, field

from floris.core import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris.utilities import cosd, sind


@define
class CrespoHernandez(BaseModel):
    """
    CrespoHernandez is a wake-turbulence model that is used to compute
    additional variability introduced to the flow field by operation of a wind
    turbine. Implementation of the model follows the original formulation and
    limitations outlined in :cite:`cht-crespo1996turbulence`.

    Note: The values for default parameters provided here differ from those in
    :cite:`cht-crespo1996turbulence. Following their recommendations, the
    default parameters would instead be:
        - initial: -0.0325*
        - constant: 0.73
        - ai: 0.8325
        - downstream: -0.32
    * The "initial" parameter is given as -0.0325 in :cite:`cht-crespo1996turbulence`,
    but the negative exponent is not clear in the scans of the paper found on the internet,
    and several subsequent paper cite the exponent as positive (0.0325). This discrepancy
    is noted in :cite:`zehtabiyan_rezaie_CH_2023`. Moreover, :cite:`zehtabiyan_rezaie_CH_2023`
    argues that positive values for this exponent are not representative of the physical
    phenomena occurring. For more details, see https://github.com/NREL/floris/issues/773.
    Nonetheless, the default value here is set to 0.1 for consistency with previous
    FLORIS versions. The default value may be updated in a future release.

    Args:
        parameter_dictionary (dict): Model-specific parameters.
            Default values are used when a parameter is not included
            in `parameter_dictionary`. Possible key-value pairs include:

            -   **initial** (*float*): The exponent on the initial ambient
                turbulence intensity.
            -   **constant** (*float*): The constant used to scale the
                wake-added turbulence intensity.
            -   **ai** (*float*): The axial induction factor exponent used
                in in the calculation of wake-added turbulence.
            -   **downstream** (*float*): The exponent applied to the
                distance downstream of an upstream turbine normalized by
                the rotor diameter used in the calculation of wake-added
                turbulence.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: cht-
    """

    initial: float = field(converter=float, default=0.1)
    constant: float = field(converter=float, default=0.9)
    ai: float = field(converter=float, default=0.8)
    downstream: float = field(converter=float, default=-0.32)

    def prepare_function(self) -> dict:
        pass

    def function(
        self,
        ambient_TI: float,
        x: np.ndarray,
        x_i: np.ndarray,
        rotor_diameter: float,
        axial_induction: np.ndarray,
    ) -> None:
        # Replace zeros and negatives with 1 to prevent nans/infs
        delta_x = x - x_i

        # TODO: ensure that these fudge factors are needed for different rotations
        upstream_mask = delta_x <= 0.1
        downstream_mask = delta_x > -0.1

        #        Keep downstream components          Set upstream to 1.0
        delta_x = delta_x * downstream_mask + np.ones_like(delta_x) * upstream_mask

        # turbulence intensity calculation based on Crespo et. al.
        constant = self.constant
        ai = self.ai
        initial = self.initial
        downstream = self.downstream
        ti = ne.evaluate(
            "constant"
            " * axial_induction ** ai"
            " * ambient_TI ** initial"
            " * (delta_x / rotor_diameter) ** downstream"
        )
        # Mask the 1 values from above with zeros
        return ti * downstream_mask
