
from typing import Any, Dict

import numexpr as ne
import numpy as np
from attrs import (
    define,
    field,
    fields,
)

from floris.core import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)


NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class CurledWakeVelocityDeficit(BaseModel):
    """
    TO DO
    """

    a: float = field(default=0)
    b: float = field(default=0)
    dx_D: float = field(default=0.1)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:
        """
        This function prepares the inputs from the various FLORIS data structures
        for use in the Jensen model. This should only be used to 'initialize'
        the inputs. For any data that should be updated successively,
        do not use this function and instead pass that data directly to
        the model function.
        """
        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
        }
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i,
        rotor_diameter_i,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> None:
        """
        Placeholder, can move solution code in here eventually.
        """

        velocity_deficit = None

        return velocity_deficit
