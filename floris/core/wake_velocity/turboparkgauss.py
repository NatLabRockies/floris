# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

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
from floris.core.wake_velocity.gauss import gaussian_function
from floris.utilities import (
    cosd,
    sind,
    tand,
)


@define
class TurboparkgaussVelocityDeficit(BaseModel):
    """
    Model based on the TurbOPark model with Gaussian wake profile.
    For model details see:
    Pedersen J G, Svensen E, Poulsen L, and Nygaard N G. "Turbulence Optimized
    Park model with Gaussian wake profile." Journal of Physics: Conference
    Series. Vol. 2265. No. 022063. IOP Publishing, 2020.
    doi:10.1088/1742-6596/2265/2/022063
    """

    A: float = field(default=0.04)
    include_mirror_wake: bool = field(default=True)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "u_initial": flow_field.u_initial_sorted,
            "wind_veer": flow_field.wind_veer
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
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
        wind_veer: float,
    ) -> None:

        # Initialize the velocity deficit array
        velocity_deficit = np.zeros_like(u_initial)

        downstream_mask = (x - x_i >= self.NUM_EPS)
        x_dist = (x - x_i) * downstream_mask / rotor_diameter_i

        # Characteristic wake widths from all turbines relative to turbine i
        sigma = characteristic_wake_width(
            x_dist, turbulence_intensity_i, ct_i, self.A
        ) * rotor_diameter_i

        # Peak wake deficits
        C = 1 - np.sqrt(np.clip(1 - ct_i / (8 * (sigma / rotor_diameter_i) ** 2), 0.0, 1.0))

        r_dist = np.sqrt((y - y_i) ** 2 + (z - z_i) ** 2)

        # Compute deficits for real turbines and for mirrored (image) turbines
        delta_real  = (x_dist > 0) * gaussian_function(C, r_dist, 2, sigma)
        if self.include_mirror_wake:
            r_dist_image = np.sqrt((y - y_i) ** 2 + (z - 3*z_i) ** 2)
            delta_image = (x_dist > 0) * gaussian_function(C, r_dist_image, 2, sigma)
            delta = np.hypot(delta_real, delta_image)
        else: # No mirror wakes
            delta = delta_real

        velocity_deficit = np.nan_to_num(delta)

        return velocity_deficit


def characteristic_wake_width(x_D, ambient_TI, Cts, A):
    # Parameter values taken from S. T. Frandsen, “Risø-R-1188(EN) Turbulence
    # and turbulence generated structural loading in wind turbine clusters”
    # Risø, Roskilde, Denmark, 2007.
    c1 = 1.5
    c2 = 0.8

    alpha = ambient_TI * c1
    beta = c2 * ambient_TI / np.sqrt(Cts)

    # Term for the initial width at the turbine location (denoted epsilon in Pedersen et al.)
    # Saturate term in initial width to 3.0, as is done in Orsted Matlab code.
    initial_width = 0.25 * np.sqrt(np.minimum(0.5 * (1 + np.sqrt(1 - Cts)) / np.sqrt(1 - Cts), 3.0))

    # Term for the added width downstream of the turbine
    added_width = A * ambient_TI / beta * (
        np.sqrt((alpha + beta * x_D) ** 2 + 1)
        - np.sqrt(1 + alpha ** 2)
        - np.log(
            ((np.sqrt((alpha + beta * x_D) ** 2 + 1) + 1) * alpha)
            / ((np.sqrt(1 + alpha ** 2) + 1) * (alpha + beta * x_D))
        )
    )

    sigma_w_D = initial_width + added_width

    return sigma_w_D
