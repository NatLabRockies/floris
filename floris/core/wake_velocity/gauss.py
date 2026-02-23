
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
from floris.utilities import (
    cosd,
    sind,
    tand,
)


@define
class GaussVelocityDeficit(BaseModel):

    alpha: float = field(default=0.58)
    beta: float = field(default=0.077)
    ka: float = field(default=0.38)
    kb: float = field(default=0.004)

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

        # yaw_angle is all turbine yaw angles for each wind speed
        # Extract and broadcast only the current turbine yaw setting
        # for all wind speeds

        # Opposite sign convention in this model
        yaw_angle = -1 * yaw_angle_i

        # Initialize the velocity deficit
        uR = u_initial * ct_i / (2.0 * (1 - np.sqrt(1 - ct_i)))
        u0 = u_initial * np.sqrt(1 - ct_i)

        # Initial lateral bounds
        sigma_z0 = rotor_diameter_i * 0.5 * np.sqrt(uR / (u_initial + u0))
        sigma_y0 = sigma_z0 * cosd(yaw_angle) * cosd(wind_veer)

        # Compute the bounds of the near and far wake regions and a mask

        # Start of the near wake
        xR = x_i

        # Start of the far wake
        x0 = np.ones_like(u_initial)
        x0 *= rotor_diameter_i * cosd(yaw_angle) * (1 + np.sqrt(1 - ct_i) )
        x0 /= np.sqrt(2) * (
            4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i) )
        )
        x0 += x_i

        # Initialize the velocity deficit array
        velocity_deficit = np.zeros_like(u_initial)

        # Masks
        # When we have only an inequality, the current turbine may be applied its own
        # wake in cases where numerical precision cause in incorrect comparison. We've
        # applied a small bump to avoid this. "0.1" is arbitrary but it is a small, non
        # zero value.

        # This mask defines the near wake; keeps the areas downstream of xR and upstream of x0
        near_wake_mask = (x > xR + 0.1) * (x < x0)
        far_wake_mask = (x >= x0)

        # Compute the velocity deficit in the NEAR WAKE region
        # ONLY If there are points within the near wake boundary
        # TODO: for the TurbineGrid, do we need to do this near wake calculation at all?
        #       same question for any grid with a resolution larger than the near wake region
        if np.sum(near_wake_mask):

            # Calculate the wake expansion

            # This is a linear ramp from 0 to 1 from the start of the near wake to the start
            # of the far wake.
            near_wake_ramp_up = (x - xR) / (x0 - xR)
            # Another linear ramp, but positive upstream of the far wake and negative in the
            # far wake; 0 at the start of the far wake
            near_wake_ramp_down = (x0 - x) / (x0 - xR)
            # near_wake_ramp_down = -1 * (near_wake_ramp_up - 1)  # : this is equivalent, right?

            sigma_y = near_wake_ramp_down * 0.501 * rotor_diameter_i * np.sqrt(ct_i / 2.0)
            sigma_y += near_wake_ramp_up * sigma_y0
            sigma_y *= (x >= xR)
            sigma_y += np.ones_like(sigma_y) * (x < xR) * 0.5 * rotor_diameter_i

            sigma_z = near_wake_ramp_down * 0.501 * rotor_diameter_i * np.sqrt(ct_i / 2.0)
            sigma_z += near_wake_ramp_up * sigma_z0
            sigma_z *= (x >= xR)
            sigma_z += np.ones_like(sigma_z) * (x < xR) * 0.5 * rotor_diameter_i

            r_squared, C = rC(
                wind_veer,
                sigma_y,
                sigma_z,
                y,
                y_i,
                deflection_field_i,
                z,
                hub_height_i,
                ct_i,
                yaw_angle,
                rotor_diameter_i,
            )

            near_wake_deficit = gaussian_function(C, r_squared, 1, np.sqrt(0.5))
            near_wake_deficit *= near_wake_mask

            velocity_deficit += near_wake_deficit

        # Compute the velocity deficit in the FAR WAKE region
        if np.sum(far_wake_mask):

            # Wake expansion in the lateral (y) and the vertical (z)
            ky = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
            kz = self.ka * turbulence_intensity_i + self.kb  # wake expansion parameters
            sigma_y = (ky * (x - x0) + sigma_y0) * far_wake_mask + sigma_y0 * (x < x0)
            sigma_z = (kz * (x - x0) + sigma_z0) * far_wake_mask + sigma_z0 * (x < x0)

            r_squared, C = rC(
                wind_veer,
                sigma_y,
                sigma_z,
                y,
                y_i,
                deflection_field_i,
                z,
                hub_height_i,
                ct_i,
                yaw_angle,
                rotor_diameter_i,
            )

            far_wake_deficit = gaussian_function(C, r_squared, 1, np.sqrt(0.5))
            far_wake_deficit *= far_wake_mask

            velocity_deficit += far_wake_deficit

        return velocity_deficit


# @profile
def rC(wind_veer, sigma_y, sigma_z, y, y_i, delta, z, HH, Ct, yaw, D):

    ## original
    # a = cosd(wind_veer) ** 2 / (2 * sigma_y ** 2) + sind(wind_veer) ** 2 / (2 * sigma_z ** 2)
    # b = -sind(2 * wind_veer) / (4 * sigma_y ** 2) + sind(2 * wind_veer) / (4 * sigma_z ** 2)
    # c = sind(wind_veer) ** 2 / (2 * sigma_y ** 2) + cosd(wind_veer) ** 2 / (2 * sigma_z ** 2)
    # r_squared = (
    #     a * (y - y_i - delta) ** 2
    #     - 2 * b * (y - y_i - delta) * (z - HH)
    #     + c * (z - HH) ** 2
    # )
    # C = 1 - np.sqrt(np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / D ** 2)), 0.0, 1.0))

    ## Precalculate some parts
    # twox_sigmay_2 = 2 * sigma_y ** 2
    # twox_sigmaz_2 = 2 * sigma_z ** 2
    # a = cosd(wind_veer) ** 2 / (twox_sigmay_2) + sind(wind_veer) ** 2 / (twox_sigmaz_2)
    # b = -sind(2 * wind_veer) / (2 * twox_sigmay_2) + sind(2 * wind_veer) / (2 * twox_sigmaz_2)
    # c = sind(wind_veer) ** 2 / (twox_sigmay_2) + cosd(wind_veer) ** 2 / (twox_sigmaz_2)
    # delta_y = y - y_i - delta
    # delta_z = z - HH
    # r_squared = (a * (delta_y ** 2) - 2 * b * (delta_y) * (delta_z) + c * (delta_z ** 2))
    # C = 1 - np.sqrt(np.clip(1 - (Ct * cosd(yaw) / (8.0 * sigma_y * sigma_z / (D * D))), 0.0, 1.0))

    ## Numexpr
    wind_veer = np.deg2rad(wind_veer)
    a = ne.evaluate(
        "cos(wind_veer) ** 2 / (2 * sigma_y ** 2) + sin(wind_veer) ** 2 / (2 * sigma_z ** 2)"
    )
    b = ne.evaluate(
        "-sin(2 * wind_veer) / (4 * sigma_y ** 2) + sin(2 * wind_veer) / (4 * sigma_z ** 2)"
    )
    c = ne.evaluate(
        "sin(wind_veer) ** 2 / (2 * sigma_y ** 2) + cos(wind_veer) ** 2 / (2 * sigma_z ** 2)"
    )
    r_squared = ne.evaluate(
        "a * ((y - y_i - delta) ** 2) - 2 * b * (y - y_i - delta) * (z - HH) + c * ((z - HH) ** 2)"
    )
    d = np.clip(1 - (Ct * cosd(yaw) / ( 8.0 * sigma_y * sigma_z / (D * D) )), 0.0, 1.0)
    C = ne.evaluate("1 - sqrt(d)")
    return r_squared, C


def mask_upstream_wake(mesh_y_rotated, x_coord_rotated, y_coord_rotated, turbine_yaw):
    yR = mesh_y_rotated - y_coord_rotated
    xR = yR * tand(turbine_yaw) + x_coord_rotated
    return xR, yR


def gaussian_function(C, r_squared, n, sigma):
    result = ne.evaluate("C * exp(-1 * r_squared ** n / (2 * sigma ** 2))")
    return result
