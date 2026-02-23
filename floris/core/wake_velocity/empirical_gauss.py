
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
from floris.type_dec import floris_float_type
from floris.utilities import (
    cosd,
    sind,
    tand,
)


@define
class EmpiricalGaussVelocityDeficit(BaseModel):
    """
    The Empirical Gauss velocity model has a Gaussian profile
    (see :cite:`bastankhah2016experimental` and
    :cite:`King2019Controls`) throughout and expands in a (smoothed)
    piecewise linear fashion.

    parameter_dictionary (dict): Model-specific parameters.
        Default values are used when a parameter is not included
        in `parameter_dictionary`. Possible key-value pairs include:

            -   **wake_expansion_rates** (*list*): List of expansion
                rates for the Gaussian wake width. Must be of length 1
                or greater.
            -   **breakpoints_D** (*list*): List of downstream
                locations, specified in terms of rotor diameters, where
                the expansion rates go into effect. Must be one element
                shorter than wake_expansion_rates. May be empty.
            -   **sigma_0_D** (*float*): Initial width of the Gaussian
                wake at the turbine location, specified as a multiplier
                of the rotor diameter.
            -   **smoothing_length_D** (*float*): Distance over which
                the corners in the piece-wise linear wake expansion rate
                are smoothed (specified as a multiplier of the rotor
                diameter).
            -   **mixing_gain_deflection** (*float*): Gain to set the
                increase in wake expansion due to wake-induced mixing.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
    """
    wake_expansion_rates: list = field(factory=lambda: [0.023, 0.008])
    breakpoints_D: list = field(factory=lambda: [10])
    sigma_0_D: float = field(default=0.28)
    smoothing_length_D: float = field(default=2.0)
    mixing_gain_velocity: float = field(default=2.0)
    awc_mode: str = field(default="baseline")
    awc_wake_exp: float = field(default=1.2)
    awc_wake_denominator: float = field(default=400)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "wind_veer": flow_field.wind_veer
        }
        return kwargs

    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_y_i: np.ndarray,
        deflection_field_z_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        tilt_angle_i: np.ndarray,
        mixing_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        wind_veer: float
    ) -> None:
        """
        Calculates the velocity deficits in the wake.

        Args:
            x_i (np.array): Streamwise direction grid coordinates of
                the ith turbine (m).
            y_i (np.array): Cross stream direction grid coordinates of
                the ith turbine (m).
            z_i (np.array): Vertical direction grid coordinates of
                the ith turbine (m) [not used].
            axial_induction_i (np.array): Axial induction factor of the
                ith turbine (-) [not used].
            deflection_field_y_i (np.array): Horizontal wake deflections
                due to the ith turbine's yaw misalignment (m).
            deflection_field_z_i (np.array): Vertical wake deflections
                due to the ith turbine's tilt angle (m).
            yaw_angle_i (np.array): Yaw angle of the ith turbine (deg).
            tilt_angle_i (np.array): Tilt angle of the ith turbine
                (deg).
            mixing_i (np.array): The wake-induced mixing term for the
                ith turbine.
            ct_i (np.array): Thrust coefficient for the ith turbine (-).
            hub_height_i (float): Hub height for the ith turbine (m).
            rotor_diameter_i (np.array): Rotor diameter for the ith
                turbine (m).

            x (np.array): Streamwise direction grid coordinates of the
                flow field domain (m).
            y (np.array): Cross stream direction grid coordinates of the
                flow field domain (m).
            z (np.array): Vertical direction grid coordinates of the
                flow field domain (m).
            wind_veer (np.array): Wind veer (deg).

        Returns:
            np.array: Velocity deficits (-).
        """

        include_mirror_wake = True # Could add this as a user preference.

        # Only symmetric terms using yaw, but keep for consistency
        yaw_angle = -1 * yaw_angle_i

        # Initial wake widths
        sigma_y0 = self.sigma_0_D * rotor_diameter_i * cosd(yaw_angle)
        sigma_z0 = self.sigma_0_D * rotor_diameter_i * cosd(tilt_angle_i)

        # No specific near, far wakes in this model
        downstream_mask = (x > x_i + 0.1)
        upstream_mask = (x < x_i - 0.1)

        # Wake expansion in the lateral (y) and the vertical (z)
        # TODO: could compute shared components in sigma_z, sigma_y
        # with one function call.
        sigma_y = empirical_gauss_model_wake_width(
            x - x_i,
            self.wake_expansion_rates,
            [b * rotor_diameter_i for b in self.breakpoints_D], # .flatten()[0]
            sigma_y0,
            self.smoothing_length_D * rotor_diameter_i,
            self.mixing_gain_velocity * mixing_i,
        )
        sigma_y[upstream_mask] = \
            np.tile(sigma_y0, np.shape(sigma_y)[1:])[upstream_mask]

        sigma_z = empirical_gauss_model_wake_width(
            x - x_i,
            self.wake_expansion_rates,
            [b * rotor_diameter_i for b in self.breakpoints_D], # .flatten()[0]
            sigma_z0,
            self.smoothing_length_D * rotor_diameter_i,
            self.mixing_gain_velocity * mixing_i,
        )
        sigma_z[upstream_mask] = \
            np.tile(sigma_z0, np.shape(sigma_z)[1:])[upstream_mask]

        # 'Standard' wake component
        r, C = rCalt(
            wind_veer,
            sigma_y,
            sigma_z,
            y,
            y_i,
            deflection_field_y_i,
            deflection_field_z_i,
            z,
            hub_height_i,
            ct_i,
            yaw_angle,
            tilt_angle_i,
            rotor_diameter_i,
            sigma_y0,
            sigma_z0
        )
        # Normalize to match end of actuator disk model tube
        C = C / (8 * self.sigma_0_D**2 )

        wake_deficit = gaussian_function(C, r, 1, np.sqrt(0.5))

        if include_mirror_wake:
            # TODO: speed up this option by calculating various elements in
            #       rCalt only once.
            # Mirror component
            r_mirr, C_mirr = rCalt(
                wind_veer, # TODO: Is veer OK with mirror wakes?
                sigma_y,
                sigma_z,
                y,
                y_i,
                deflection_field_y_i,
                deflection_field_z_i,
                z,
                -hub_height_i, # Turbine at negative hub height location
                ct_i,
                yaw_angle,
                tilt_angle_i,
                rotor_diameter_i,
                sigma_y0,
                sigma_z0
            )
            # Normalize to match end of actuator disk model tube
            C_mirr = C_mirr / (8 * self.sigma_0_D**2)

            # ASSUME sum-of-squares superposition for the real and mirror wakes
            wake_deficit = np.sqrt(
                wake_deficit**2 +
                gaussian_function(C_mirr, r_mirr, 1, np.sqrt(0.5))**2
            )

        velocity_deficit = wake_deficit * downstream_mask

        return velocity_deficit

def rCalt(wind_veer, sigma_y, sigma_z, y, y_i, delta_y, delta_z, z, HH, Ct,
    yaw, tilt, D, sigma_y0, sigma_z0):

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
    r = ne.evaluate(
        "a * ( (y - y_i - delta_y) ** 2) - "+\
        "2 * b * (y - y_i - delta_y) * (z - HH - delta_z) + "+\
        "c * ((z - HH - delta_z) ** 2)"
    )
    d = 1 - Ct * (sigma_y0 * sigma_z0)/(sigma_y * sigma_z) * cosd(yaw) * cosd(tilt)
    C = ne.evaluate("1 - sqrt(d)")
    return r, C

def sigmoid_integral(x, center=0, width=1):
    y = np.zeros_like(x)
    # TODO: Can this be made faster?
    above_smoothing_zone = (x-center) > width/2
    y[above_smoothing_zone] = (x-center)[above_smoothing_zone]
    in_smoothing_zone = ((x-center) >= -width/2) & ((x-center) <= width/2)
    z = ((x-center)/width + 0.5)[in_smoothing_zone]
    if width.shape[0] > 1: # multiple turbine sizes
        width = np.broadcast_to(width, x.shape)[in_smoothing_zone]
    y[in_smoothing_zone] = (width*(z**6 - 3*z**5 + 5/2*z**4)).flatten()
    return y

def empirical_gauss_model_wake_width(
    x,
    wake_expansion_rates,
    breakpoints,
    sigma_0,
    smoothing_length,
    mixing_final,
    ):
    assert len(wake_expansion_rates) == len(breakpoints) + 1, \
        "Invalid combination of wake_expansion_rates and breakpoints."

    sigma = (wake_expansion_rates[0] + mixing_final) * x + sigma_0
    for ib, b in enumerate(breakpoints):
        sigma += (wake_expansion_rates[ib+1] - wake_expansion_rates[ib]) * \
            sigmoid_integral(x, center=b, width=smoothing_length)

    return sigma

def awc_added_wake_mixing(
    awc_mode_i,
    awc_amplitude_i,
    awc_frequency_i,
    awc_wake_exp,
    awc_wake_denominator
):
    # Drop surplus (grid) dimensions
    awc_amplitude_i = awc_amplitude_i[:,:,0,0]
    awc_mode_i = awc_mode_i[:,:,0,0]

    # TODO: Add TI in the mix, finetune amplitude/freq effect
    awc_mixing_factor = np.zeros_like(awc_amplitude_i, dtype=floris_float_type)
    helix_mask = awc_mode_i == 'helix'

    awc_mixing_factor[helix_mask] = (
        awc_amplitude_i[helix_mask]**awc_wake_exp/awc_wake_denominator
    )

    return awc_mixing_factor
