
from typing import Any, Dict

import numpy as np
from attrs import define, field
from scipy.special import gamma

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
class CumulativeGaussCurlVelocityDeficit(BaseModel):
    """
    The cumulative curl model is an implementation of the model described in
    :cite:`cc-bay_2022`, which itself is based on the cumulative model of
    :cite:`cc-bastankhah_2021`.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: cc-
    """

    a_s: float = field(default=0.179367259)
    b_s: float = field(default=0.0118889215)
    c_s1: float = field(default=0.0563691592)
    c_s2: float = field(default=0.13290157)
    a_f: float = field(default=3.11)
    b_f: float = field(default=-0.68)
    c_f: float = field(default=2.41)
    alpha_mod: float = field(default=1.0)

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
        }
        return kwargs

    def function(
        self,
        ii: int,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        u_i: np.ndarray,
        deflection_field: np.ndarray,
        yaw_i: np.ndarray,
        turbulence_intensity: np.ndarray,
        ct: np.ndarray,
        turbine_diameter: np.ndarray,
        turb_u_wake: np.ndarray,
        Ctmp: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
    ) -> None:

        turbine_Ct = ct
        turbine_ti = turbulence_intensity
        turbine_yaw = yaw_i

        # TODO Should this be cbrt? This is done to match v2
        turb_avg_vels = np.cbrt(np.mean(u_i ** 3, axis=(2, 3), keepdims=True))

        delta_x = x - x_i

        sigma_n = wake_expansion(
            delta_x,
            turbine_Ct[:, ii:ii+1],
            turbine_ti[:, ii:ii+1],
            turbine_diameter[:, ii:ii+1],
            self.a_s,
            self.b_s,
            self.c_s1,
            self.c_s2,
        )

        y_i_loc = np.mean(y_i, axis=(2, 3), keepdims=True)
        z_i_loc = np.mean(z_i, axis=(2, 3), keepdims=True)

        x_coord = np.mean(x, axis=(2, 3), keepdims=True)

        y_loc = y
        y_coord = np.mean(y, axis=(2, 3), keepdims=True)

        z_loc = z  # np.mean(z, axis=(3,4))
        z_coord = np.mean(z, axis=(2, 3), keepdims=True)

        sum_lbda = np.zeros_like(u_initial)

        for m in range(0, ii - 1):
            x_coord_m = x_coord[:, m:m+1]
            y_coord_m = y_coord[:, m:m+1]
            z_coord_m = z_coord[:, m:m+1]

            # For computing cross planes, we don't need to compute downstream
            # turbines from out cross plane position.
            if x_coord[:, m:m+1].size == 0:
                break

            delta_x_m = x - x_coord_m

            sigma_i = wake_expansion(
                delta_x_m,
                turbine_Ct[:, m:m+1],
                turbine_ti[:, m:m+1],
                turbine_diameter[:, m:m+1],
                self.a_s,
                self.b_s,
                self.c_s1,
                self.c_s2,
            )

            S_i = sigma_n ** 2 + sigma_i ** 2

            Y_i = (y_i_loc - y_coord_m - deflection_field) ** 2 / (2 * S_i)
            Z_i = (z_i_loc - z_coord_m) ** 2 / (2 * S_i)

            lbda = 1.0 * sigma_i ** 2 / S_i * np.exp(-Y_i) * np.exp(-Z_i)

            sum_lbda = sum_lbda + lbda * (Ctmp[m] / u_initial)

        # Vectorized version of sum_lbda calc; has issues with y_coord (needs to be
        # down-selected appropriately. Prelim. timings show vectorized form takes
        # longer than for loop.)
        # if ii >= 2:
        #     S = sigma_n ** 2 + sigma_i[0:ii-1, :, :, :, :, :] ** 2
        #     Y = (y_i_loc - y_coord - deflection_field) ** 2 / (2 * S)
        #     Z = (z_i_loc - z_coord) ** 2 / (2 * S)

        #     lbda = self.alpha_mod * sigma_i[0:ii-1, :, :, :, :, :] ** 2
        #     lbda /= S * np.exp(-Y) * np.exp(-Z)
        #     sum_lbda = np.sum(lbda * (Ctmp[0:ii-1, :, :, :, :, :] / u_initial), axis=0)
        # else:
        #     sum_lbda = 0.0

        # sigma_i[ii] = sigma_n

        # blondel
        # super gaussian
        # b_f = self.b_f1 * np.exp(self.b_f2 * TI) + self.b_f3
        x_tilde = np.abs(delta_x) / turbine_diameter[:,ii:ii+1]
        r_tilde = np.sqrt( (y_loc - y_i_loc - deflection_field) ** 2 + (z_loc - z_i_loc) ** 2 )
        r_tilde /= turbine_diameter[:,ii:ii+1]

        n = self.a_f * np.exp(self.b_f * x_tilde) + self.c_f
        a1 = 2 ** (2 / n - 1)
        a2 = 2 ** (4 / n - 2)

        # based on Blondel model, modified to include cumulative effects
        tmp = a2 - (
            (n * turbine_Ct[:, ii:ii+1])
            * cosd(turbine_yaw)
            / (
                16.0
                * gamma(2 / n)
                * np.sign(sigma_n)
                * (np.abs(sigma_n) ** (4 / n))
                * (1 - sum_lbda) ** 2
            )
        )

        # for some low wind speeds, tmp can become slightly negative, which causes NANs,
        # so replace the slightly negative values with zeros
        tmp = tmp * (tmp >= 0)

        C = a1 - np.sqrt(tmp)

        C = C * (1 - sum_lbda)

        Ctmp[ii] = C

        yR = y_loc - y_i_loc
        xR = yR * tand(turbine_yaw) + x_i

        # add turbines together
        velDef = C * np.exp((-1 * r_tilde ** n) / (2 * sigma_n ** 2))

        velDef = velDef * (x - xR >= 0.1)

        turb_u_wake = turb_u_wake + turb_avg_vels * velDef
        return (turb_u_wake, Ctmp)


def wake_expansion(
    delta_x,
    ct_i,
    turbulence_intensity_i,
    rotor_diameter,
    a_s,
    b_s,
    c_s1,
    c_s2,
):
    # Calculate Beta (Eq 10, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    beta = 0.5 * (1.0 + np.sqrt(1.0 - ct_i)) / np.sqrt(1.0 - ct_i)
    k = a_s * turbulence_intensity_i + b_s
    eps = (c_s1 * ct_i + c_s2) * np.sqrt(beta)

    # Calculate sigma_tilde (Eq 9, pp 5 of ref. [1] and table 4 of ref. [2] in docstring)
    x_tilde = np.abs(delta_x) / rotor_diameter
    sigma_y = k * x_tilde + eps

    # [added dimension to get upstream values, empty, wd, ws, x, y, z  ]
    # return sigma_y[na, :, :, :, :, :, :]
    # Do this ^^ in the main function

    return sigma_y
