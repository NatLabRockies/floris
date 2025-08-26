
import numpy as np

from floris.core.turbine.operation_models import POWER_SETPOINT_DISABLED
from floris.utilities import rotate_coordinates_rel_west

from .yaw_optimization_base import YawOptimization


class YawOptimizationGeometric(YawOptimization):
    """
    YawOptimizationGeometric is a subclass of
    :py:class:`floris.optimization.general_library.YawOptimization` that is
    used to provide a rough estimate of optimal yaw angles based purely on the
    wind farm geometry. Main use case is for coupled layout and yaw optimization.

    See Stanely et al. (2023) for details: https://wes.copernicus.org/articles/8/1341/2023/
    """

    def __init__(
        self,
        fmodel,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
    ):
        """
        Instantiate YawOptimizationGeometric object with a FlorisModel
        object assign parameter values.
        """

        super().__init__(
            fmodel=fmodel,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            calc_baseline_power=False
        )

    def optimize(self):
        """
        Find rough yaw angles based on wind farm geometry.
        Assumes all wind turbines have the same rotor diameter.

        Returns:
            opt_yaw_angles (np.array): Optimal yaw angles in degrees. This
            array is equal in length to the number of turbines in the farm.
        """
        # Loop through every WD individually. WS ignored!
        wd_array = self.fmodel_subset.core.flow_field.wind_directions

        active_turbines = self.fmodel_subset.core.farm.power_setpoints > POWER_SETPOINT_DISABLED
        for nwdi, wd in enumerate(wd_array):
            self._yaw_angles_opt_subset[nwdi, active_turbines[nwdi]] = geometric_yaw(
                self.fmodel_subset.layout_x[active_turbines[nwdi]],
                self.fmodel_subset.layout_y[active_turbines[nwdi]],
                wd,
                self.fmodel.core.farm.turbine_definitions[0]["rotor_diameter"],
                top_left_yaw_upper=self.maximum_yaw_angle[0, 0],
                bottom_left_yaw_upper=self.maximum_yaw_angle[0, 0],
                top_left_yaw_lower=self.minimum_yaw_angle[0, 0],
                bottom_left_yaw_lower=self.minimum_yaw_angle[0, 0],
            )

        # Finalize optimization, i.e., retrieve full solutions
        df_opt = self._finalize()

        # Otherwise, df_opt will just copy the farm_power_baseline in
        return df_opt

def geometric_yaw(
    turbine_x,
    turbine_y,
    wind_direction,
    rotor_diameter,
    left_x=0.0,
    top_left_y=1.0,
    right_x=25.0,
    top_right_y=1.0,
    top_left_yaw_upper=30.0,
    top_right_yaw_upper=0.0,
    bottom_left_yaw_upper=30.0,
    bottom_right_yaw_upper=0.0,
    top_left_yaw_lower=-30.0,
    top_right_yaw_lower=0.0,
    bottom_left_yaw_lower=-30.0,
    bottom_right_yaw_lower=0.0,
):
    """
    turbine_x: unrotated x turbine coords
    turbine_y: unrotated y turbine coords
    wind_direction: float, degrees
    rotor_diameter: float
    left_x: where we start the trapezoid. Should be left as 0.
    top_left_y: trapezoid top left coord
    right_x: where to stop the trapezoid downstream.
        Max coord after which the upstream turbine won't yaw.
    top_right_y: trapezoid top right coord
    top_left_yaw_upper: yaw angle associated with top left point (upper trapezoid)
    top_right_yaw_upper: yaw angle associated with top right point
    bottom_left_yaw_upper: yaw angle associated with bottom left point
    bottom_right_yaw_upper: yaw angle associated with bottom right point
    top_left_yaw_lower: yaw angle associated with top left point (lower trapezoid)
    top_right_yaw_lower: yaw angle associated with top right point
    bottom_left_yaw_lower: yaw angle associated with bottom left point
    bottom_right_yaw_lower: yaw angle associated with bottom right point
    """

    nturbs = len(turbine_x)
    turbine_coordinates_array = np.zeros((nturbs,3))
    turbine_coordinates_array[:,0] = turbine_x[:]
    turbine_coordinates_array[:,1] = turbine_y[:]

    rotated_x, rotated_y, _, _, _ = rotate_coordinates_rel_west(
        np.array([wind_direction]),
        turbine_coordinates_array
    )
    processed_x, processed_y = _process_layout(rotated_x[0], rotated_y[0], rotor_diameter)
    yaw_array = np.zeros(nturbs)
    for i in range(nturbs):
        # TODO: fix shape of top left yaw etc?
        yaw_array[i] = _get_yaw_angles(
            processed_x[i],
            processed_y[i],
            left_x,
            top_left_y,
            right_x,
            top_right_y,
            top_left_yaw_upper,
            top_right_yaw_upper,
            bottom_left_yaw_upper,
            bottom_right_yaw_upper,
            top_left_yaw_lower,
            top_right_yaw_lower,
            bottom_left_yaw_lower,
            bottom_right_yaw_lower,
        )

    return yaw_array

def _process_layout(
    turbine_x,
    turbine_y,
    rotor_diameter,
    spread=0.1
):
    """
    returns the distance from each turbine to the nearest downstream waked turbine
    normalized by the rotor diameter. Right now "waked" is determind by a Jensen-like
    wake spread, but this could/should be modified to be the same as the trapezoid rule
    used to determine the yaw angles.

    turbine_x: turbine x coords (rotated)
    turbine_y: turbine y coords (rotated)
    rotor_diameter: turbine rotor diameter (float)
    spread=0.1: Jensen alpha wake spread value
    """
    len(turbine_x)

    # # Intialize storage
    # dx = np.zeros(nturbs) + 1E10
    # dy = np.zeros(nturbs)

    # for waking_index in range(nturbs):
    #     for waked_index in range(nturbs):
    #         if turbine_x[waked_index] > turbine_x[waking_index]:
    #             r = spread*(turbine_x[waked_index]-turbine_x[waking_index]) + rotor_diameter/2.0
    #             if abs(turbine_y[waked_index]-turbine_y[waking_index]) < (r+rotor_diameter/2.0):
    #                 if (turbine_x[waked_index] - turbine_x[waking_index]) < dx[waking_index]:
    #                     dx[waking_index] = turbine_x[waked_index] - turbine_x[waking_index]
    #                     dy[waking_index] = turbine_y[waked_index]- turbine_y[waking_index]
    #     if dx[waking_index] == 1E10:
    #         dx[waking_index] = 0.0

    # dx_ = dx
    # dy_ = dy

    # Compute distances
    x_dists = turbine_x.reshape(-1,1).T - turbine_x.reshape(-1,1)
    y_dists = turbine_y.reshape(-1,1).T - turbine_y.reshape(-1,1)

    # Any turbines upstream or at the turbine location are ineligble
    x_dists[x_dists <= 0.] = np.inf

    # Check within Jensen model spread
    in_Jensen_wake = (abs(y_dists) < spread * x_dists + rotor_diameter)
    x_dists[~in_Jensen_wake] = np.inf

    # Get minimums (and arguments to select the correct y values also)
    dx = x_dists.min(axis=1)
    dy = y_dists[range(len(turbine_x)), x_dists.argmin(axis=1)]

    # Handle last turbine downstream
    furthest_ds_turb_idx = np.where(dx == np.inf)[0]
    dx[furthest_ds_turb_idx] = 0.
    dy[furthest_ds_turb_idx] = 0.

    return dx/rotor_diameter, dy/rotor_diameter


def _get_yaw_angles(
    x,
    y,
    left_x,
    top_left_y,
    right_x,
    top_right_y,
    top_left_yaw_upper,
    top_right_yaw_upper,
    bottom_left_yaw_upper,
    bottom_right_yaw_upper,
    top_left_yaw_lower,
    top_right_yaw_lower,
    bottom_left_yaw_lower,
    bottom_right_yaw_lower
):
    """
    _______2,5___________________________4,6
    |.......................................
    |......1,7...........................3,8
    |.......................................
    ________________________________________

    x and y: dx and dy to the nearest downstream turbine in rotor diameteters with
        turbines rotated so wind is coming left to right
    left_x: where we start the trapezoid. Should be left as 0.
    top_left_y: trapezoid top left coord
    right_x: where to stop the trapezoid downstream.
        Max coord after which the upstream turbine won't yaw.
    top_right_y: trapezoid top right coord
    top_left_yaw_upper: yaw angle associated with top left point (upper trapezoid)
    top_right_yaw_upper: yaw angle associated with top right point
    bottom_left_yaw_upper: yaw angle associated with bottom left point
    bottom_right_yaw_upper: yaw angle associated with bottom right point
    top_left_yaw_lower: yaw angle associated with top left point (lower trapezoid)
    top_right_yaw_lower: yaw angle associated with top right point
    bottom_left_yaw_lower: yaw angle associated with bottom left point
    bottom_right_yaw_lower: yaw angle associated with bottom right point
    """

    if x <= 0:
        return 0.0
    else:
        dx = (x-left_x)/(right_x-left_x)
        if dx >= 1.0:
            return 0.0
        edge_y = top_left_y + (top_right_y-top_left_y)*dx
        if abs(y) > edge_y:
            return 0.0
        elif y >= -0.01: # Tolerance to handle numerical issues
            top_yaw = top_left_yaw_upper + (top_right_yaw_upper-top_left_yaw_upper)*dx
            bottom_yaw = bottom_left_yaw_upper + (bottom_right_yaw_upper-bottom_left_yaw_upper)*dx
            return bottom_yaw + (top_yaw-bottom_yaw)*abs(y)/edge_y
        elif y < -0.01:
            top_yaw = top_left_yaw_lower + (top_right_yaw_lower-top_left_yaw_lower)*dx
            bottom_yaw = bottom_left_yaw_lower + (bottom_right_yaw_lower-bottom_left_yaw_lower)*dx
            return bottom_yaw + (top_yaw-bottom_yaw)*abs(y)/edge_y
