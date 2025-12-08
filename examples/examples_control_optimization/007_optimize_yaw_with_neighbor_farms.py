"""Example: Optimize yaw with neighbor farm

This example demonstrates how to optimize the yaw angles of a subset of turbines
in order to maximize the annual energy production (AEP) of a wind farm.  In this
case, the wind farm is part of a larger collection of turbines, some of which are
part of a neighboring farm.  The optimization is performed in two ways: first by
accounting for the wakes of the neighboring farm (while not including those turbines)
in the optimization as a target of yaw angle changes or including their power
in the objective function.  In the second method the neighboring farms are removed
from FLORIS for the optimization.  The AEP is then calculated for the optimized
yaw angles (accounting for and not accounting for the neighboring farm) and compared
to the baseline AEP.
"""

import matplotlib.pyplot as plt
import numpy as np

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


# Load the wind rose from csv
wind_rose = WindRose.read_csv_long(
    "../inputs/wind_rose.csv", wd_col="wd", ws_col="ws", freq_col="freq_val", ti_col_or_value=0.06
)

# Load FLORIS
fmodel = FlorisModel("../inputs/gch.yaml")

# Specify a layout of turbines in which only the first 10 turbines are part
# of the farm to be optimized, while the others belong to a neighboring farm
X = (
    np.array(
        [
            0.0,
            756.0,
            1512.0,
            2268.0,
            3024.0,
            0.0,
            756.0,
            1512.0,
            2268.0,
            3024.0,
            0.0,
            756.0,
            1512.0,
            2268.0,
            3024.0,
            0.0,
            756.0,
            1512.0,
            2268.0,
            3024.0,
            4500.0,
            5264.0,
            6028.0,
            4878.0,
            0.0,
            756.0,
            1512.0,
            2268.0,
            3024.0,
        ]
    )
    / 1.5
)
Y = (
    np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            504.0,
            504.0,
            504.0,
            504.0,
            504.0,
            1008.0,
            1008.0,
            1008.0,
            1008.0,
            1008.0,
            1512.0,
            1512.0,
            1512.0,
            1512.0,
            1512.0,
            4500.0,
            4059.0,
            3618.0,
            5155.0,
            -504.0,
            -504.0,
            -504.0,
            -504.0,
            -504.0,
        ]
    )
    / 1.5
)

# Turbine weights: we want to only optimize for the first 10 turbines
turbine_weights = np.zeros(len(X), dtype=int)
turbine_weights[0:10] = 1.0

# Now reinitialize FLORIS layout
fmodel.set(layout_x=X, layout_y=Y)

# And visualize the floris layout
fig, ax = plt.subplots()
ax.plot(X[turbine_weights == 0], Y[turbine_weights == 0], "ro", label="Neighboring farms")
ax.plot(X[turbine_weights == 1], Y[turbine_weights == 1], "go", label="Farm subset")
ax.grid(True)
ax.set_xlabel("x coordinate (m)")
ax.set_ylabel("y coordinate (m)")
ax.legend()

# Indicate turbine 0 in the plot above with an annotation arrow
ax.annotate(
    "Turbine 0",
    (X[0], Y[0]),
    xytext=(X[0] + 100, Y[0] + 100),
    arrowprops={"facecolor": "black", "shrink": 0.05},
)

# Optimize the yaw angles.  This could be done for every wind direction and wind speed
# but in practice it is much faster to optimize only for one speed and infer the rest
# using a rule of thumb
time_series = TimeSeries(
    wind_directions=wind_rose.wind_directions, wind_speeds=8.0, turbulence_intensities=0.06
)
fmodel.set(wind_data=time_series)

# CASE 1: Optimize the yaw angles of the included farm while accounting for the
# wake effects of the neighboring farm by using turbine weights

# It's important here to do two things:
# 1. Exclude the downstream turbines from the power optimization goal via
#    turbine_weights
# 2. Prevent the optimizer from changing the yaw angles of the turbines in the
#    neighboring farm by limiting the yaw angles min max both to 0

# Set the yaw angles max min according to point(2) above
minimum_yaw_angle = np.zeros(
    (
        fmodel.n_findex,
        fmodel.n_turbines,
    )
)
maximum_yaw_angle = np.zeros(
    (
        fmodel.n_findex,
        fmodel.n_turbines,
    )
)
maximum_yaw_angle[:, :10] = 30.0


yaw_opt = YawOptimizationSR(
    fmodel=fmodel,
    minimum_yaw_angle=minimum_yaw_angle,  # Allowable yaw angles lower bound
    maximum_yaw_angle=maximum_yaw_angle,  # Allowable yaw angles upper bound
    Ny_passes=[5, 4],
    exclude_downstream_turbines=True,
    turbine_weights=turbine_weights,
)
df_opt_with_neighbor = yaw_opt.optimize()

# CASE 2: Repeat the optimization, this time ignoring the wakes of the neighboring farm
# by limiting the FLORIS model to only the turbines in the farm to be optimized
f_model_subset = fmodel.copy()
f_model_subset.set(
    layout_x=X[:10],
    layout_y=Y[:10],
)
yaw_opt = YawOptimizationSR(
    fmodel=f_model_subset,
    minimum_yaw_angle=0,  # Allowable yaw angles lower bound
    maximum_yaw_angle=30,  # Allowable yaw angles upper bound
    Ny_passes=[5, 4],
    exclude_downstream_turbines=True,
)
df_opt_without_neighbor = yaw_opt.optimize()


# Calculate the AEP in the baseline case
# Use turbine weights again to only consider the first 10 turbines power
fmodel.set(wind_data=wind_rose)
fmodel.run()
farm_power_baseline = fmodel.get_farm_power(turbine_weights=turbine_weights)
aep_baseline = fmodel.get_farm_AEP(turbine_weights=turbine_weights)


# Now need to apply the optimal yaw angles to the wind rose to get the optimized AEP
# do this by applying a rule of thumb where the optimal yaw is applied between 6 and 12 m/s
# and ramped down to 0 above and below this range

# Grab wind speeds and wind directions from the fmodel.  Note that we do this because the
# yaw angles will need to be n_findex long, and accounting for the fact that some wind
# directions and wind speeds may not be present in the wind rose (0 frequency) and aren't
# included in the fmodel
wind_directions = fmodel.wind_directions
wind_speeds = fmodel.wind_speeds
n_findex = fmodel.n_findex

yaw_angles_wind_rose_with_neighbor = np.zeros((n_findex, fmodel.n_turbines))
yaw_angles_wind_rose_without_neighbor = np.zeros((n_findex, fmodel.n_turbines))
for i in range(n_findex):
    wind_speed = wind_speeds[i]
    wind_direction = wind_directions[i]

    # Interpolate the optimal yaw angles for this wind direction from df_opt
    id_opt_with_neighbor = df_opt_with_neighbor["wind_direction"] == wind_direction
    id_opt_without_neighbor = df_opt_without_neighbor["wind_direction"] == wind_direction

    # Get the yaw angles for this wind direction
    yaw_opt_full_with_neighbor = np.array(
        df_opt_with_neighbor.loc[id_opt_with_neighbor, "yaw_angles_opt"]
    )[0]
    yaw_opt_full_without_neighbor = np.array(
        df_opt_without_neighbor.loc[id_opt_without_neighbor, "yaw_angles_opt"]
    )[0]

    # Extend the yaw angles from 10 turbine to n_turbine by filling with 0s
    # in the case of the removed neighboring farms
    yaw_opt_full_without_neighbor = np.concatenate(
        (yaw_opt_full_without_neighbor, np.zeros(fmodel.n_turbines - 10))
    )

    # Now decide what to do for different wind speeds
    wind_speed_low_no_steer = 4.0
    wind_speed_high_no_steer = 14.0
    wind_speed_low_steer = 6.0
    wind_speed_high_steer = 12.0
    if (wind_speed < wind_speed_low_no_steer) | (wind_speed > wind_speed_high_no_steer):
        yaw_opt_with_neighbor = np.zeros(fmodel.n_turbines)  # do nothing for very low/high speeds
        yaw_opt_without_neighbor = np.zeros(
            fmodel.n_turbines
        )  # do nothing for very low/high speeds
    elif wind_speed < wind_speed_low_steer:
        yaw_opt_with_neighbor = (
            yaw_opt_full_with_neighbor
            * (wind_speed_low_steer - wind_speed)
            / (wind_speed_low_steer - wind_speed_low_no_steer)
        )  # Linear ramp up
        yaw_opt_without_neighbor = (
            yaw_opt_full_without_neighbor
            * (wind_speed_low_steer - wind_speed)
            / (wind_speed_low_steer - wind_speed_low_no_steer)
        )  # Linear ramp up
    elif wind_speed > wind_speed_high_steer:
        yaw_opt_with_neighbor = (
            yaw_opt_full_with_neighbor
            * (wind_speed_high_no_steer - wind_speed)
            / (wind_speed_high_no_steer - wind_speed_high_steer)
        )  # Linear ramp down
        yaw_opt_without_neighbor = (
            yaw_opt_full_without_neighbor
            * (wind_speed_high_no_steer - wind_speed)
            / (wind_speed_high_no_steer - wind_speed_high_steer)
        )  # Linear ramp down
    else:
        yaw_opt_with_neighbor = (
            yaw_opt_full_with_neighbor  # Apply full offsets between 6.0 and 12.0 m/s
        )
        yaw_opt_without_neighbor = (
            yaw_opt_full_without_neighbor  # Apply full offsets between 6.0 and 12.0 m/s
        )

    # Save to collective array
    yaw_angles_wind_rose_with_neighbor[i, :] = yaw_opt_with_neighbor
    yaw_angles_wind_rose_without_neighbor[i, :] = yaw_opt_without_neighbor


# Now apply the optimal yaw angles and get the AEP, first accounting for the neighboring farm
fmodel.set(yaw_angles=yaw_angles_wind_rose_with_neighbor)
fmodel.run()
aep_opt_with_neighbor = fmodel.get_farm_AEP(turbine_weights=turbine_weights)
aep_uplift_with_neighbor = 100.0 * (aep_opt_with_neighbor / aep_baseline - 1)
farm_power_opt_with_neighbor = fmodel.get_farm_power(turbine_weights=turbine_weights)

# Repeat without accounting for neighboring farm
fmodel.set(yaw_angles=yaw_angles_wind_rose_without_neighbor)
fmodel.run()
aep_opt_without_neighbor = fmodel.get_farm_AEP(turbine_weights=turbine_weights)
aep_uplift_without_neighbor = 100.0 * (aep_opt_without_neighbor / aep_baseline - 1)
farm_power_opt_without_neighbor = fmodel.get_farm_power(turbine_weights=turbine_weights)

print(f"Baseline AEP: {aep_baseline / 1e9:.2f} GWh.")
print(
    f"Optimal AEP (Not accounting for neighboring farm): {aep_opt_without_neighbor / 1e9:.2f} GWh."
)
print(f"Optimal AEP (Accounting for neighboring farm): {aep_opt_with_neighbor / 1e9:.2f} GWh.")

# Plot the optimal yaw angles for turbine 0 with and without accounting for the neighboring farm
yaw_angles_0_with_neighbor = np.vstack(df_opt_with_neighbor["yaw_angles_opt"])[:, 0]
yaw_angles_0_without_neighbor = np.vstack(df_opt_without_neighbor["yaw_angles_opt"])[:, 0]

fig, ax = plt.subplots()
ax.plot(
    df_opt_with_neighbor["wind_direction"],
    yaw_angles_0_with_neighbor,
    label="Accounting for neighboring farm",
)
ax.plot(
    df_opt_without_neighbor["wind_direction"],
    yaw_angles_0_without_neighbor,
    label="Not accounting for neighboring farm",
)
ax.set_xlabel("Wind direction (deg)")
ax.set_ylabel("Yaw angle (deg)")
ax.legend()
ax.grid(True)
ax.set_title("Optimal yaw angles for turbine 0")

plt.show()
