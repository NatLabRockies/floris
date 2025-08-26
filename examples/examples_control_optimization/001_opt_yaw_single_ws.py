
"""Example: Optimize yaw for a single wind speed and multiple wind directions

Use the serial-refine method to optimize the yaw angles for a 3-turbine wind farm

"""

import matplotlib.pyplot as plt
import numpy as np

import floris.flow_visualization as flowviz
import floris.layout_visualization as layoutviz
from floris import FlorisModel, TimeSeries
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


# Load the default example floris object
fmodel = FlorisModel("../inputs/gch.yaml")

# Define an inflow that
# keeps wind speed and TI constant while sweeping the wind directions
wind_directions = np.arange(0.0, 360.0, 3.0)
time_series = TimeSeries(
    wind_directions=wind_directions,
    wind_speeds=8.0,
    turbulence_intensities=0.06,
)

# Reinitialize as a 3-turbine using the above inflow
D = 126.0 # Rotor diameter for the NREL 5 MW
fmodel.set(
    layout_x=[0.0, 5 * D, 10 * D],
    layout_y=[0.0, 0.0, 0.0],
    wind_data=time_series,
)

# Initialize optimizer object and run optimization using the Serial-Refine method
yaw_opt = YawOptimizationSR(fmodel)
df_opt = yaw_opt.optimize()

print("Optimization results:")
print(df_opt)

# Split out the turbine results
for t in range(3):
    df_opt['t%d' % t] = df_opt.yaw_angles_opt.apply(lambda x: x[t])

# Show the results
fig, axarr = plt.subplots(2,1,sharex=True,sharey=False,figsize=(8,8))

# Yaw results
ax = axarr[0]
for t in range(3):
    ax.plot(df_opt.wind_direction,df_opt['t%d' % t],label='t%d' % t)
ax.set_ylabel('Yaw Offset (deg')
ax.legend()
ax.grid(True)

# Power results
ax = axarr[1]
ax.plot(df_opt.wind_direction,df_opt.farm_power_baseline,color='k',label='Baseline Farm Power')
ax.plot(df_opt.wind_direction,df_opt.farm_power_opt,color='r',label='Optimized Farm Power')
ax.set_ylabel('Power (W)')
ax.set_xlabel('Wind Direction (deg)')
ax.legend()
ax.grid(True)

# Visualize results for a single wind direction (270 deg) and wind speed (8 m/s)
fig, axarr = plt.subplots(2, 1, figsize=(10, 5), sharex=False)
ax = axarr[0] # Baseline aligned operation
fmodel.reset_operation()
fmodel.set(wind_directions=[270.0], wind_speeds=[8.0], turbulence_intensities=[0.06])
fmodel.run()
horizontal_plane = fmodel.calculate_horizontal_plane(height=90.0)
flowviz.visualize_cut_plane(horizontal_plane, ax=ax)
layoutviz.plot_turbine_rotors(fmodel, ax=ax)
ax.set_title("Turbines aligned")

ax = axarr[1] # Optimized yaw angles
optimal_yaw_angles = (
    df_opt[(df_opt["wind_direction"] == 270.0) & (df_opt["wind_speed"] == 8.0)]
    .yaw_angles_opt.values[0]
).reshape(1,-1)
fmodel.set(yaw_angles=optimal_yaw_angles)
fmodel.run()
horizontal_plane = fmodel.calculate_horizontal_plane(height=90.0)
flowviz.visualize_cut_plane(horizontal_plane, ax=ax)
layoutviz.plot_turbine_rotors(fmodel, ax=ax, yaw_angles=optimal_yaw_angles)
ax.set_title("Optimized yaw angles")

plt.show()
