"""Example: Optimizing yaw angles with disabled turbines

This example demonstrates how to optimize yaw angles in FLORIS, when some turbines are disabled.
The example optimization is run using both YawOptimizerSR and YawOptimizerGeometric, the two
yaw optimizers that support disabling turbines.
"""

import numpy as np

from floris import FlorisModel
from floris.optimization.yaw_optimization.yaw_optimizer_geometric import YawOptimizationGeometric
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


# Load a 3-turbine model
fmodel = FlorisModel("../inputs/gch.yaml")

# Set wind conditions to be the same for two cases
fmodel.set(wind_directions=[270.]*2, wind_speeds=[8.]*2, turbulence_intensities=[.06]*2)

# First run the case where all turbines are active and print results
yaw_opt = YawOptimizationSR(fmodel)
df_opt = yaw_opt.optimize()
print("Serial Refine optimized yaw angles (all turbines active) [deg]:\n", df_opt.yaw_angles_opt)

yaw_opt = YawOptimizationGeometric(fmodel)
df_opt = yaw_opt.optimize()
print("\nGeometric optimized yaw angles (all turbines active) [deg]:\n", df_opt.yaw_angles_opt)

# Disable turbines (different pattern for each of the two cases)
# First case: disable the middle turbine
# Second case: disable the front turbine
fmodel.set_operation_model('mixed')
fmodel.set(disable_turbines=np.array([[False, True, False], [True, False, False]]))

# Rerun optimizations and print results
yaw_opt = YawOptimizationSR(fmodel)
df_opt = yaw_opt.optimize()
print(
    "\nSerial Refine optimized yaw angles (some turbines disabled) [deg]:\n",
    df_opt.yaw_angles_opt
)
# Note that disabled turbines are assigned a zero yaw angle, but their yaw angle is arbitrary as it
# does not affect the total power output.

yaw_opt = YawOptimizationGeometric(fmodel)
df_opt = yaw_opt.optimize()
print("\nGeometric optimized yaw angles (some turbines disabled) [deg]:\n", df_opt.yaw_angles_opt)
