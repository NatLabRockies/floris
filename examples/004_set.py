"""Example 4: Set

This example illustrates the use of the set method.  The set method is used to
change the wind conditions, the wind farm layout, the turbine type,
and the controls settings.

This example demonstrates setting each of the following:
    1) Wind conditions
    2) Wind farm layout
    3) Controls settings

"""


import numpy as np

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)


fmodel = FlorisModel("inputs/gch.yaml")

######################################################
# Atmospheric Conditions
######################################################


# Change the wind directions, wind speeds, and turbulence intensities using numpy arrays
fmodel.set(
    wind_directions=np.array([270.0, 270.0, 270.0]),
    wind_speeds=[8.0, 9.0, 10.0],
    turbulence_intensities=np.array([0.06, 0.06, 0.06]),
)

# Set the wind conditions as above using the TimeSeries object
fmodel.set(
    wind_data=TimeSeries(
        wind_directions=270.0, wind_speeds=np.array([8.0, 9.0, 10.0]), turbulence_intensities=0.06
    )
)

# Set the wind conditions as above using the WindRose object
fmodel.set(
    wind_data=WindRose(
        wind_directions=np.array([270.0]),
        wind_speeds=np.array([8.0, 9.0, 10.0]),
        ti_table=0.06,
    )
)

# Set the wind shear
fmodel.set(wind_shear=0.2)


# Set the air density
fmodel.set(air_density=1.1)

# Set the reference wind height (which is the height at which the wind speed is given)
fmodel.set(reference_wind_height=92.0)


######################################################
# Array Settings
######################################################

# Changing the wind farm layout uses FLORIS' set method to a two-turbine layout
fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])

######################################################
# Controls Settings
######################################################

# Changes to controls settings can be made using the set method
# Note the dimension must match (n_findex, n_turbines) or (number of conditions, number of turbines)
# Above we n_findex = 3 and n_turbines = 2 so the matrix of yaw angles must be 3x2
yaw_angles = np.array([[0.0, 0.0], [25.0, 0.0], [0.0, 0.0]])
fmodel.set(yaw_angles=yaw_angles)

# By default for the turbines in the turbine_library, the power
# thrust model is set to "cosine-loss" which adjusts
# power and thrust according to cos^cosine_loss_exponent(yaw | tilt)
# where the default exponent is 1.88.  For other
# control capabilities, the power thrust model can be set to "mixed"
#  which provides the same cosine loss model, and
# additionally methods for specifying derating levels for power and disabling turbines.

# Use the reset operation method to clear out control signals
fmodel.reset_operation()

# Change to the mixed model turbine
fmodel.set_operation_model("mixed")

# Shut down the front turbine for the first two findex
disable_turbines = np.array([[True, False], [True, False], [False, False]])
fmodel.set(disable_turbines=disable_turbines)

# Derate the front turbine for the first two findex
RATED_POWER = 5e6  # 5MW (Anything above true rated power will still result in rated power)
power_setpoints = np.array(
    [[RATED_POWER * 0.3, RATED_POWER], [RATED_POWER * 0.3, RATED_POWER], [RATED_POWER, RATED_POWER]]
)
fmodel.set(power_setpoints=power_setpoints)
