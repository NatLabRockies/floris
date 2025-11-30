"""Example 1: Opening FLORIS and Computing Power

This example illustrates several of the key concepts in FLORIS. It demonstrates:

  1) Initializing a FLORIS model
  2) Changing the wind farm layout
  3) Changing the incoming wind speed, wind direction and turbulence intensity
  4) Running the FLORIS simulation
  5) Getting the power output of the turbines

Main concept is introduce FLORIS and illustrate essential structure of most-used FLORIS calls
"""


import numpy as np

from floris import FlorisModel


# The FlorisModel class is the entry point for most usage.
# Initialize using an input yaml file
fmodel = FlorisModel("inputs/gch.yaml")

# Changing the wind farm layout uses FLORIS' set method to a two-turbine layout
fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])

# Changing wind speed, wind direction, and turbulence intensity uses the set method
# as well. Note that the wind_speeds, wind_directions, and turbulence_intensities
# are all specified as arrays of the same length.
fmodel.set(
    wind_directions=np.array([270.0]), wind_speeds=[8.0], turbulence_intensities=np.array([0.06])
)

# Note that typically all 3, wind_directions, wind_speeds and turbulence_intensities
# must be supplied to set.  However, the exception is if not changing the length
# of the arrays, then only one or two may be supplied.
fmodel.set(turbulence_intensities=np.array([0.07]))

# The number of elements in the wind_speeds, wind_directions, and turbulence_intensities
# corresponds to the number of conditions to be simulated.  In FLORIS, each of these are
# tracked by a simple index called a findex.  There is no requirement that the values
# be unique.  Internally in FLORIS, most data structures will have the findex as their
# 0th dimension.  The value n_findex is the total number of conditions to be simulated.
# This command would simulate 4 conditions (n_findex = 4).
fmodel.set(
    wind_directions=np.array([270.0, 270.0, 270.0, 270.0]),
    wind_speeds=[8.0, 8.0, 10.0, 10.0],
    turbulence_intensities=np.array([0.06, 0.06, 0.06, 0.06]),
)

# After the set method, the run method is called to perform the simulation
fmodel.run()

# There are functions to get either the power of each turbine, or the farm power
turbine_powers = fmodel.get_turbine_powers() / 1000.0
farm_power = fmodel.get_farm_power() / 1000.0

print("The turbine power matrix should be of dimensions 4 (n_findex) X 2 (n_turbines)")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

print("The farm power should be a 1D array of length 4 (n_findex)")
print(farm_power)
print("Shape: ", farm_power.shape)
