"""Example: Multi-dimensional power/thrust coefficient data
This example creates a FLORIS instance and:
1) Makes a two-turbine layout
2) Demonstrates single ws/wd simulations
3) Demonstrates multiple ws/wd simulations

with the modification of using a turbine definition that has a multi-dimensional
power/thrust coefficient table.

In the input file `gch_multi_dim_cp_ct.yaml`, the turbine_type points to a turbine definition,
iea_15MW_floating_multi_dim_cp_ct.yaml located in the turbine_library,
that supplies a multi-dimensional power/thrust coefficient data file in the form of a .csv file.
This .csv file contains two additional conditions to define power and thrust coefficient values for:
Tp for wave period, and Hs for wave height. For every combination of Tp and Hs defined, a
power/thrust coefficient/Wind speed table of values is also defined. It is required for this .csv
file to have the last 3 columns be ws, power, and thrust coefficient. In order
for this table to be used, the flag 'multi_dimensional_cp_ct' must be present and set to true in
the turbine definition. With this flag enabled, the solver will down-select to use the
interpolant defined at the closest conditions. The user must supply these conditions in the
main input file under the 'flow_field' section, e.g.:

NOTE: The multi-dimensional power/thrust coefficient data used in this example is fictional for the
purposes of facilitating this example. The power/thrust coefficient values for the different wave
conditions are scaled values of the original power/thrust coefficient data for the IEA 15MW turbine.

flow_field:
  multidim_conditions:
    Tp: 2.5
    Hs: 3.01

The solver will then use the nearest-neighbor interpolant. These conditions are currently global
and used to select the interpolant at each turbine.

Also note in the example below that there is a specific method for computing powers when
using turbines with multi-dimensional power/thrust coefficient data under FlorisModel, called
'get_turbine_powers_multidim'. The normal 'get_turbine_powers' method will not work.
"""

import numpy as np

from floris import FlorisModel


# Initialize FLORIS with the given input file.
fmodel = FlorisModel("../inputs/gch_multi_dim_cp_ct.yaml")

# Convert to a simple two turbine layout
fmodel.set(layout_x=[0.0, 500.0], layout_y=[0.0, 0.0])

# Single wind speed and wind direction
print("\n========================= Single Wind Direction and Wind Speed =========================")

# Get the turbine powers assuming 1 wind speed and 1 wind direction
fmodel.set(wind_directions=[270.0], wind_speeds=[8.0], turbulence_intensities=[0.06])

# Set the yaw angles to 0
yaw_angles = np.zeros([1, 2])  # 1 wind direction and wind speed, 2 turbines
fmodel.set(yaw_angles=yaw_angles)

# Calculate
fmodel.run()

# Get the turbine powers
turbine_powers = fmodel.get_turbine_powers() / 1000.0
print("The turbine power matrix should be of dimensions 1 findex X 2 Turbines")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

# Single wind speed and multiple wind directions
print("\n========================= Single Wind Direction and Multiple Wind Speeds ===============")

wind_speeds = np.array([8.0, 9.0, 10.0])
wind_directions = np.array([270.0, 270.0, 270.0])
turbulence_intensities = np.array([0.06, 0.06, 0.06])

yaw_angles = np.zeros([3, 2])  # 3 wind directions/ speeds, 2 turbines
fmodel.set(
    wind_speeds=wind_speeds,
    wind_directions=wind_directions,
    turbulence_intensities=turbulence_intensities,
    yaw_angles=yaw_angles,
)
fmodel.run()
turbine_powers = fmodel.get_turbine_powers() / 1000.0
print("The turbine power matrix should be of dimensions 3 findex X 2 Turbines")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

# Multiple wind speeds and multiple wind directions
print("\n========================= Multiple Wind Directions and Multiple Wind Speeds ============")

wind_speeds = np.tile([8.0, 9.0, 10.0], 3)
wind_directions = np.repeat([260.0, 270.0, 280.0], 3)
turbulence_intensities = 0.06 * np.ones_like(wind_speeds)

yaw_angles = np.zeros([9, 2])  # 9 wind directions/ speeds, 2 turbines
fmodel.set(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    turbulence_intensities=turbulence_intensities,
    yaw_angles=yaw_angles,
)
fmodel.run()
turbine_powers = fmodel.get_turbine_powers() / 1000.0
print("The turbine power matrix should be of dimensions 9 WD/WS X 2 Turbines")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)
