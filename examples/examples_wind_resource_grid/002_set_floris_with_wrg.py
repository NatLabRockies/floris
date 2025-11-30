"""Example: Setting FLORIS with WindRoseWRG

This example shows how to set a FLORIS model with a WindRoseWRG object.  When a WindRoseWRG object
is set as the wind data in a FLORIS model, the wind roses for each turbine in the layout are
generated and stored in the WindRoseWRG object.  The wind roses are then used to calculate the
expected turbine powers and farm power.
"""

import matplotlib.pyplot as plt
import numpy as np

from floris import (
    FlorisModel,
    WindRoseWRG,
)


# Read the WRG file into a WindRoseWRG object
wind_rose_wrg = WindRoseWRG("wrg_example.wrg")

# Print out some information about the
print(wind_rose_wrg)
print("=====================================")

# Since we didn't specify the wind speeds or fixed ti, the default values are used
# and displayed in the above printout.  Note the wrg file itself, does not specify
# turbulence intensity, or the wind speed bins FLORIS should use.  These can be chosen
# at initialization of the WindRoseWRG object, or set later.  We can update these values
# using the following methods
wind_rose_wrg.set_wind_speeds(np.arange(0, 26, 2.0))  # Use 2m/s steps
wind_rose_wrg.set_ti_table(0.07)  # Use a fixed ti of 7% for all wind directions and wind speeds


# Select a turbine layout
# The original grid in example 000 was defined by having the wind rose rotate with increasing y
# and move to higher speeds with increasing x.  Here we will define a turbine layout that
# has a line of turbine along the diagonals of the grid.
layout_x = np.array([0, 500, 1000])
layout_y = np.array([0, 1000, 2000])

# Set the turbine layout in the WindRoseWRG object, note that until this is done, the
# wind_rose_wrg object contains no WindRose objects.
# Note that if the wind_rose_wrg is applied to a FlorisModel, the layout_x and layout_y
# will be set in the FlorisModel and the wind roses will be generated for each turbine
wind_rose_wrg.set_layout(layout_x, layout_y)

# Plot the wind roses in the first figure
fig, axarr = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"polar": True})
wind_rose_wrg.plot_wind_roses(axarr=axarr, ws_step=5)

# Apply the wind_rose_wrg to a FlorisModel

# Load the FLORIS model and set that layout
fmodel = FlorisModel("../inputs/gch.yaml")
fmodel.set(layout_x=layout_x, layout_y=layout_y)

# Set the wind data as the wind_rose_wrg
fmodel.set(wind_data=wind_rose_wrg)

# Run the model and get the expected turbine powers and farm power
fmodel.run()
expected_turbine_powers = fmodel.get_expected_turbine_powers()
expected_farm_power = fmodel.get_expected_farm_power()

# Print the expected turbine powers, farm power, and the sum of the expected turbine powers
print("=====================================")
print("Expected turbine powers:", expected_turbine_powers)
print("Expected farm power:", expected_farm_power)
print("Sum of expected turbine powers:", expected_turbine_powers.sum())
print("=====================================")

# Now re-run using one of the turbine wind roses alone

# Compare with the result if just using the first wind rose or the last wind rose
fmodel.set(wind_data=wind_rose_wrg.wind_roses[0])
fmodel.run()
expected_turbine_powers_first = fmodel.get_expected_turbine_powers()

fmodel.set(wind_data=wind_rose_wrg.wind_roses[-1])
fmodel.run()
expected_turbine_powers_last = fmodel.get_expected_turbine_powers()

# Print the results to show match of first and last turbine when using their respective wind roses
print("Expected turbine powers:")
print("All wind roses:", expected_turbine_powers)
print(
    "First wind rose (1st power matches): ("
    + str(expected_turbine_powers_first[0])
    + "), "
    + str(expected_turbine_powers_first[1])
    + ", "
    + str(expected_turbine_powers_first[2])
    + ")"
)
print(
    "Last wind rose (Last power matches): "
    + str(expected_turbine_powers_last[0])
    + ", "
    + str(expected_turbine_powers_last[1])
    + ", ("
    + str(expected_turbine_powers_last[2])
    + ")"
)


plt.show()
