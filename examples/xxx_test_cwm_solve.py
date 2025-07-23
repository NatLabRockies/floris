import numpy as np

from floris import FlorisModel


# The FlorisModel class is the entry point for most usage.
# Initialize using an input yaml file
fmodel = FlorisModel("inputs/cwm.yaml")

# Changing the wind farm layout uses FLORIS' set method to a two-turbine layout
fmodel.set(layout_x=[0, 500.0], layout_y=[0.0, 0.0])

# Single wind condition
fmodel.set(
    wind_directions=np.array([270.0]), wind_speeds=[8.0], turbulence_intensities=np.array([0.06])
)

# After the set method, the run method is called to perform the simulation
fmodel.run()

# There are functions to get either the power of each turbine, or the farm power
turbine_powers = fmodel.get_turbine_powers() / 1000.0
farm_power = fmodel.get_farm_power() / 1000.0

print("Turbine powers")
print(turbine_powers)
print("Shape: ", turbine_powers.shape)

print("Farm power")
print(farm_power)
print("Shape: ", farm_power.shape)
