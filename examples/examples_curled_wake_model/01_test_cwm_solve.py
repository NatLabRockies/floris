import numpy as np

from floris import FlorisModel


fmodel = FlorisModel("inputs/cwm.yaml")

# Changing the wind farm layout uses FLORIS' set method to a two-turbine layout
fmodel.set(layout_x=[100, 500.0, 1000.0], layout_y=[0.0, 0.0, 0.0])


single_condition = False

if single_condition:
    fmodel.set(
        wind_directions=np.array([270.0]),
        wind_speeds=np.array([8.0]),
        turbulence_intensities=np.array([0.06])
    )
else:
    fmodel.set(
        wind_directions=np.array([270.0, 180.0]),
        wind_speeds=np.array([9.0, 8.0]),
        turbulence_intensities=np.array([0.06, 0.06])
    )

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
