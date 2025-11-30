"""Example: Extract wind speed at turbines

This example demonstrates how to extract the wind speed at the turbine points
from the FLORIS model.  Both the u velocities and the turbine average
velocities are grabbed from the model, then the turbine average is
recalculated from the u velocities to show that they are equivalent.
"""


import numpy as np

from floris import FlorisModel


# Initialize the FLORIS model
fmodel = FlorisModel("../inputs/gch.yaml")

# Create a 4-turbine layouts
fmodel.set(layout_x=[0, 0.0, 500.0, 500.0], layout_y=[0.0, 300.0, 0.0, 300.0])

# Calculate wake
fmodel.run()

# Collect the wind speed at all the turbine points
u_points = fmodel.core.flow_field.u

print("U points is 1 findex x 4 turbines x 3 x 3 points (turbine_grid_points=3)")
print(u_points.shape)

print("turbine_average_velocities is 1 findex x 4 turbines")
print(fmodel.turbine_average_velocities)

# Show that one is equivalent to the other following averaging
print(
    "turbine_average_velocities is determined by taking the cube root of mean "
    "of the cubed value across the points "
)
print(f"turbine_average_velocities: {fmodel.turbine_average_velocities}")
print(f"Recomputed:       {np.cbrt(np.mean(u_points**3, axis=(2,3)))}")
