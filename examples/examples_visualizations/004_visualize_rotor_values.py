"""Example: Visualize rotor velocities

Demonstrate visualizing the flow velocities at the rotor using plot_rotor_values

"""

import matplotlib.pyplot as plt

import floris.flow_visualization as flowviz
from floris import FlorisModel


fmodel = FlorisModel("../inputs/gch.yaml")

# Set a 2 turbine layout
fmodel.set(
    layout_x=[0, 500],
    layout_y=[0, 0],
    wind_directions=[270],
    wind_speeds=[8],
    turbulence_intensities=[0.06],
)

# Run the model
fmodel.run()

# Plot the values at each rotor
fig, axes, _, _ = flowviz.plot_rotor_values(
    fmodel.core.flow_field.u, findex=0, n_rows=1, n_cols=2, return_fig_objects=True
)
fig.suptitle("Rotor Plane Visualization, Original Resolution")

plt.show()
