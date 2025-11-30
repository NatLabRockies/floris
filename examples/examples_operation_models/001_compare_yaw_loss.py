"""Example: Compare yaw loss under different operation models

This example shows demonstrates how the Controller-dependent operation model (developed at TUM)
and Unified Momentum Model (developed at MIT) alter how a turbine loses power to yaw compared
to the standard cosine loss model.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel


# Parameters
N  = 101 # How many steps to cover yaw range in
yaw_max = 30 # Maximum yaw to test

# Set up the yaw angle sweep
yaw_angles = np.zeros((N,1))
yaw_angles[:,0] = np.linspace(-yaw_max, yaw_max, N)

# Create the FLORIS model
fmodel = FlorisModel("../inputs/gch.yaml")
fmodel.set(
    layout_x=[0.0],
    layout_y=[0.0],
    wind_directions=np.ones(N) * 270.0,
    wind_speeds=1.0*np.ones(N), # Will be replaced
    turbulence_intensities=0.06 * np.ones(N),
    yaw_angles=yaw_angles,
)

# Define a function to evaluate the power under various yaw angles
def evaluate_yawed_power(wsp: float, op_model: str) -> float:
    print(f"Evaluating model: {op_model}   wind speed: {wsp} m/s")
    fmodel.set(wind_speeds=wsp * np.ones(N))
    fmodel.set_operation_model(op_model)
    fmodel.run()

    return fmodel.get_turbine_powers()[:, 0]

# Loop over the operational models and wind speeds to compare
op_models = ["simple", "cosine-loss", "controller-dependent", "unified-momentum"]
wind_speeds = [8.0, 11.5, 15.0]
results = {}
for op_model, wsp in itertools.product(op_models, wind_speeds):
    results[(op_model, wsp)] = evaluate_yawed_power(wsp, op_model)

# Plot the results
fig, axes = plt.subplots(1, len(wind_speeds), sharey=True, figsize=(10, 5))

colors = ["k", "k", "C0", "C1"]
linestyles = ["dashed", "solid", "dashed", "dotted"]
for wsp, ax in zip(wind_speeds, axes):
    ax.set_title(f"Wind speed: {wsp} m/s")
    ax.set_xlabel("Yaw angle [deg]")
    ax.grid(True)
    for op_model, c, ls in zip(op_models, colors, linestyles):

        upstream_yaw_angle = yaw_angles[:, 0]
        central_power = results[(op_model, wsp)][upstream_yaw_angle == 0]
        ax.plot(
            upstream_yaw_angle,
            results[(op_model, wsp)] / central_power,
            label=op_model,
            color=c,
            linestyle=ls,
        )

ax.grid(True)
ax.legend()
axes[0].set_xlabel("Yaw angle [deg]")
axes[0].set_ylabel("Normalized turbine power [-]")

plt.show()
