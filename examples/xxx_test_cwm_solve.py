import numpy as np

from floris import FlorisModel


def main():
    # The FlorisModel class is the entry point for most usage.
    # Initialize using an input yaml file
    fmodel = FlorisModel("inputs/cwm.yaml")

    # Changing the wind farm layout uses FLORIS' set method to a two-turbine layout
    #fmodel.set(layout_x=[100, 500.0], layout_y=[300.0, 100.0])

#    layout_x = [100, 100, 500, 600, 800, 1200]
#    layout_y = [0, 100, 500, 0, 300, 400]

    # Example
    D = 126.  # Turbine diameter [m]

    # Test a random layout generator
    N= 20  # Number of turbines
    xd = 5 * D  # Minimum distance in x-direction
    yd = 3 * D  # Minimum distance in y-direction
    xlim = (0, np.sqrt(N*2.5) * xd)  # X limits
    ylim = (0, np.sqrt(N*2.5) * yd)  # Y limits
    layout_x, layout_y = random_layout(N=N, xd=5*D, yd=3*D, xlim=xlim, ylim=ylim)
    print(layout_x, layout_y)

    yaw_angles = np.random.default_rng(0).uniform(-20, 20, size=(1, N))
    #yaw_angles = -20 * np.ones((1, N))
    print("Yaw angles:", yaw_angles)

    fmodel.set(layout_x=layout_x, layout_y=layout_y,
               yaw_angles=yaw_angles,
               )


    # Single wind condition
    fmodel.set(
        wind_directions=np.array([270.0]),
        wind_speeds=np.array([8.0]),
        turbulence_intensities=np.array([0.06])
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



def random_layout(N, xd, yd, xlim=(0, 100), ylim=(0, 100), max_attempts=100000):
    layout_x = []
    layout_y = []
    attempts = 0

    while len(layout_x) < N and attempts < max_attempts:
        x_new = np.random.uniform(*xlim)
        y_new = np.random.uniform(*ylim)

        # Check distances with existing points
        if layout_x:
            dx = np.abs(np.array(layout_x) - x_new)
            dy = np.abs(np.array(layout_y) - y_new)

            # Valid if no other point is too close in BOTH directions
            if np.any((dx < xd) & (dy < yd)):
                attempts += 1
                continue

        layout_x.append(x_new)
        layout_y.append(y_new)
        attempts = 0  # reset since we succeeded

    if len(layout_x) < N:
        raise RuntimeError(f"Could not place {N} points after {max_attempts} attempts.")

    return np.array(layout_x), np.array(layout_y)


if __name__ == "__main__":
    main()
