import numpy as np

from floris.core.turbine.unified_momentum_model import UnifiedMomentumModelTurbine
from tests.conftest import SampleInputs


def test_submodel_attributes():

    assert hasattr(UnifiedMomentumModelTurbine, "power")
    assert hasattr(UnifiedMomentumModelTurbine, "thrust_coefficient")
    assert hasattr(UnifiedMomentumModelTurbine, "axial_induction")

def test_UnifiedMomentumModelTurbine():

    n_turbines = 1
    wind_speed = 10.0
    turbine_data = SampleInputs().turbine

    yaw_angles_nom = 0 * np.ones((1, n_turbines))
    tilt_angles_nom = turbine_data["power_thrust_table"]["ref_tilt"] * np.ones((1, n_turbines))
    yaw_angles_test = 20 * np.ones((1, n_turbines))
    tilt_angles_test = 0 * np.ones((1, n_turbines))


    # Check that power works as expected
    test_power = UnifiedMomentumModelTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"], # Matches ref_air_density
        yaw_angles=yaw_angles_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    truth_index = turbine_data["power_thrust_table"]["wind_speed"].index(wind_speed)
    baseline_power = turbine_data["power_thrust_table"]["power"][truth_index] * 1000
    assert np.allclose(baseline_power, test_power)

    # Check that yaw and tilt angle have an effect
    test_power = UnifiedMomentumModelTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=turbine_data["power_thrust_table"]["ref_air_density"], # Matches ref_air_density
        yaw_angles=yaw_angles_test,
        tilt_angles=tilt_angles_test,
        tilt_interp=None
    )
    assert test_power < baseline_power

    # Check that a lower air density decreases power appropriately
    test_power = UnifiedMomentumModelTurbine.power(
        power_thrust_table=turbine_data["power_thrust_table"],
        velocities=wind_speed * np.ones((1, n_turbines, 3, 3)), # 1 findex, 1 turbine, 3x3 grid
        air_density=1.1,
        yaw_angles=yaw_angles_nom,
        tilt_angles=tilt_angles_nom,
        tilt_interp=None
    )
    assert test_power < baseline_power
