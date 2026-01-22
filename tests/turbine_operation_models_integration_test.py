import numpy as np
import pytest
from attrs import define, field

from floris import FlorisModel
from floris.core.turbine import BaseOperationModel


# Establish a static class
@define
class UserDefinedStatic(BaseOperationModel):
    def power(velocities, **_):
        return 1000*np.ones(velocities.shape[:2])
    def thrust_coefficient(velocities, **_):
        return 0.8*np.ones(velocities.shape[:2])
    def axial_induction(velocities, **_):
        return 1/3*np.ones(velocities.shape[:2])

# Establish a dynamic class
@define
class UserDefinedDynamic(BaseOperationModel):
    _flat_power = field(init=True, default=500.0)
    _flat_thrust_coefficient = field(init=True, default=0.7)
    _flat_axial_induction = field(init=True, default=0.3)
    def power(self, velocities, **_):
        return self._flat_power*np.ones(velocities.shape[:2])
    def thrust_coefficient(self, velocities, **_):
        return self._flat_thrust_coefficient*np.ones(velocities.shape[:2])
    def axial_induction(self, velocities, **_):
        return self._flat_axial_induction*np.ones(velocities.shape[:2])


def test_static_user_defined_op_model():
    fmodel = FlorisModel("defaults")
    fmodel.set(
        layout_x=[0.0, 500.0, 1000.0],
        layout_y=[0.0, 0.0, 0.0],
        wind_speeds=[8.0, 9.0],
        wind_directions=[270.0, 280.0],
        turbulence_intensities=[0.06, 0.06]
    )
    fmodel.set_operation_model(UserDefinedStatic)
    fmodel.run()
    power = fmodel.get_turbine_powers()
    thrust_coefficients = fmodel.get_turbine_thrust_coefficients()
    axial_inductions = fmodel.get_turbine_axial_induction_factors()

    assert np.all(power.shape == (2, 3))
    assert np.all(thrust_coefficients.shape == (2, 3))
    assert np.all(axial_inductions.shape == (2, 3))

    assert np.allclose(power, 1000.0)
    assert np.allclose(thrust_coefficients, 0.8)
    assert np.allclose(axial_inductions, 1/3)

def test_dynamic_user_defined_op_model():

    fmodel = FlorisModel("defaults")
    fmodel.set(
        layout_x=[0.0, 500.0, 1000.0],
        layout_y=[0.0, 0.0, 0.0],
        wind_speeds=[8.0, 9.0],
        wind_directions=[270.0, 280.0],
        turbulence_intensities=[0.06, 0.06]
    )
    # Try without instantiating (TODO: create more helpful error?)
    with pytest.raises(TypeError):
        fmodel.set_operation_model(UserDefinedDynamic)
        fmodel.run()
    # Now instantiate and try again
    instantiated_operation_model = UserDefinedDynamic()
    fmodel.set_operation_model(instantiated_operation_model)
    fmodel.run()
    power = fmodel.get_turbine_powers()
    thrust_coefficients = fmodel.get_turbine_thrust_coefficients()
    axial_inductions = fmodel.get_turbine_axial_induction_factors()

    assert np.all(power.shape == (2, 3))
    assert np.all(thrust_coefficients.shape == (2, 3))
    assert np.all(axial_inductions.shape == (2, 3))

    assert np.allclose(power, 500.0)
    assert np.allclose(thrust_coefficients, 0.7)
    assert np.allclose(axial_inductions, 0.3)

def test_set_run_ordering():
    fmodel = FlorisModel("defaults")
    fmodel.set_operation_model(UserDefinedStatic)
    fmodel.set(
        layout_x=[0.0, 500.0, 1000.0],
        layout_y=[0.0, 0.0, 0.0],
        wind_speeds=[8.0, 9.0],
        wind_directions=[270.0, 280.0],
        turbulence_intensities=[0.06, 0.06]
    )
    fmodel.run()

    # Reset, rerun
    fmodel.set(
        wind_directions=[300.0, 310.0],
    )
    fmodel.run()

    # Now, try a dynamic model
    fmodel.set_operation_model(UserDefinedDynamic(flat_power=850.0))
    fmodel.run()
    fmodel.set(
        wind_directions=[240.0, 250.0],
    )
    fmodel.run()
