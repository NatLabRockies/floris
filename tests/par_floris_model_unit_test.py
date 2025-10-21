
import copy
import logging

import numpy as np
import pytest

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)
from floris.par_floris_model import ParFlorisModel


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

def test_None_interface(sample_inputs_fixture):
    """
    With interface=None, the ParFlorisModel should behave exactly like the FlorisModel.
    (ParFlorisModel.run() simply calls the parent FlorisModel.run()).
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParFlorisModel(
        sample_inputs_fixture.core,
        interface=None,
        n_wind_condition_splits=2 # Not used when interface=None
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_multiprocessing_interface(sample_inputs_fixture):
    """
    With interface="multiprocessing", the ParFlorisModel should return the same powers
    as the FlorisModel.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParFlorisModel(
        sample_inputs_fixture.core,
        interface="multiprocessing",
        n_wind_condition_splits=2
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_pathos_interface(sample_inputs_fixture):
    """
    With interface="pathos", the ParFlorisModel should return the same powers
    as the FlorisModel.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParFlorisModel(
        sample_inputs_fixture.core,
        interface="pathos",
        n_wind_condition_splits=2
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

    # Run in powers_only mode
    pfmodel = ParFlorisModel(
        sample_inputs_fixture.core,
        interface="pathos",
        n_wind_condition_splits=2,
        return_turbine_powers_only=True
    )

    pfmodel.run()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_concurrent_interface(sample_inputs_fixture):
    """
    With interface="concurrent", the ParFlorisModel should return the same powers
    as the FlorisModel.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParFlorisModel(
        sample_inputs_fixture.core,
        interface="concurrent",
        n_wind_condition_splits=2,
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

    # Run in powers_only mode
    pfmodel = ParFlorisModel(
        sample_inputs_fixture.core,
        interface="concurrent",
        n_wind_condition_splits=2,
        return_turbine_powers_only=True
    )

    pfmodel.run()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_return_turbine_powers_only(sample_inputs_fixture):
    """
    With return_turbine_powers_only=True, the ParFlorisModel should return only the
    turbine powers, not the full results.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParFlorisModel(
        sample_inputs_fixture.core,
        interface="multiprocessing",
        n_wind_condition_splits=2,
        return_turbine_powers_only=True
    )

    fmodel.run()
    pfmodel.run()

    f_turb_powers = fmodel.get_turbine_powers()
    pf_turb_powers = pfmodel.get_turbine_powers()

    assert np.allclose(f_turb_powers, pf_turb_powers)

def test_run_error(sample_inputs_fixture, caplog):
    """
    Check that an error is raised if an output is requested before calling run().
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    pfmodel = ParFlorisModel(
        sample_inputs_fixture.core,
        interface="multiprocessing",
        n_wind_condition_splits=2
    )

    # In future versions, error will be raised
    # with pytest.raises(RuntimeError):
    #     pfmodel.get_turbine_powers()
    # with pytest.raises(RuntimeError):
    #     pfmodel.get_farm_AEP()

    # For now, only a warning is raised for backwards compatibility
    with caplog.at_level(logging.WARNING):
        pfmodel.get_turbine_powers()
    assert caplog.text != "" # Checking not empty
    caplog.clear()

def test_configuration_compatibility(sample_inputs_fixture, caplog):
    """
    Check that the ParFlorisModel is compatible with FlorisModel and
    UncertainFlorisModel configurations.
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)

    # Allowed to instantiate ParFlorisModel using fmodel
    with caplog.at_level(logging.WARNING):
        ParFlorisModel(fmodel)
    assert caplog.text == "" # Checking empty
    caplog.clear()

    pfmodel = ParFlorisModel(sample_inputs_fixture.core)
    with caplog.at_level(logging.WARNING):
        pfmodel.fmodel
    assert caplog.text != "" # Checking not empty
    caplog.clear()

    with pytest.raises(AttributeError):
        pfmodel.fmodel.core

def test_wind_data_objects(sample_inputs_fixture):
    """
    Check that the ParFlorisModel is compatible with WindData objects.
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParFlorisModel(sample_inputs_fixture.core, max_workers=2)

    # Create a wind rose and set onto both models
    wind_speeds = np.array([8.0, 10.0, 12.0, 8.0, 10.0, 12.0])
    wind_directions = np.array([270.0, 270.0, 270.0, 280.0, 280.0, 280.0])
    wind_rose = WindRose(
        wind_directions=np.unique(wind_directions),
        wind_speeds=np.unique(wind_speeds),
        ti_table=0.06
    )
    fmodel.set(wind_data=wind_rose)
    pfmodel.set(wind_data=wind_rose)

    # Run; get turbine powers; compare results
    fmodel.run()
    powers_fmodel_wr = fmodel.get_turbine_powers()
    pfmodel.run()
    powers_pfmodel_wr = pfmodel.get_turbine_powers()

    assert powers_fmodel_wr.shape == powers_pfmodel_wr.shape
    assert np.allclose(powers_fmodel_wr, powers_pfmodel_wr)

    # Test a TimeSeries object
    wind_speeds = np.array([8.0, 8.0, 9.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    values = np.array([30.0, 20.0, 10.0])
    time_series = TimeSeries(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=0.06,
        values=values,
    )
    fmodel.set(wind_data=time_series)
    pfmodel.set(wind_data=time_series)

    fmodel.run()
    powers_fmodel_ts = fmodel.get_turbine_powers()
    pfmodel.run()
    powers_pfmodel_ts = pfmodel.get_turbine_powers()

    assert powers_fmodel_ts.shape == powers_pfmodel_ts.shape
    assert np.allclose(powers_fmodel_ts, powers_pfmodel_ts)

def test_control_setpoints(sample_inputs_fixture):
    """
    Check that the ParFlorisModel is compatible with yaw angles.
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel = ParFlorisModel(sample_inputs_fixture.core, max_workers=2)

    # Set yaw angles
    yaw_angles = np.tile(np.array([[10.0, 20.0, 30.0]]), (fmodel.n_findex,1))
    fmodel.set(yaw_angles=yaw_angles)
    pfmodel.set(yaw_angles=yaw_angles)

    # Run; get turbine powers; compare results
    fmodel.run()
    powers_fmodel = fmodel.get_turbine_powers()
    pfmodel.run()
    powers_pfmodel = pfmodel.get_turbine_powers()

    assert powers_fmodel.shape == powers_pfmodel.shape
    assert np.allclose(powers_fmodel, powers_pfmodel)

def test_sample_flow_at_points(sample_inputs_fixture):

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)

    wind_speeds = np.array([8.0, 8.0, 9.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    fmodel.set(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=0.06 * np.ones_like(wind_speeds.flatten()),
    )

    x_test = np.array([500.0, 750.0, 1000.0, 1250.0, 1500.0])
    y_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    z_test = np.array([90.0, 90.0, 90.0, 90.0, 90.0])

    ws_base = fmodel.sample_flow_at_points(x_test, y_test, z_test)

    for interface in ["multiprocessing", "pathos", "concurrent"]:
        pfmodel = ParFlorisModel(fmodel, max_workers=2, interface=interface)
        ws_test = pfmodel.sample_flow_at_points(x_test, y_test, z_test)
        assert np.allclose(ws_base, ws_test)

def test_sample_ti_at_points(sample_inputs_fixture):

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["turbulence_model"] = "crespo_hernandez"

    fmodel = FlorisModel(sample_inputs_fixture.core)

    wind_speeds = np.array([8.0, 8.0, 9.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    fmodel.set(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=0.06 * np.ones_like(wind_speeds.flatten()),
    )

    x_test = np.array([500.0, 750.0, 1000.0, 1250.0, 1500.0])
    y_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    z_test = np.array([90.0, 90.0, 90.0, 90.0, 90.0])

    ti_base = fmodel.sample_ti_at_points(x_test, y_test, z_test)

    for interface in ["multiprocessing", "pathos", "concurrent"]:
        pfmodel = ParFlorisModel(fmodel, max_workers=2, interface=interface)
        ti_test = pfmodel.sample_ti_at_points(x_test, y_test, z_test)
        assert np.allclose(ti_base, ti_test)

def test_copy(sample_inputs_fixture):
    """
    Check that the ParFlorisModel copies correctly as a ParFlorisModel.
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    pfmodel = ParFlorisModel(sample_inputs_fixture.core, max_workers=2)
    pfmodel_copy = pfmodel.copy()
    assert isinstance(pfmodel_copy, ParFlorisModel)
    assert pfmodel_copy.max_workers == 2
