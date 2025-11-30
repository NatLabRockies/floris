
import copy
import logging

import numpy as np
import pytest

from floris import (
    FlorisModel,
    ParallelFlorisModel,
    UncertainFlorisModel,
)
from tests.conftest import (
    assert_results_arrays,
)


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

def test_raise_deprecation_warning(sample_inputs_fixture, caplog):
    """
    Test that a warning is raised when instantiating the ParallelFlorisModel.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ParallelFlorisModel(
            fmodel=fmodel,
            max_workers=2,
            n_wind_condition_splits=2,
            interface="concurrent",
            print_timings=False,
        )
    assert caplog.text != "" # Checking not empty
    caplog.clear()

def test_parallel_turbine_powers(sample_inputs_fixture):
    """
    The parallel computing interface behaves like the floris interface, but distributes
    calculations among available cores to speed up the necessary computations. This test compares
    the individual turbine powers computed with the parallel interface to those computed with
    the serial floris interface. The expected result is that the turbine powers should be
    exactly the same.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel_input = copy.deepcopy(fmodel)
    fmodel.run()

    serial_turbine_powers = fmodel.get_turbine_powers()

    pfmodel = ParallelFlorisModel(
        fmodel=pfmodel_input,
        max_workers=2,
        n_wind_condition_splits=2,
        interface="concurrent",
        print_timings=False,
    )

    parallel_turbine_powers = pfmodel.get_turbine_powers()

    if DEBUG:
        print(serial_turbine_powers)
        print(parallel_turbine_powers)

    assert_results_arrays(parallel_turbine_powers, serial_turbine_powers)

def test_parallel_get_AEP(sample_inputs_fixture):

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    freq=np.linspace(0, 1, 16)/8

    fmodel = FlorisModel(sample_inputs_fixture.core)
    pfmodel_input = copy.deepcopy(fmodel)

    fmodel.run()
    serial_farm_AEP = fmodel.get_farm_AEP(freq=freq)

    pfmodel = ParallelFlorisModel(
        fmodel=pfmodel_input,
        max_workers=2,
        n_wind_condition_splits=2,
        interface="concurrent",
        print_timings=False,
    )

    parallel_farm_AEP = pfmodel.get_farm_AEP(freq=freq)

    assert np.allclose(parallel_farm_AEP, serial_farm_AEP)

def test_parallel_uncertain_error(sample_inputs_fixture):
    """

    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    ufmodel = UncertainFlorisModel(
        sample_inputs_fixture.core,
        wd_sample_points=[-3, 0, 3],
        wd_std=3
    )

    with pytest.raises(ValueError):
        ParallelFlorisModel(
            fmodel=ufmodel,
            max_workers=2,
            n_wind_condition_splits=2,
            interface="multiprocessing",
            print_timings=False,
        )
