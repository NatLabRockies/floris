"""Example 9: Parallel Models

This example demonstrates how to use the ParFlorisModel class to parallelize the
calculation of the FLORIS model. ParFlorisModel inherits from the FlorisModel
and so can be used in the same way with a consistent interface. ParFlorisModel
replaces the ParallelFlorisModel, which will be deprecated in a future release.

"""

import numpy as np

from floris import (
    FlorisModel,
    ParFlorisModel,
    TimeSeries,
    UncertainFlorisModel,
)


# When using parallel optimization it is important the "root" script include this
# if __name__ == "__main__": block to avoid problems
if __name__ == "__main__":
    # Instantiate the FlorisModel
    fmodel = FlorisModel("inputs/gch.yaml")

    # The ParFlorisModel can be instantiated either from a FlorisModel or from
    # the input file.
    pfmodel_1 = ParFlorisModel("inputs/gch.yaml")  # Via input file
    pfmodel_2 = ParFlorisModel(fmodel)  # Via FlorisModel

    # The ParFlorisModel has additional inputs which define the parallelization
    # but don't affect the output.
    pfmodel_3 = ParFlorisModel(
        fmodel,
        interface="multiprocessing",  # Default
        max_workers=2,  # Defaults to num_cpu
        n_wind_condition_splits=2,  # Defaults to max_workers
    )

    # Define a simple inflow
    time_series = TimeSeries(
        wind_speeds=np.arange(1, 25, 0.5), wind_directions=270.0, turbulence_intensities=0.06
    )

    # Demonstrate that interface and results are the same
    fmodel.set(wind_data=time_series)
    pfmodel_1.set(wind_data=time_series)
    pfmodel_2.set(wind_data=time_series)
    pfmodel_3.set(wind_data=time_series)

    fmodel.run()
    pfmodel_1.run()
    pfmodel_2.run()
    pfmodel_3.run()

    # Compare the results
    powers_fmodel = fmodel.get_turbine_powers()
    powers_pfmodel_1 = pfmodel_1.get_turbine_powers()
    powers_pfmodel_2 = pfmodel_2.get_turbine_powers()
    powers_pfmodel_3 = pfmodel_3.get_turbine_powers()

    print(
        f"Testing if outputs of fmodel and pfmodel_1 are "
        f"close: {np.allclose(powers_fmodel, powers_pfmodel_1)}"
    )
    print(
        f"Testing if outputs of fmodel and pfmodel_2 are "
        f"close: {np.allclose(powers_fmodel, powers_pfmodel_2)}"
    )
    print(
        f"Testing if outputs of fmodel and pfmodel_3 are "
        f"close: {np.allclose(powers_fmodel, powers_pfmodel_3)}"
    )

    # Because ParFlorisModel is a subclass of FlorisModel, it can also be used as
    # an input to the UncertainFlorisModel class. This allows for parallelization of
    # the uncertainty calculations.
    ufmodel = UncertainFlorisModel(fmodel)
    pufmodel = UncertainFlorisModel(pfmodel_1)

    # Demonstrate matched results
    ufmodel.set(wind_data=time_series)
    pufmodel.set(wind_data=time_series)

    ufmodel.run()
    pufmodel.run()

    powers_ufmodel = ufmodel.get_turbine_powers()
    powers_pufmodel = pufmodel.get_turbine_powers()

    print("--------------------")
    print(
        f"Testing if outputs of ufmodel and pufmodel are "
        f"close: {np.allclose(powers_ufmodel, powers_pufmodel)}"
    )
