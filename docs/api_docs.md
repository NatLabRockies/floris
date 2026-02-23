# API Documentation

FLORIS is primarily divided into the {py:mod}`floris` package, which contains the user-level API,
and {py:mod}`floris.core` is the core code that models the wind turbines and wind farms.
Additionally, the {py:mod}`turbine_library` package contains turbine models that ship with FLORIS;
and the {py:mod}`optimization` package contains high-level optimization routines that accept and
work on instantiated `FlorisModel` objects.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   floris.flow_visualization
   floris.floris_model
   floris.wind_data
   floris.uncertain_floris_model
   floris.turbine_library
   floris.parallel_floris_model
   floris.optimization
   floris.layout_visualization
   floris.cut_plane
   floris.core
   floris.convert_turbine_v3_to_v4
   floris.convert_floris_input_v3_to_v4
   floris.utilities
   floris.type_dec
   floris.logging_manager
```
