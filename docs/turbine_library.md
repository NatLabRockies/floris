(turbine_library)=
# Turbine Library

FLORIS includes a library of predefined wind turbine models that can be used to quickly set up
simulations without needing to define the turbine characteristics manually. These include standard
reference wind turbines as well as fictional wind turbine models for the purpose of demonstrating
various features of FLORIS. These turbines are stored in the `floris.turbine_library` module.

## NREL 5MW reference wind turbine

FLORIS representation of the NREL 5MW reference wind turbine {cite:t}`jonkman_NREL5MW_2009`. Data
based on https://github.com/NREL/turbine-models/blob/master/Offshore/NREL_5MW_126_RWT_corrected.csv.
Specified as `"nrel_5MW"` in the `turbine_type` field of the FLORIS input dictionary.

The NREL 5MW turbine is the default turbine model used in most FLORIS examples and tutorials. It is
also the model used if FLORIS is instantiated in the defaults configuration using
`FlorisModel("defaults")`.


## IEA 15MW reference wind turbine

FLORIS representation of the IEA 15MW reference wind turbine {cite:t}`gaertner_IEA15MW_2020`. Data
based on https://github.com/IEAWindTask37/IEA-15-240-RWT/blob/master/Documentation/IEA-15-240-RWT_tabular.xlsx.
Specified as `"iea_15MW"` in the `turbine_type` field of the FLORIS input dictionary.

The IEA 15MW turbine is used in the following examples:
- examples/examples_control_types/004_helix_active_wake_mixing.py

## IEA 10MW reference wind turbine

FLORIS representation of the IEA 10MW reference wind turbine {cite:t}`kainz_IEA10MW_2024`. Data
based on https://github.com/NREL/turbine-models/blob/master/Offshore/IEA_10MW_198_RWT.csv.
Specified as `"iea_10MW"` in the `turbine_type` field of the FLORIS input dictionary.

The IEA 10MW turbine is used in the following examples:
- examples/examples_turbine/002_multiple_turbine_types.py

## IEA 15MW multidimensional

Fictional IEA 15MW turbine model used to demonstrate the use of multidimensional power and thrust
coefficient data. Reads in fictional multidimensional data describing the power and thrust coefficient
relationships on wave period `Tp` and wave height `Hs` from `iea_15MW_multi_dim_Tp_Hs.csv` in the
`turbine_library` folder. Specified as `"iea_15MW_multi_dim"` in the `turbine_type` field of the FLORIS
input dictionary. This data should be treated as fictional and for demonstrative purposes only.

This fictional turbine model is not currently used in examples.

## IEA 15MW floating, multidimensional

The same as the multidimensional IEA 15MW turbine model above, but with an additional fictional
floating platform tilt table. This model is used to demonstrate the floating wind turbine capabilities
in FLORIS. Specified as `"iea_15MW_floating_multi_dim"` in the `turbine_type` field of the FLORIS input
dictionary. This data should be treated as fictional and for demonstrative purposes only.

This fictional turbine model is used in the following examples:
- examples/examples_multidim/001_multi_dimensional_cp_ct.py
- examples/examples_multidim/002_multi_dimensional_cp_ct_2Hs.py
