
# Floating Wind Turbine Modeling

The FLORIS wind turbine description includes a definition of the performance curves
(`power` and `thrust_coefficient`) as a function of wind speed, and this lookup table is used
directly in the calculation of power production for a steady-state atmospheric condition
(wind speed and wind direction). The power curve definition typically assumes a
fixed-bottom wind turbine with a fixed shaft tilt. However, floating
wind turbines have an additional rotational degrees of freedom in the platform pitch, which
adds a tilt angle to the rotor. As the turbine tilts, its performance is affected
because the turbine is no longer operating on its defined performance curve.

FLORIS allows the user to correct for the tilt angle of the turbine as a function of wind speed.
This is accomplished by including an additional input, `floating_tilt_table`, in the turbine definition that sets the steady tilt angle of the turbine based on wind speed. An interpolation is created and the tilt angle is computed for each turbine based on its rotor effective velocity. Taking into account the turbine rotor's built-in tilt, the absolute tilt is used to compute the power and thrust coefficient. To enable the use of the `floating_tilt_table`, the `correct_cp_ct_for_tilt` input on the turbine definition should be set to `True`.

The tilt angle is then used directly in the selected wake models to compute wake effects of tilted turbines.
