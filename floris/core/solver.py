
from __future__ import annotations

import copy

import numpy as np

from floris.core import (
    axial_induction,
    Farm,
    FlowField,
    FlowFieldGrid,
    FlowFieldPlanarGrid,
    PointsGrid,
    thrust_coefficient,
    TurbineGrid,
)
from floris.core.rotor_velocity import average_velocity
from floris.core.wake import WakeModelManager
from floris.core.wake_deflection.empirical_gauss import yaw_added_wake_mixing
from floris.core.wake_deflection.gauss import (
    calculate_transverse_velocity,
    wake_added_yaw,
    yaw_added_turbulence_mixing,
)
from floris.core.wake_velocity.empirical_gauss import awc_added_wake_mixing
from floris.type_dec import NDArrayFloat
from floris.utilities import cosd


def calculate_area_overlap(wake_velocities, freestream_velocities, y_ngrid, z_ngrid):
    """
    compute wake overlap based on the number of points that are not freestream
    velocity, i.e. affected by the wake
    """
    # Count all of the rotor points with a negligible difference from freestream
    # count = np.sum(freestream_velocities - wake_velocities <= 0.05, axis=(3, 4))
    # return (y_ngrid * z_ngrid - count) / (y_ngrid * z_ngrid)
    # return 1 - count / (y_ngrid * z_ngrid)

    # Find the points on the rotor grids with a difference from freestream of greater
    # than some tolerance. These are all the points in the wake. The ratio of
    # these points to the total points is the portion of wake overlap.
    return np.sum(freestream_velocities - wake_velocities > 0.05, axis=(3, 4)) / (y_ngrid * z_ngrid)


# @profile
def sequential_solver(
    farm: Farm,
    flow_field: FlowField,
    grid: TurbineGrid,
    model_manager: WakeModelManager
) -> None:
    # Algorithm
    # For each turbine, calculate its effect on every downstream turbine.
    # For the current turbine, we are calculating the deficit that it adds to downstream turbines.
    # Integrate this into the main data structure.
    # Move on to the next turbine.

    # <<interface>>
    deflection_model_args = model_manager.deflection_model.prepare_function(grid, flow_field)
    deficit_model_args = model_manager.velocity_model.prepare_function(grid, flow_field)

    # This is u_wake
    wake_field = np.zeros_like(flow_field.u_initial_sorted)
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)

    # Expand input turbulence intensity to 4d for (n_turbines, grid, grid)
    turbine_turbulence_intensity = flow_field.turbulence_intensities[:, None, None, None]
    turbine_turbulence_intensity = np.repeat(turbine_turbulence_intensity, farm.n_turbines, axis=1)

    # Ambient turbulent intensity should be a copy of n_findex-long turbulence_intensity
    # with dimensions expanded for (n_turbines, grid, grid)
    ambient_turbulence_intensities = flow_field.turbulence_intensities.copy()
    ambient_turbulence_intensities = ambient_turbulence_intensities[:, None, None, None]

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        y_i = np.mean(grid.y_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        z_i = np.mean(grid.z_sorted[:, i:i+1], axis=(2, 3), keepdims=True)

        u_i = flow_field.u_sorted[:, i:i+1]
        v_i = flow_field.v_sorted[:, i:i+1]

        ct_i = thrust_coefficient(
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            thrust_coefficient_functions=farm.turbine_thrust_coefficient_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions
        )
        # Since we are filtering for the i'th turbine in the thrust coefficient function,
        # get the first index here (0:1)
        ct_i = ct_i[:, 0:1, None, None]
        axial_induction_i = axial_induction(
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            axial_induction_functions=farm.turbine_axial_induction_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions
        )
        # Since we are filtering for the i'th turbine in the axial induction function,
        # get the first index here (0:1)
        axial_induction_i = axial_induction_i[:, 0:1, None, None]
        turbulence_intensity_i = turbine_turbulence_intensity[:, i:i+1]
        yaw_angle_i = farm.yaw_angles_sorted[:, i:i+1, None, None]
        hub_height_i = farm.hub_heights_sorted[:, i:i+1, None, None]
        rotor_diameter_i = farm.rotor_diameters_sorted[:, i:i+1, None, None]
        TSR_i = farm.TSRs_sorted[:, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                flow_field.u_initial_sorted,
                grid.y_sorted[:, i:i+1] - y_i,
                grid.z_sorted[:, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                ct_i,
                TSR_i,
                axial_induction_i,
                flow_field.wind_shear,
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            ct_i,
            rotor_diameter_i,
            **deflection_model_args,
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial_sorted,
                flow_field.dudz_initial_sorted,
                grid.x_sorted - x_i,
                grid.y_sorted - y_i,
                grid.z_sorted,
                rotor_diameter_i,
                hub_height_i,
                yaw_angle_i,
                ct_i,
                TSR_i,
                axial_induction_i,
                flow_field.wind_shear,
            )

        if model_manager.enable_yaw_added_recovery:
            I_mixing = yaw_added_turbulence_mixing(
                u_i,
                turbulence_intensity_i,
                v_i,
                flow_field.w_sorted[:, i:i+1],
                v_wake[:, i:i+1],
                w_wake[:, i:i+1],
            )
            gch_gain = 2
            turbine_turbulence_intensity[:, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            axial_induction_i,
            deflection_field,
            yaw_angle_i,
            turbulence_intensity_i,
            ct_i,
            hub_height_i,
            rotor_diameter_i,
            **deficit_model_args,
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial_sorted
        )

        wake_added_turbulence_intensity = model_manager.turbulence_model.function(
            ambient_turbulence_intensities,
            grid.x_sorted,
            x_i,
            rotor_diameter_i,
            axial_induction_i,
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = (
            np.sum(velocity_deficit * flow_field.u_initial_sorted > 0.05, axis=(2, 3))
            / (grid.grid_resolution * grid.grid_resolution)
        )
        area_overlap = area_overlap[:, :, None, None]

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * (grid.x_sorted > x_i)
            * (np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
            * (grid.x_sorted <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum(
            np.sqrt(ti_added**2 + ambient_turbulence_intensities**2), turbine_turbulence_intensity
        )

        flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake

    flow_field.turbulence_intensity_field_sorted = turbine_turbulence_intensity
    flow_field.turbulence_intensity_field_sorted_avg = np.mean(
        turbine_turbulence_intensity,
        axis=(2,3),
        keepdims=True
    )


def full_flow_sequential_solver(
    farm: Farm,
    flow_field: FlowField,
    flow_field_grid: FlowFieldGrid | FlowFieldPlanarGrid | PointsGrid,
    model_manager: WakeModelManager
) -> None:

    # Get the flow quantities and turbine performance
    turbine_grid_farm = copy.deepcopy(farm)
    turbine_grid_flow_field = copy.deepcopy(flow_field)

    turbine_grid_farm.construct_turbine_map()
    turbine_grid_farm.construct_turbine_thrust_coefficient_functions()
    turbine_grid_farm.construct_turbine_axial_induction_functions()
    turbine_grid_farm.construct_turbine_power_functions()
    turbine_grid_farm.construct_hub_heights()
    turbine_grid_farm.construct_rotor_diameters()
    turbine_grid_farm.construct_turbine_TSRs()
    turbine_grid_farm.construct_turbine_ref_tilts()
    turbine_grid_farm.construct_turbine_tilt_interps()
    turbine_grid_farm.construct_turbine_correct_cp_ct_for_tilt()
    turbine_grid_farm.set_tilt_to_ref_tilt(flow_field.n_findex)

    turbine_grid = TurbineGrid(
        turbine_coordinates=turbine_grid_farm.coordinates,
        turbine_diameters=turbine_grid_farm.rotor_diameters,
        wind_directions=turbine_grid_flow_field.wind_directions,
        grid_resolution=3,
    )
    turbine_grid_farm.expand_farm_properties(
        turbine_grid_flow_field.n_findex,
        turbine_grid.sorted_coord_indices,
    )
    turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
    turbine_grid_farm.initialize(turbine_grid.sorted_indices)
    sequential_solver(turbine_grid_farm, turbine_grid_flow_field, turbine_grid, model_manager)

    ### Referring to the quantities from above, calculate the wake in the full grid

    # Use full flow_field here to use the full grid in the wake models
    deflection_model_args = model_manager.deflection_model.prepare_function(
        flow_field_grid,
        flow_field
    )
    deficit_model_args = model_manager.velocity_model.prepare_function(
        flow_field_grid,
        flow_field
    )

    wake_field = np.zeros_like(flow_field.u_initial_sorted)
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)

    # Initialize the turbulence intensity field over the entire flow field grid
    n_points = flow_field_grid.x_sorted.shape[1]
    ambient_turbulence_intensities = flow_field.turbulence_intensities[:, None, None, None]
    ambient_turbulence_intensities = np.repeat(ambient_turbulence_intensities, n_points, axis=1)
    turbulence_intensity_field = ambient_turbulence_intensities.copy()

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(flow_field_grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(turbine_grid.x_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        y_i = np.mean(turbine_grid.y_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        z_i = np.mean(turbine_grid.z_sorted[:, i:i+1], axis=(2, 3), keepdims=True)

        u_i = turbine_grid_flow_field.u_sorted[:, i:i+1]
        v_i = turbine_grid_flow_field.v_sorted[:, i:i+1]

        ct_i = thrust_coefficient(
            velocities=turbine_grid_flow_field.u_sorted,
            turbulence_intensities=turbine_grid_flow_field.turbulence_intensity_field_sorted,
            air_density=turbine_grid_flow_field.air_density,
            yaw_angles=turbine_grid_farm.yaw_angles_sorted,
            tilt_angles=turbine_grid_farm.tilt_angles_sorted,
            power_setpoints=turbine_grid_farm.power_setpoints_sorted,
            awc_modes=turbine_grid_farm.awc_modes_sorted,
            awc_amplitudes=turbine_grid_farm.awc_amplitudes_sorted,
            thrust_coefficient_functions=turbine_grid_farm.turbine_thrust_coefficient_functions,
            tilt_interps=turbine_grid_farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=turbine_grid_farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=turbine_grid_farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=turbine_grid.average_method,
            cubature_weights=turbine_grid.cubature_weights,
            multidim_condition=turbine_grid_flow_field.multidim_conditions,
        )
        # Since we are filtering for the i'th turbine in the thrust_coefficient function,
        # get the first index here (0:1)
        ct_i = ct_i[:, 0:1, None, None]
        axial_induction_i = axial_induction(
            velocities=turbine_grid_flow_field.u_sorted,
            turbulence_intensities=turbine_grid_flow_field.turbulence_intensity_field_sorted,
            air_density=turbine_grid_flow_field.air_density,
            yaw_angles=turbine_grid_farm.yaw_angles_sorted,
            tilt_angles=turbine_grid_farm.tilt_angles_sorted,
            power_setpoints=turbine_grid_farm.power_setpoints_sorted,
            awc_modes=turbine_grid_farm.awc_modes_sorted,
            awc_amplitudes=turbine_grid_farm.awc_amplitudes_sorted,
            axial_induction_functions=turbine_grid_farm.turbine_axial_induction_functions,
            tilt_interps=turbine_grid_farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=turbine_grid_farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=turbine_grid_farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=turbine_grid.average_method,
            cubature_weights=turbine_grid.cubature_weights,
            multidim_condition=turbine_grid_flow_field.multidim_conditions,
        )
        # Since we are filtering for the i'th turbine in the axial induction function,
        # get the first index here (0:1)
        axial_induction_i = axial_induction_i[:, 0:1, None, None]
        turbulence_intensity_i = \
            turbine_grid_flow_field.turbulence_intensity_field_sorted_avg[:, i:i+1]
        yaw_angle_i = turbine_grid_farm.yaw_angles_sorted[:, i:i+1, None, None]
        hub_height_i = turbine_grid_farm.hub_heights_sorted[:, i:i+1, None, None]
        rotor_diameter_i = turbine_grid_farm.rotor_diameters_sorted[:, i:i+1, None, None]
        TSR_i = turbine_grid_farm.TSRs_sorted[:, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                turbine_grid_flow_field.u_initial_sorted,
                turbine_grid.y_sorted[:, i:i+1] - y_i,
                turbine_grid.z_sorted[:, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                ct_i,
                TSR_i,
                axial_induction_i,
                flow_field.wind_shear,
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            ct_i,
            rotor_diameter_i,
            **deflection_model_args,
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial_sorted,
                flow_field.dudz_initial_sorted,
                flow_field_grid.x_sorted - x_i,
                flow_field_grid.y_sorted - y_i,
                flow_field_grid.z_sorted,
                rotor_diameter_i,
                hub_height_i,
                yaw_angle_i,
                ct_i,
                TSR_i,
                axial_induction_i,
                flow_field.wind_shear,
            )

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            axial_induction_i,
            deflection_field,
            yaw_angle_i,
            turbulence_intensity_i,
            ct_i,
            hub_height_i,
            rotor_diameter_i,
            **deficit_model_args,
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial_sorted
        )

        wake_added_ti = model_manager.turbulence_model.function(
            ambient_turbulence_intensities,
            flow_field_grid.x_sorted,
            x_i,
            rotor_diameter_i,
            axial_induction_i,
        )

        # Calculate locations where wake-added turbulence (WAT) applies
        area_overlap = np.where(velocity_deficit * flow_field.u_initial_sorted > 0.05, 1, 0)

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_ti, posinf=0.0)
            * (flow_field_grid.x_sorted > x_i)
            * (np.abs(y_i - flow_field_grid.y_sorted) < 2 * rotor_diameter_i)
            * (flow_field_grid.x_sorted <= downstream_influence_length + x_i)
        )
        # Combine turbine TIs with WAT
        turbulence_intensity_field = np.maximum(
            np.sqrt(ti_added**2 + ambient_turbulence_intensities**2), turbulence_intensity_field
        )

        flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake

    flow_field.turbulence_intensity_field_sorted = turbulence_intensity_field


def cc_solver(
    farm: Farm,
    flow_field: FlowField,
    grid: TurbineGrid,
    model_manager: WakeModelManager
) -> None:
    # <<interface>>
    deflection_model_args = model_manager.deflection_model.prepare_function(grid, flow_field)
    deficit_model_args = model_manager.velocity_model.prepare_function(grid, flow_field)

    # This is u_wake
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)
    turb_u_wake = np.zeros_like(flow_field.u_initial_sorted)
    turb_inflow_field = copy.deepcopy(flow_field.u_initial_sorted)

    # Set up turbulence arrays
    turbine_turbulence_intensity = flow_field.turbulence_intensities[:, None, None, None]
    turbine_turbulence_intensity = np.repeat(turbine_turbulence_intensity, farm.n_turbines, axis=1)

    # Ambient turbulent intensity should be a copy of n_findex-long turbulence_intensities
    # with extra dimension to reach 4d
    ambient_turbulence_intensities = flow_field.turbulence_intensities.copy()
    ambient_turbulence_intensities = ambient_turbulence_intensities[:, None, None, None]

    shape = (farm.n_turbines,) + np.shape(flow_field.u_initial_sorted)
    Ctmp = np.zeros((shape))
    # Ctmp = np.zeros((len(x_coord), len(wd), len(ws), len(x_coord), y_ngrid, z_ngrid))

    # sigma_i = np.zeros((shape))
    # sigma_i = np.zeros((len(x_coord), len(wd), len(ws), len(x_coord), y_ngrid, z_ngrid))

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        y_i = np.mean(grid.y_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        z_i = np.mean(grid.z_sorted[:, i:i+1], axis=(2, 3), keepdims=True)

        rotor_diameter_i = farm.rotor_diameters_sorted[:, i:i+1, None, None]

        mask2 = (
            (grid.x_sorted < x_i + 0.01)
            * (grid.x_sorted > x_i - 0.01)
            * (grid.y_sorted < y_i + 0.51 * rotor_diameter_i)
            * (grid.y_sorted > y_i - 0.51 * rotor_diameter_i)
        )
        turb_inflow_field = (
            turb_inflow_field * ~mask2
            + (flow_field.u_initial_sorted - turb_u_wake) * mask2
        )

        turb_avg_vels = average_velocity(turb_inflow_field)
        turb_Cts = thrust_coefficient(
            turb_avg_vels,
            flow_field.turbulence_intensity_field_sorted,
            flow_field.air_density,
            farm.yaw_angles_sorted,
            farm.tilt_angles_sorted,
            farm.power_setpoints_sorted,
            farm.awc_modes_sorted,
            farm.awc_amplitudes_sorted,
            farm.turbine_thrust_coefficient_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions,
        )
        turb_Cts = turb_Cts[:, :, None, None]
        turb_aIs = axial_induction(
            turb_avg_vels,
            flow_field.turbulence_intensity_field_sorted,
            flow_field.air_density,
            farm.yaw_angles_sorted,
            farm.tilt_angles_sorted,
            farm.power_setpoints_sorted,
            farm.awc_modes_sorted,
            farm.awc_amplitudes_sorted,
            farm.turbine_axial_induction_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions,
        )
        turb_aIs = turb_aIs[:, :, None, None]

        u_i = turb_inflow_field[:, i:i+1]
        v_i = flow_field.v_sorted[:, i:i+1]

        axial_induction_i = axial_induction(
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            axial_induction_functions=farm.turbine_axial_induction_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions,
        )

        axial_induction_i = axial_induction_i[:, :, None, None]

        turbulence_intensity_i = turbine_turbulence_intensity[:, i:i+1]
        yaw_angle_i = farm.yaw_angles_sorted[:, i:i+1, None, None]
        hub_height_i = farm.hub_heights_sorted[:, i:i+1, None, None]
        TSR_i = farm.TSRs_sorted[:, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                flow_field.u_initial_sorted,
                grid.y_sorted[:, i:i+1] - y_i,
                grid.z_sorted[:, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                turb_Cts[:, i:i+1],
                TSR_i,
                axial_induction_i,
                flow_field.wind_shear,
                scale=2.0,
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            turb_Cts[:, i:i+1],
            rotor_diameter_i,
            **deflection_model_args,
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial_sorted,
                flow_field.dudz_initial_sorted,
                grid.x_sorted - x_i,
                grid.y_sorted - y_i,
                grid.z_sorted,
                rotor_diameter_i,
                hub_height_i,
                yaw_angle_i,
                turb_Cts[:, i:i+1],
                TSR_i,
                axial_induction_i,
                flow_field.wind_shear,
                scale=2.0,
            )

        if model_manager.enable_yaw_added_recovery:
            I_mixing = yaw_added_turbulence_mixing(
                u_i,
                turbulence_intensity_i,
                v_i,
                flow_field.w_sorted[:, i:i+1],
                v_wake[:, i:i+1],
                w_wake[:, i:i+1],
            )
            gch_gain = 1.0
            turbine_turbulence_intensity[:, i:i+1] = turbulence_intensity_i + gch_gain * I_mixing

        turb_u_wake, Ctmp = model_manager.velocity_model.function(
            i,
            x_i,
            y_i,
            z_i,
            u_i,
            deflection_field,
            yaw_angle_i,
            turbine_turbulence_intensity,
            turb_Cts,
            farm.rotor_diameters_sorted[:, :, None, None],
            turb_u_wake,
            Ctmp,
            **deficit_model_args,
        )

        wake_added_turbulence_intensity = model_manager.turbulence_model.function(
            ambient_turbulence_intensities,
            grid.x_sorted,
            x_i,
            rotor_diameter_i,
            turb_aIs
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = 1 - (
            np.sum(turb_u_wake <= 0.05, axis=(2, 3), keepdims=True)
            / (grid.grid_resolution * grid.grid_resolution)
        )

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * (grid.x_sorted > x_i)
            * (np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
            * (grid.x_sorted <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum(
            np.sqrt(ti_added**2 + ambient_turbulence_intensities**2), turbine_turbulence_intensity
        )

        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake
    flow_field.u_sorted = turb_inflow_field

    flow_field.turbulence_intensity_field_sorted = turbine_turbulence_intensity
    flow_field.turbulence_intensity_field_sorted_avg = np.mean(
        turbine_turbulence_intensity,
        axis=(2,3),
        keepdims=True
    )


def full_flow_cc_solver(
    farm: Farm,
    flow_field: FlowField,
    flow_field_grid: FlowFieldGrid | FlowFieldPlanarGrid | PointsGrid,
    model_manager: WakeModelManager,
) -> None:
    # Get the flow quantities and turbine performance
    turbine_grid_farm = copy.deepcopy(farm)
    turbine_grid_flow_field = copy.deepcopy(flow_field)

    turbine_grid_farm.construct_turbine_map()
    turbine_grid_farm.construct_turbine_thrust_coefficient_functions()
    turbine_grid_farm.construct_turbine_axial_induction_functions()
    turbine_grid_farm.construct_turbine_power_functions()
    turbine_grid_farm.construct_hub_heights()
    turbine_grid_farm.construct_rotor_diameters()
    turbine_grid_farm.construct_turbine_TSRs()
    turbine_grid_farm.construct_turbine_ref_tilts()
    turbine_grid_farm.construct_turbine_tilt_interps()
    turbine_grid_farm.construct_turbine_correct_cp_ct_for_tilt()
    turbine_grid_farm.set_tilt_to_ref_tilt(flow_field.n_findex)

    turbine_grid = TurbineGrid(
        turbine_coordinates=turbine_grid_farm.coordinates,
        turbine_diameters=turbine_grid_farm.rotor_diameters,
        wind_directions=turbine_grid_flow_field.wind_directions,
        grid_resolution=3,
    )
    turbine_grid_farm.expand_farm_properties(
        turbine_grid_flow_field.n_findex,
        turbine_grid.sorted_coord_indices,
    )
    turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
    turbine_grid_farm.initialize(turbine_grid.sorted_indices)
    cc_solver(turbine_grid_farm, turbine_grid_flow_field, turbine_grid, model_manager)

    ### Referring to the quantities from above, calculate the wake in the full grid

    # Use full flow_field here to use the full grid in the wake models
    deflection_model_args = model_manager.deflection_model.prepare_function(
        flow_field_grid,
        flow_field
    )
    deficit_model_args = model_manager.velocity_model.prepare_function(
        flow_field_grid,
        flow_field
    )

    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)
    turb_u_wake = np.zeros_like(flow_field.u_initial_sorted)

    # Initialize the turbulence intensity field over the entire flow field grid
    n_points = flow_field_grid.x_sorted.shape[1]
    ambient_turbulence_intensities = flow_field.turbulence_intensities[:, None, None, None]
    ambient_turbulence_intensities = np.repeat(ambient_turbulence_intensities, n_points, axis=1)
    turbulence_intensity_field = ambient_turbulence_intensities.copy()

    shape = (farm.n_turbines,) + np.shape(flow_field.u_initial_sorted)
    Ctmp = np.zeros((shape))

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(flow_field_grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(turbine_grid.x_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        y_i = np.mean(turbine_grid.y_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        z_i = np.mean(turbine_grid.z_sorted[:, i:i+1], axis=(2, 3), keepdims=True)

        u_i = turbine_grid_flow_field.u_sorted[:, i:i+1]
        v_i = turbine_grid_flow_field.v_sorted[:, i:i+1]

        turb_avg_vels = average_velocity(turbine_grid_flow_field.u_sorted)
        turb_Cts = thrust_coefficient(
            velocities=turb_avg_vels,
            turbulence_intensities=turbine_grid_flow_field.turbulence_intensity_field_sorted,
            air_density=turbine_grid_flow_field.air_density,
            yaw_angles=turbine_grid_farm.yaw_angles_sorted,
            tilt_angles=turbine_grid_farm.tilt_angles_sorted,
            power_setpoints=turbine_grid_farm.power_setpoints_sorted,
            awc_modes=turbine_grid_farm.awc_modes_sorted,
            awc_amplitudes=turbine_grid_farm.awc_amplitudes_sorted,
            thrust_coefficient_functions=turbine_grid_farm.turbine_thrust_coefficient_functions,
            tilt_interps=turbine_grid_farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=turbine_grid_farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=turbine_grid_farm.turbine_power_thrust_tables,
            average_method=turbine_grid.average_method,
            cubature_weights=turbine_grid.cubature_weights,
            multidim_condition=turbine_grid_flow_field.multidim_conditions,
        )
        turb_Cts = turb_Cts[:, :, None, None]

        axial_induction_i = axial_induction(
            velocities=turbine_grid_flow_field.u_sorted,
            turbulence_intensities=turbine_grid_flow_field.turbulence_intensity_field_sorted,
            air_density=turbine_grid_flow_field.air_density,
            yaw_angles=turbine_grid_farm.yaw_angles_sorted,
            tilt_angles=turbine_grid_farm.tilt_angles_sorted,
            power_setpoints=turbine_grid_farm.power_setpoints_sorted,
            awc_modes=turbine_grid_farm.awc_modes_sorted,
            awc_amplitudes=turbine_grid_farm.awc_amplitudes_sorted,
            axial_induction_functions=turbine_grid_farm.turbine_axial_induction_functions,
            tilt_interps=turbine_grid_farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=turbine_grid_farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=turbine_grid_farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=turbine_grid.average_method,
            cubature_weights=turbine_grid.cubature_weights,
            multidim_condition=turbine_grid_flow_field.multidim_conditions,
        )
        axial_induction_i = axial_induction_i[:, :, None, None]

        turbulence_intensity_i = \
            turbine_grid_flow_field.turbulence_intensity_field_sorted_avg[:, i:i+1]
        yaw_angle_i = turbine_grid_farm.yaw_angles_sorted[:, i:i+1, None, None]
        hub_height_i = turbine_grid_farm.hub_heights_sorted[:, i:i+1, None, None]
        rotor_diameter_i = turbine_grid_farm.rotor_diameters_sorted[:, i:i+1, None, None]
        TSR_i = turbine_grid_farm.TSRs_sorted[:, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        if model_manager.enable_secondary_steering:
            added_yaw = wake_added_yaw(
                u_i,
                v_i,
                turbine_grid_flow_field.u_initial_sorted,
                turbine_grid.y_sorted[:, i:i+1] - y_i,
                turbine_grid.z_sorted[:, i:i+1],
                rotor_diameter_i,
                hub_height_i,
                turb_Cts[:, i:i+1],
                TSR_i,
                axial_induction_i,
                flow_field.wind_shear,
                scale=2.0,
            )
            effective_yaw_i += added_yaw

        # Model calculations
        # NOTE: exponential
        deflection_field = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            turbulence_intensity_i,
            turb_Cts[:, i:i+1],
            rotor_diameter_i,
            **deflection_model_args,
        )

        if model_manager.enable_transverse_velocities:
            v_wake, w_wake = calculate_transverse_velocity(
                u_i,
                flow_field.u_initial_sorted,
                flow_field.dudz_initial_sorted,
                flow_field_grid.x_sorted - x_i,
                flow_field_grid.y_sorted - y_i,
                flow_field_grid.z_sorted,
                rotor_diameter_i,
                hub_height_i,
                yaw_angle_i,
                turb_Cts[:, i:i+1],
                TSR_i,
                axial_induction_i,
                flow_field.wind_shear,
                scale=2.0,
            )

        # NOTE: exponential
        turb_u_wake, Ctmp = model_manager.velocity_model.function(
            i,
            x_i,
            y_i,
            z_i,
            u_i,
            deflection_field,
            yaw_angle_i,
            turbine_grid_flow_field.turbulence_intensity_field_sorted_avg,
            turb_Cts,
            turbine_grid_farm.rotor_diameters_sorted[:, :, None, None],
            turb_u_wake,
            Ctmp,
            **deficit_model_args,
        )

        wake_added_turbulence_intensity = model_manager.turbulence_model.function(
            ambient_turbulence_intensities,
            flow_field_grid.x_sorted,
            x_i,
            rotor_diameter_i,
            axial_induction_i
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = np.where(turb_u_wake > 0.05, 1, 0)

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * (flow_field_grid.x_sorted > x_i)
            * (np.abs(y_i - flow_field_grid.y_sorted) < 2 * rotor_diameter_i)
            * (flow_field_grid.x_sorted <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbulence_intensity_field = np.maximum(
            np.sqrt(ti_added**2 + ambient_turbulence_intensities**2), turbulence_intensity_field
        )

        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake

    flow_field.u_sorted = flow_field.u_initial_sorted - turb_u_wake
    flow_field.turbulence_intensity_field_sorted = turbulence_intensity_field


def turbopark_solver(
    farm: Farm,
    flow_field: FlowField,
    grid: TurbineGrid,
    model_manager: WakeModelManager
) -> None:
    # Algorithm
    # For each turbine, calculate its effect on every downstream turbine.
    # For the current turbine, we are calculating the deficit that it adds to downstream turbines.
    # Integrate this into the main data structure.
    # Move on to the next turbine.

    # <<interface>>
    deflection_model_args = model_manager.deflection_model.prepare_function(grid, flow_field)
    deficit_model_args = model_manager.velocity_model.prepare_function(grid, flow_field)

    # This is u_wake
    wake_field = np.zeros_like(flow_field.u_initial_sorted)
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)
    shape = (farm.n_turbines,) + np.shape(flow_field.u_initial_sorted)
    velocity_deficit = np.zeros(shape)
    deflection_field = np.zeros_like(flow_field.u_initial_sorted)

    # Set up turbulence arrays
    turbine_turbulence_intensity = flow_field.turbulence_intensities[:, None, None, None]
    turbine_turbulence_intensity = np.repeat(turbine_turbulence_intensity, farm.n_turbines, axis=1)

    # Ambient turbulent intensity should be a copy of n_findex-long turbulence_intensities
    # with extra dimension to reach 4d
    ambient_turbulence_intensities = flow_field.turbulence_intensities.copy()
    ambient_turbulence_intensities = ambient_turbulence_intensities[:, None, None, None]

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):
        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        y_i = np.mean(grid.y_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        z_i = np.mean(grid.z_sorted[:, i:i+1], axis=(2, 3), keepdims=True)

        Cts = thrust_coefficient(
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            thrust_coefficient_functions=farm.turbine_thrust_coefficient_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions,
        )

        ct_i = thrust_coefficient(
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            thrust_coefficient_functions=farm.turbine_thrust_coefficient_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions,
        )
        # Since we are filtering for the i'th turbine in the thrust coefficient function,
        # get the first index here (0:1)
        ct_i = ct_i[:, 0:1, None, None]
        axial_induction_i = axial_induction(
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            axial_induction_functions=farm.turbine_axial_induction_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions,
        )
        # Since we are filtering for the i'th turbine in the axial induction function,
        # get the first index here (0:1)
        axial_induction_i = axial_induction_i[:, 0:1, None, None]
        yaw_angle_i = farm.yaw_angles_sorted[:, i:i+1, None, None]
        rotor_diameter_i = farm.rotor_diameters_sorted[:, i:i+1, None, None]

        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i


        if model_manager.enable_secondary_steering:
            raise NotImplementedError(
                "Secondary steering not available for this model.")

        # Model calculations
        # NOTE: exponential
        if np.any(farm.yaw_angles_sorted):
            model_manager.deflection_model.logger.warning(
                "WARNING: Deflection with the TurbOPark model has not been fully validated. "
                "This is an initial implementation, and we advise you use at your own risk "
                "and perform a thorough examination of the results."
            )
            for ii in range(i):
                x_ii = np.mean(grid.x_sorted[:, ii:ii+1], axis=(2, 3), keepdims=True)
                y_ii = np.mean(grid.y_sorted[:, ii:ii+1], axis=(2, 3), keepdims=True)

                yaw_ii = farm.yaw_angles_sorted[:, ii:ii+1, None, None]
                turbulence_intensity_ii = turbine_turbulence_intensity[:, ii:ii+1]
                ct_ii = thrust_coefficient(
                    velocities=flow_field.u_sorted,
                    turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
                    air_density=flow_field.air_density,
                    yaw_angles=farm.yaw_angles_sorted,
                    tilt_angles=farm.tilt_angles_sorted,
                    power_setpoints=farm.power_setpoints_sorted,
                    awc_modes=farm.awc_modes_sorted,
                    awc_amplitudes=farm.awc_amplitudes_sorted,
                    thrust_coefficient_functions=farm.turbine_thrust_coefficient_functions,
                    tilt_interps=farm.turbine_tilt_interps,
                    correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
                    turbine_type_map=farm.turbine_type_map_sorted,
                    turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
                    ix_filter=[ii],
                    average_method=grid.average_method,
                    cubature_weights=grid.cubature_weights,
                    multidim_condition=flow_field.multidim_conditions,
                )
                ct_ii = ct_ii[:, 0:1, None, None]
                rotor_diameter_ii = farm.rotor_diameters_sorted[:, ii:ii+1, None, None]

                deflection_field_ii = model_manager.deflection_model.function(
                    x_ii,
                    y_ii,
                    yaw_ii,
                    turbulence_intensity_ii,
                    ct_ii,
                    rotor_diameter_ii,
                    **deflection_model_args,
                )

                deflection_field[:, ii:ii+1, :, :] = deflection_field_ii[:, i:i+1, :, :]

        if model_manager.enable_transverse_velocities:
            raise NotImplementedError(
                "Transverse velocities not used in this model.")

        if model_manager.enable_yaw_added_recovery:
            raise NotImplementedError(
                "Yaw added recovery not used in this model.")

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            turbine_turbulence_intensity,
            Cts[:, :, None, None],
            rotor_diameter_i,
            farm.rotor_diameters_sorted[:, :, None, None],
            i,
            deflection_field,
            **deficit_model_args,
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial_sorted
        )

        wake_added_turbulence_intensity = model_manager.turbulence_model.function(
            ambient_turbulence_intensities,
            grid.x_sorted,
            x_i,
            rotor_diameter_i,
            axial_induction_i
        )

        # TODO: leaving this in for GCH quantities; will need to find another way to
        # compute area_overlap as the current wake deficit is solved for only upstream
        # turbines; could use WAT_upstream
        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = (
            np.sum(
                velocity_deficit * flow_field.u_initial_sorted > 0.05,
                axis=(2, 3),
                keepdims=True
            )
            / (grid.grid_resolution * grid.grid_resolution)
        )

        # Modify wake added turbulence by wake area overlap
        downstream_influence_length = 15 * rotor_diameter_i
        ti_added = (
            area_overlap
            * np.nan_to_num(wake_added_turbulence_intensity, posinf=0.0)
            * (grid.x_sorted > x_i)
            * (np.abs(y_i - grid.y_sorted) < 2 * rotor_diameter_i)
            * (grid.x_sorted <= downstream_influence_length + x_i)
        )

        # Combine turbine TIs with WAT
        turbine_turbulence_intensity = np.maximum(
            np.sqrt(ti_added**2 + ambient_turbulence_intensities**2), turbine_turbulence_intensity
        )

        flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake

    flow_field.turbulence_intensity_field_sorted = turbine_turbulence_intensity
    flow_field.turbulence_intensity_field_sorted_avg = np.mean(
        turbine_turbulence_intensity,
        axis=(2, 3),
        keepdims=True
    )


def full_flow_turbopark_solver(
    farm: Farm,
    flow_field: FlowField,
    flow_field_grid: FlowFieldGrid,
    model_manager: WakeModelManager
) -> None:
    raise NotImplementedError("Plotting for the TurbOPark model is not currently implemented.")


def empirical_gauss_solver(
    farm: Farm,
    flow_field: FlowField,
    grid: TurbineGrid,
    model_manager: WakeModelManager
) -> NDArrayFloat:
    """
    Algorithm:
    For each turbine, calculate its effect on every downstream turbine.
    For the current turbine, we are calculating the deficit that it adds to downstream turbines.
    Integrate this into the main data structure.
    Move on to the next turbine.

    Args:
        farm (Farm)
        flow_field (FlowField)
        grid (TurbineGrid)
        model_manager (WakeModelManager)

    Raises:
        NotImplementedError: Raised if secondary steering is enabled with the EmGauss model.
        NotImplementedError: Raised if transverse velocities is enabled with the EmGauss model.

    Returns:
        NDArrayFloat: wake induced mixing field primarily for use in the full-flow EmGauss solver
    """


    # <<interface>>
    deflection_model_args = model_manager.deflection_model.prepare_function(grid, flow_field)
    deficit_model_args = model_manager.velocity_model.prepare_function(grid, flow_field)

    # This is u_wake
    wake_field = np.zeros_like(flow_field.u_initial_sorted)
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)

    x_locs = np.mean(grid.x_sorted, axis=(2, 3))[:,:,None]
    downstream_distance_D = x_locs - np.transpose(x_locs, axes=(0,2,1))
    downstream_distance_D = downstream_distance_D / \
        np.repeat(farm.rotor_diameters_sorted[:,:,None], grid.n_turbines, axis=-1)
    downstream_distance_D = np.maximum(downstream_distance_D, 0.1) # For ease
    # Initialize the mixing factor model using TI if specified
    initial_mixing_factor = model_manager.turbulence_model.atmospheric_ti_gain * np.eye(
        grid.n_turbines
    )
    mixing_factor = np.repeat(
        initial_mixing_factor[None, :, :],
        flow_field.n_findex,
        axis=0
    )
    mixing_factor = mixing_factor * flow_field.turbulence_intensities[:, None, None]

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(grid.x_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        y_i = np.mean(grid.y_sorted[:, i:i+1], axis=(2, 3), keepdims=True)
        z_i = np.mean(grid.z_sorted[:, i:i+1], axis=(2, 3), keepdims=True)

        ct_i = thrust_coefficient(
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            thrust_coefficient_functions=farm.turbine_thrust_coefficient_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions,
        )
        # Since we are filtering for the i'th turbine in the thrust coefficient function,
        # get the first index here (0:1)
        ct_i = ct_i[:, 0:1, None, None]
        axial_induction_i = axial_induction(
            velocities=flow_field.u_sorted,
            turbulence_intensities=flow_field.turbulence_intensity_field_sorted,
            air_density=flow_field.air_density,
            yaw_angles=farm.yaw_angles_sorted,
            tilt_angles=farm.tilt_angles_sorted,
            power_setpoints=farm.power_setpoints_sorted,
            awc_modes=farm.awc_modes_sorted,
            awc_amplitudes=farm.awc_amplitudes_sorted,
            axial_induction_functions=farm.turbine_axial_induction_functions,
            tilt_interps=farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=grid.average_method,
            cubature_weights=grid.cubature_weights,
            multidim_condition=flow_field.multidim_conditions,
        )
        # Since we are filtering for the i'th turbine in the axial induction function,
        # get the first index here (0:1)
        axial_induction_i = axial_induction_i[:, 0:1, None, None]
        yaw_angle_i = farm.yaw_angles_sorted[:, i:i+1, None, None]
        awc_mode_i = farm.awc_modes_sorted[:, i:i+1, None, None]
        awc_amplitude_i = farm.awc_amplitudes_sorted[:, i:i+1, None, None]
        awc_frequency_i = farm.awc_frequencies_sorted[:, i:i+1, None, None]
        hub_height_i = farm.hub_heights_sorted[:, i:i+1, None, None]
        rotor_diameter_i = farm.rotor_diameters_sorted[:, i:i+1, None, None]

        # Secondary steering not currently implemented in EmGauss model
        # effective_yaw_i = np.zeros_like(yaw_angle_i)
        # effective_yaw_i += yaw_angle_i

        average_velocities = average_velocity(
            flow_field.u_sorted,
            method=grid.average_method,
            cubature_weights=grid.cubature_weights
        )
        tilt_angle_i = farm.calculate_tilt_for_eff_velocities(average_velocities)
        tilt_angle_i = tilt_angle_i[:, i:i+1, None, None]

        if model_manager.enable_secondary_steering:
            raise NotImplementedError(
                "Secondary steering not available for this model.")

        if model_manager.enable_transverse_velocities:
            raise NotImplementedError(
                "Transverse velocities not used in this model.")

        if model_manager.enable_yaw_added_recovery:
            # Influence of yawing on turbine's own wake
            mixing_factor[:, i:i+1, i] += \
                yaw_added_wake_mixing(
                    axial_induction_i,
                    yaw_angle_i,
                    1,
                    model_manager.deflection_model.yaw_added_mixing_gain
                )

        if model_manager.enable_active_wake_mixing:
            # Influence of awc on turbine's own wake
            mixing_factor[:, i:i+1, i] += \
                awc_added_wake_mixing(
                    awc_mode_i,
                    awc_amplitude_i,
                    awc_frequency_i,
                    model_manager.velocity_model.awc_wake_exp,
                    model_manager.velocity_model.awc_wake_denominator
                )

        # Extract total wake induced mixing for turbine i
        mixing_i = np.linalg.norm(
            mixing_factor[:, i:i+1, :, None],
            ord=2, axis=2, keepdims=True
        )

        # Model calculations
        # NOTE: exponential
        deflection_field_y, deflection_field_z = model_manager.deflection_model.function(
            x_i,
            y_i,
            yaw_angle_i,
            tilt_angle_i,
            mixing_i,
            ct_i,
            rotor_diameter_i,
            **deflection_model_args
        )

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            axial_induction_i,
            deflection_field_y,
            deflection_field_z,
            yaw_angle_i,
            tilt_angle_i,
            mixing_i,
            ct_i,
            hub_height_i,
            rotor_diameter_i,
            **deficit_model_args
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial_sorted
        )

        # Calculate wake overlap for wake-added turbulence (WAT)
        area_overlap = np.sum(velocity_deficit * flow_field.u_initial_sorted > 0.05, axis=(2, 3))\
            / (grid.grid_resolution * grid.grid_resolution)

        # Compute wake induced mixing factor
        mixing_factor[:,:,i] += \
            area_overlap * model_manager.turbulence_model.function(
                axial_induction_i, downstream_distance_D[:,:,i]
            )
        if model_manager.enable_yaw_added_recovery:
            mixing_factor[:,:,i] += \
                area_overlap * yaw_added_wake_mixing(
                axial_induction_i,
                yaw_angle_i,
                downstream_distance_D[:,:,i],
                model_manager.deflection_model.yaw_added_mixing_gain
            )

        flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake

    return mixing_factor


def full_flow_empirical_gauss_solver(
    farm: Farm,
    flow_field: FlowField,
    flow_field_grid: FlowFieldGrid,
    model_manager: WakeModelManager
) -> None:

    # Get the flow quantities and turbine performance
    turbine_grid_farm = copy.deepcopy(farm)
    turbine_grid_flow_field = copy.deepcopy(flow_field)

    turbine_grid_farm.construct_turbine_map()
    turbine_grid_farm.construct_turbine_thrust_coefficient_functions()
    turbine_grid_farm.construct_turbine_axial_induction_functions()
    turbine_grid_farm.construct_turbine_power_functions()
    turbine_grid_farm.construct_hub_heights()
    turbine_grid_farm.construct_rotor_diameters()
    turbine_grid_farm.construct_turbine_TSRs()
    turbine_grid_farm.construct_turbine_ref_tilts()
    turbine_grid_farm.construct_turbine_tilt_interps()
    turbine_grid_farm.construct_turbine_correct_cp_ct_for_tilt()
    turbine_grid_farm.set_tilt_to_ref_tilt(flow_field.n_findex)

    turbine_grid = TurbineGrid(
        turbine_coordinates=turbine_grid_farm.coordinates,
        turbine_diameters=turbine_grid_farm.rotor_diameters,
        wind_directions=turbine_grid_flow_field.wind_directions,
        grid_resolution=3,
    )
    turbine_grid_farm.expand_farm_properties(
        turbine_grid_flow_field.n_findex,
        turbine_grid.sorted_coord_indices
    )
    turbine_grid_flow_field.initialize_velocity_field(turbine_grid)
    turbine_grid_farm.initialize(turbine_grid.sorted_indices)
    wim_field = empirical_gauss_solver(
        turbine_grid_farm,
        turbine_grid_flow_field,
        turbine_grid,
        model_manager
    )

    # Create placeholder for TI, which is not currently used in the EmG model
    n_points = flow_field_grid.x_sorted.shape[1]
    ambient_turbulence_intensities = flow_field.turbulence_intensities[:, None, None, None]
    ambient_turbulence_intensities = np.repeat(ambient_turbulence_intensities, n_points, axis=1)
    turbulence_intensity_field = ambient_turbulence_intensities.copy()

    ### Referring to the quantities from above, calculate the wake in the full grid

    # Use full flow_field here to use the full grid in the wake models
    deflection_model_args = model_manager.deflection_model.prepare_function(
        flow_field_grid, flow_field
    )
    deficit_model_args = model_manager.velocity_model.prepare_function(flow_field_grid, flow_field)

    wake_field = np.zeros_like(flow_field.u_initial_sorted)
    v_wake = np.zeros_like(flow_field.v_initial_sorted)
    w_wake = np.zeros_like(flow_field.w_initial_sorted)

    # Calculate the velocity deficit sequentially from upstream to downstream turbines
    for i in range(flow_field_grid.n_turbines):

        # Get the current turbine quantities
        x_i = np.mean(turbine_grid.x_sorted[:, i:i+1], axis=(2,3), keepdims=True)
        y_i = np.mean(turbine_grid.y_sorted[:, i:i+1], axis=(2,3), keepdims=True)
        z_i = np.mean(turbine_grid.z_sorted[:, i:i+1], axis=(2,3), keepdims=True)

        ct_i = thrust_coefficient(
            velocities=turbine_grid_flow_field.u_sorted,
            turbulence_intensities=turbine_grid_flow_field.turbulence_intensity_field_sorted,
            air_density=turbine_grid_flow_field.air_density,
            yaw_angles=turbine_grid_farm.yaw_angles_sorted,
            tilt_angles=turbine_grid_farm.tilt_angles_sorted,
            power_setpoints=turbine_grid_farm.power_setpoints_sorted,
            awc_modes=turbine_grid_farm.awc_modes_sorted,
            awc_amplitudes=turbine_grid_farm.awc_amplitudes_sorted,
            thrust_coefficient_functions=turbine_grid_farm.turbine_thrust_coefficient_functions,
            tilt_interps=turbine_grid_farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=turbine_grid_farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=turbine_grid_farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=turbine_grid.average_method,
            cubature_weights=turbine_grid.cubature_weights,
            multidim_condition=turbine_grid_flow_field.multidim_conditions,
        )
        # Since we are filtering for the i'th turbine in the thrust coefficient function,
        # get the first index here (0:1)
        ct_i = ct_i[:, 0:1, None, None]
        axial_induction_i = axial_induction(
            velocities=turbine_grid_flow_field.u_sorted,
            turbulence_intensities=turbine_grid_flow_field.turbulence_intensity_field_sorted,
            air_density=turbine_grid_flow_field.air_density,
            yaw_angles=turbine_grid_farm.yaw_angles_sorted,
            tilt_angles=turbine_grid_farm.tilt_angles_sorted,
            power_setpoints=turbine_grid_farm.power_setpoints_sorted,
            awc_modes=turbine_grid_farm.awc_modes_sorted,
            awc_amplitudes=turbine_grid_farm.awc_amplitudes_sorted,
            axial_induction_functions=turbine_grid_farm.turbine_axial_induction_functions,
            tilt_interps=turbine_grid_farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=turbine_grid_farm.correct_cp_ct_for_tilt_sorted,
            turbine_type_map=turbine_grid_farm.turbine_type_map_sorted,
            turbine_power_thrust_tables=turbine_grid_farm.turbine_power_thrust_tables,
            ix_filter=[i],
            average_method=turbine_grid.average_method,
            cubature_weights=turbine_grid.cubature_weights,
            multidim_condition=turbine_grid_flow_field.multidim_conditions,
        )
        # Since we are filtering for the i'th turbine in the axial induction function,
        # get the first index here (0:1)
        axial_induction_i = axial_induction_i[:, 0:1, None, None]
        yaw_angle_i = turbine_grid_farm.yaw_angles_sorted[:, i:i+1, None, None]
        hub_height_i = turbine_grid_farm.hub_heights_sorted[:, i:i+1, None, None]
        rotor_diameter_i = turbine_grid_farm.rotor_diameters_sorted[:, i:i+1, None, None]
        wake_induced_mixing_i = wim_field[:, i:i+1, :, None].sum(axis=2, keepdims=1)
        effective_yaw_i = np.zeros_like(yaw_angle_i)
        effective_yaw_i += yaw_angle_i

        average_velocities = average_velocity(
            turbine_grid_flow_field.u_sorted,
            method=turbine_grid.average_method,
            cubature_weights=turbine_grid.cubature_weights
        )
        tilt_angle_i = turbine_grid_farm.calculate_tilt_for_eff_velocities(average_velocities)
        tilt_angle_i = tilt_angle_i[:, i:i+1, None, None]

        if model_manager.enable_secondary_steering:
            raise NotImplementedError(
                "Secondary steering not available for this model.")

        if model_manager.enable_transverse_velocities:
            raise NotImplementedError(
                "Transverse velocities not used in this model.")

        # Model calculations
        # NOTE: exponential
        deflection_field_y, deflection_field_z = model_manager.deflection_model.function(
            x_i,
            y_i,
            effective_yaw_i,
            tilt_angle_i,
            wake_induced_mixing_i,
            ct_i,
            rotor_diameter_i,
            **deflection_model_args
        )

        # NOTE: exponential
        velocity_deficit = model_manager.velocity_model.function(
            x_i,
            y_i,
            z_i,
            axial_induction_i,
            deflection_field_y,
            deflection_field_z,
            yaw_angle_i,
            tilt_angle_i,
            wake_induced_mixing_i,
            ct_i,
            hub_height_i,
            rotor_diameter_i,
            **deficit_model_args
        )

        wake_field = model_manager.combination_model.function(
            wake_field,
            velocity_deficit * flow_field.u_initial_sorted
        )

        flow_field.u_sorted = flow_field.u_initial_sorted - wake_field
        flow_field.v_sorted += v_wake
        flow_field.w_sorted += w_wake
        flow_field.turbulence_intensity_field_sorted = turbulence_intensity_field
