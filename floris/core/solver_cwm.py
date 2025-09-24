import numpy as np
from scipy.ndimage.filters import gaussian_filter

from floris.core import (
    Farm,
    FlowField,
    thrust_coefficient,
    TurbineGrid,
)
from floris.core.wake import WakeModelManager
from floris.type_dec import (
    floris_float_type,
    NDArrayFloat,
)


def curled_wake_solver(
    farm: Farm,
    flow_field: FlowField,
    grid: TurbineGrid,
    model_manager: WakeModelManager
) -> NDArrayFloat:
    # In the normal paradigm, flow_field contains quantities over the turbine
    # rotors (perhaps not the best name, but that's how it is).

    # I propose that we start by keeping it that way; i.e., flow_field.u_sorted
    # is the "final" output of the solve, with dimensions (n_findex, n_turbines, n_grid, n_grid),
    # with n_grid being a number of grid points used for averaging
    # (potentially urelated to the CWM grid).

    # I'm generating the data matrices for the CWM with an empty first dimension,
    # as a placeholder for a final implementation with an n_findex dimension.

    dx = model_manager.velocity_model.dx_D * farm.rotor_diameters_sorted.min()
    dy = model_manager.velocity_model.dy_D * farm.rotor_diameters_sorted.min()
    dz = model_manager.velocity_model.dz_D * farm.rotor_diameters_sorted.min()

    #dx /= 2  # ensure smaller dx for numerical stability
    x_min = 0 - farm.rotor_diameters_sorted.max() # 1D upstream of first turbine
    x_max = grid.x_sorted.max() + farm.rotor_diameters_sorted.max() # 1D downstream of last turbine
    y_min = grid.y_sorted.min() - 5*farm.rotor_diameters_sorted.max()
    y_max = grid.y_sorted.max() + 5*farm.rotor_diameters_sorted.max()
    x_1d = np.arange(x_min, x_max, dx, dtype=floris_float_type) # x coordinates
    y_1d = np.arange(y_min, y_max, dy, dtype=floris_float_type) # y coordinates
    z_1d = np.arange(
        0, farm.hub_heights_sorted.max()+ 2*farm.rotor_diameters_sorted.max(),
        dz,
        dtype=floris_float_type
    ) # Go up to 0.5D above highest tip point

    n_x_planes = x_1d.shape[0]

    # Create large arrays (memory intensive!)
    x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x = np.repeat(x[None,:,:,:], flow_field.n_findex, axis=0)
    y = np.repeat(y[None,:,:,:], flow_field.n_findex, axis=0)
    z = np.repeat(z[None,:,:,:], flow_field.n_findex, axis=0)

    # Use the same number of x and y points for generality once we get to multiple
    # findices at once (can consider revising later)
    # Not yet handling heterogeneity; will work that in later.
    u_freestream = (
        np.tile(
            flow_field.wind_speeds[:,None,None,None],
            (1, n_x_planes, y_1d.shape[0], z_1d.shape[0])
        )
        * ((z_1d / flow_field.reference_wind_height) ** flow_field.wind_shear)[None, None, None, :]
    )
    u_freestream = np.maximum(u_freestream, 3)
    # TODO: what is the 3 for? Does the solve not work for lower wind speeds? We will need to handle
    # this better, if so.
    v_freestream = np.zeros_like(x, dtype=floris_float_type) # May not be needed. Always zeros?
    w_freestream = np.zeros_like(x, dtype=floris_float_type) # May not be needed. Always zeros?
    u_waked = np.zeros_like(x, dtype=floris_float_type)
    v_waked = np.zeros_like(x, dtype=floris_float_type)
    w_waked = np.zeros_like(x, dtype=floris_float_type)

    #
    # Add flow features
    #

    # Add boundary layer
    print('Adding boundary layer:')
    u_sheared = (z / flow_field.reference_wind_height) ** flow_field.wind_shear * u_freestream
    u_freestream = np.maximum(u_sheared, u_freestream*.3)  # Ensure freestream is above 3 m/s

    # Add veer
    if np.abs(flow_field.wind_veer) > 1e-6:
        print("Adding veer:")
        v_sheared = model_manager.velocity_model.add_veer_theta(
            z,
            u_freestream,
            theta_deg=flow_field.wind_veer,
            th=farm.hub_heights_sorted.max(),
            D=farm.rotor_diameters_sorted.max()
        )
        print("V shear shape:", v_sheared.shape)
        v_freestream += v_sheared

    # Compute the numerical viscosity needed for stability
    Re = model_manager.velocity_model.Re
    # Minimum viscosity for numerical stability
    nu_min = u_freestream * flow_field.reference_wind_height / Re
    # Initialize the turbulence viscosity based on the standard curled wake model formulation
    nu_t = model_manager.velocity_model.initialize_turbulence_viscosity(u_freestream, z, dz, nu_min)

    # Code to generate one or multiple turbine indices that correspond to a certain x location
    # Each element in the list turbines_in_plane should be a list of the turbines that appear at
    # that plane's x location (i.e., often an empty list).
    turbine_plane_map = np.zeros((flow_field.n_findex, farm.n_turbines), dtype=int)

    # Let's try to extract the turbines in each plane
    for t in range(farm.n_turbines):
        # Get the x coordinate of the turbine
        turbine_x = np.mean(grid.x_sorted[:, t, :, :], axis=(1,2))
        # Find the index of the plane that contains this turbine
        ip = np.argmin(np.abs(x_1d.reshape(-1,1) - turbine_x.reshape(1,-1)), axis=0)
        turbine_plane_map[:, t] = ip

    count_inner_loops = 0 # Temporary

    for i in range(1, n_x_planes):
        # Ensure freestream velocity is above 3 m/s (should really be 20% of U_inf)
        u_freestream_plane = np.maximum(u_freestream[:, i, :, :], 3) # TODO: 3 m/s minimum
        v_freestream_plane = v_freestream[:, i, :, :]
        w_freestream_plane = w_freestream[:, i, :, :]
        u_waked_plane = u_waked[:, i, :, :]
        v_waked_plane = v_waked[:, i, :, :]  # noqa: F841 TODO: remove if not used
        w_waked_plane = w_waked[:, i, :, :]  # noqa: F841 TODO: remove if not used

        # Extract the 2D slices for the current plane
        u_prev = u_waked[:, i-1, :, :]     # shape (n_findex, ny, nz)
        v_prev = v_waked[:, i-1, :, :]     # shape (n_findex, ny, nz)
        w_prev = w_waked[:, i-1, :, :]     # shape (n_findex, ny, nz)
        u_fs   = u_freestream_plane     # shape (n_findex, ny, nz)
        v_fs   = v_freestream_plane     # shape (n_findex, ny, nz)
        w_fs   = w_freestream_plane     # shape (n_findex, ny, nz)
        nu_2d = nu_t[:, i, :, :]  # shape (n_findex, ny, nz)

        # Run Runge-Kutta step
        u_new = model_manager.velocity_model.function(
            u_prev,
            dx,
            u_fs,
            v_fs + v_prev,
            w_fs + w_prev,
            dy,
            dz,
            nu_2d
        )

        # Assign it back
        u_waked[:, i, :, :] = u_new

        # A simple evolution model to scale v the same way that U has scaled
        # This saves all the work of resolving the transport equation (du/dx~dv/dx)
        U = u_freestream_plane[0, :, :]
        fact = (U + u_waked[0, i-1, :, :]) / (U + u_waked[0, i, :, :])
        # This ensures that V and W do not become larger (they should always decay)
        fact = np.clip(fact, .1, 1.)
        v_waked[:,i,:,:] = v_waked[:, i-1, :, :] * fact
        w_waked[:,i,:,:] = w_waked[:, i-1, :, :] * fact

        # For this plane, find the findex-turbines that are present and modify the flow
        findex_turbine_indices = np.argwhere(turbine_plane_map == i)
        for fti in findex_turbine_indices:
            count_inner_loops += 1 # Temporary

            f, t = fti  # f is the findex, t is the turbine index

            # Let's get the rotor location for now (should access properly later)
            t_y = np.mean(grid.y_sorted[f,t,:,:])
            t_z = np.mean(grid.z_sorted[f,t,:,:])
            rotor_diameter_t = farm.rotor_diameters_sorted[f, t]

            # Create a mask for points inside the rotor disk at this x-plane
            # mask shape: (n_findex, n_y, n_z), True where (y, z) is inside rotor
            rotor_mask = (
                ((y[f, i, :, :] - t_y) ** 2 + (z[f, i, :, :] - t_z) ** 2)
                < (rotor_diameter_t / 2)**2
            )
            #print("Rotor mask shape:", rotor_mask.shape)
            # The filtered mask for points inside the rotor disk (wider)
            rotor_mask_filt = (
                ((y[f, i, :, :] - t_y) ** 2 + (z[f, i, :, :] - t_z) ** 2)
                < (1.3 * rotor_diameter_t / 2)**2
            )
            # TODO: where does this 1.3 factor come from? Is it a user setting?

            u_rotor_values = u_freestream_plane[f, rotor_mask] + u_waked_plane[f, rotor_mask]
            u_rotor_disk = np.mean(u_rotor_values) # TODO: Consider shear?
            flow_field.u_sorted[f, t, :, :] = u_rotor_disk  # or shape match if 2D needed

            ct = thrust_coefficient(
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
                ix_filter=[t],
                average_method=grid.average_method,
                cubature_weights=grid.cubature_weights,
                multidim_condition=flow_field.multidim_conditions,
            )[f]

            a = (1. - np.sqrt(1. - ct * np.cos(0)**2)) / 2

            # TODO: is there a reason not to use the axial induction method?
            # force scalar  # Set a limit to guarantee numerical stability
            a = float(np.minimum(a, 0.35)) #
            # print("a:", a)

            # Apply induction to points inside the rotor disk
            u_waked_plane[f, rotor_mask] = -2 * a * u_freestream_plane[f, rotor_mask]
            #u_waked_plane = gaussian_filter(u_waked_plane, 3)
            # (this filtered the entire plane)
            # u_waked[0, i, :, :] = gaussian_filter(u_waked_plane, 3)
            # Filter only around the rotor disk
            u_waked[f,i,:,:][rotor_mask_filt] = gaussian_filter(
                u_waked_plane[f, rotor_mask_filt],
                2
            )

            # Add Curl
            vcurl, wcurl = model_manager.velocity_model.add_curl(
                y[:, i, :, :]-t_y, #[rotor_mask_filt],
                z[:, i, :, :]-t_z, #[rotor_mask_filt],
                D=rotor_diameter_t,
                Ct=ct,
                Uh=u_rotor_disk,
                alpha=np.deg2rad(farm.yaw_angles_sorted[f, t]),
                tilt=np.deg2rad(farm.tilt_angles_sorted[f, t]),
                ground=True,
                th=farm.hub_heights_sorted[f, t],
                N=20,
            )
            v_waked[:,i,:,:] += vcurl
            w_waked[:,i,:,:] += wcurl

        # Ensure boundary conditions are satisfied
        u_waked[:, i, :, [0, -1]] = 0
        u_waked[:, i, [0, -1], :] = 0

    # This is the end of the loop that goes through the x planes.
    # k = np.argmin(np.abs(z_1d - flow_field.reference_wind_height))
    #print("z_1d:", z_1d)
    #print("dx:", dx)
    # #print("k:", k)
    # import matplotlib.pyplot as plt

    # #plt.pcolormesh(x_1d, y_1d, u_freestream[0, :, :, k].T, label='Waked', shading='gouraud')
    # plt.pcolormesh(x_1d, y_1d, u_waked[0, :, :, k].T, label='Waked', shading='gouraud',
    #                #vmin=3, vmax=9,
    #                )
    # plt.colorbar()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    # print("Solve complete.")

    # Result: flow_field.u_sorted.
    #print("Inner loop ran {0} times".format(count_inner_loops))
    return None
