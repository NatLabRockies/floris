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
    n_findex = 1


    dx = model_manager.velocity_model.dx_D * farm.rotor_diameters_sorted.min()
    dy = dx
    dz = dx
    dx /= 2  # ensure smaller dx for numerical stability
    x_min = 0 - farm.rotor_diameters_sorted.max() # 1D upstream of first turbine
    x_max = grid.x_sorted.max() + farm.rotor_diameters_sorted.max() # 1D downstream of last turbine
    y_min = grid.y_sorted.min() - farm.rotor_diameters_sorted.max()
    y_max = grid.y_sorted.max() + farm.rotor_diameters_sorted.max()
    x_1d = np.arange(x_min, x_max, dx, dtype=floris_float_type) # x coordinates
    y_1d = np.arange(y_min, y_max, dy, dtype=floris_float_type) # y coordinates
    z_1d = np.arange(
        0, farm.hub_heights_sorted.max()+farm.rotor_diameters_sorted.max(), dz, dtype=floris_float_type
    ) # Go up to 0.5D above highest tip point
    u_sheared = np.maximum(
        (z_1d / flow_field.reference_wind_height) ** flow_field.wind_shear
        * flow_field.wind_speeds, 3
    )

    n_x_planes = x_1d.shape[0]
    n_y_planes = y_1d.shape[0]
    n_z_planes = z_1d.shape[0]  

    # Create large arrays (memory intensive!)
    x, y, z = np.meshgrid(x_1d, y_1d, z_1d, indexing='ij')
    x = np.repeat(x[None,:,:,:], n_findex, axis=0)
    y = np.repeat(y[None,:,:,:], n_findex, axis=0)
    z = np.repeat(z[None,:,:,:], n_findex, axis=0)

    # Use the same number of x and y points for generality once we get to multiple
    # findices at once (can consider revising later)
    # Not yet handling heterogeneity; will work that in later.
    u_freestream = np.tile(u_sheared, (n_findex, n_x_planes, n_y_planes, 1))
    v_freestream = np.zeros_like(x, dtype=floris_float_type) # May not be needed. Always zeros?
    w_freestream = np.zeros_like(x, dtype=floris_float_type) # May not be needed. Always zeros?
    u_waked = np.zeros_like(x, dtype=floris_float_type)
    v_waked = np.zeros_like(x, dtype=floris_float_type)
    w_waked = np.zeros_like(x, dtype=floris_float_type)

    # Code to generate one or multiple turbine indices that correspond to a certain x location
    # Each element in the list turbines_in_plane should be a list of the turbines that appear at
    # that plane's x location (i.e., often an empty list).
    turbines_in_plane = [[] for _ in range(n_x_planes)]

    # Let's try to extract the turbines in each plane
    Nt = grid.x_sorted.shape[1]  # Number of turbines
    for t in range(Nt):
        # Get the x coordinate of the turbine
        turbine_x = np.mean(grid.x_sorted[:, t, :, :])
        # Find the index of the plane that contains this turbine
        ip = np.argmin(np.abs(x_1d - turbine_x))
        turbines_in_plane[ip] += [t]

    for i in range(1, n_x_planes):
        u_freestream_plane = np.maximum(u_freestream[:, i, :, :], 3) # Ensure freestream velocity is above 3 m/s (should really be 20% of U_inf)
        v_freestream_plane = v_freestream[:, i, :, :]
        w_freestream_plane = w_freestream[:, i, :, :]
        u_waked_plane = u_waked[:, i, :, :]
        v_waked_plane = v_waked[:, i, :, :]
        w_waked_plane = w_waked[:, i, :, :]

        # Compute the numerical viscosity needed for stability
        Re = model_manager.velocity_model.Re

        ### ... solver code here ...
        #u_waked[:, i, :, :] = u_waked[:, i-1, :, :]
 
        f = 4.  # turbulence visvosity factor

        # Extract the 2D slices for the current plane
        u_prev = u_waked[0, i-1, :, :]     # shape (ny, nz)
        u_fs   = u_freestream_plane[0]     # shape (ny, nz)
        v_fs   = v_freestream_plane[0]     # shape (ny, nz)
        w_fs   = w_freestream_plane[0]     # shape (ny, nz)
        nu_2d = np.maximum(u_fs * flow_field.reference_wind_height / Re, 8 * 100 / Re)

        # Debug shapes
        #print("u_prev shape:", u_prev.shape)
        #print("u_fs shape:", u_fs.shape)
        #print("v_fs shape:", v_fs.shape)
        #print("w_fs shape:", w_fs.shape)
        #print("nu_2d shape:", nu_2d.shape)
        #print("dx:", dx)
        #print("f:", f)

        # Run RK step
#        u_new = runge_kutta_step(
        u_new = model_manager.velocity_model.function(
            u_prev,
            dx,
            u_fs,
            v_fs,
            w_fs,
            dy,
            dz,
            f,
            nu_2d
        )

        #u_new = np.clip(u_new, -5, 0)
        #import matplotlib.pyplot as plt
        #plt.pcolormesh(u_new, label='u_new')
        #plt.colorbar(label='Velocity (m/s)')
        #plt.gca().set_aspect('equal', adjustable='box')
        #plt.show()

        # Assign it back
        u_waked[0, i, :, :] = u_new

        # If a turbine exists on this plane, calculate its inflow velocities as a subset of u_freestream_plane
        for t in turbines_in_plane[i]:
            turbine_x = grid.x_sorted[:,t,:,:]
            turbine_y = grid.y_sorted[:,t,:,:]
            turbine_z = grid.z_sorted[:,t,:,:]

            # Let's get the rotor location for now (should access properly later)
            t_x = np.mean(turbine_x)
            t_y = np.mean(turbine_y)
            t_z = np.mean(turbine_z)

            #rotor_diameter_i = farm.rotor_diameters_sorted[:, t:t+1, None, None].max()
            rotor_diameter_i = farm.rotor_diameters_sorted[:, t, None, None].max()

            # Create a mask for points inside the rotor disk at this x-plane
            # mask shape: (n_findex, n_y, n_z), True where (y, z) is inside rotor
            rotor_mask = ((y[:, i, :, :] - t_y) ** 2 + (z[:, i, :, :] - t_z) ** 2) < (rotor_diameter_i / 2)**2
            #print("Rotor mask shape:", rotor_mask.shape)
            # The filtered mask for points inside the rotor disk (wider)
            rotor_mask_filt = ((y[:, i, :, :] - t_y) ** 2 + (z[:, i, :, :] - t_z) ** 2) < (1.3 * rotor_diameter_i / 2)**2
            #print("Rotor mask shape:", rotor_mask_filt.shape)

            #print("Turbine:", t, "at x:", turbine_x[0,0,0], "y:", turbine_y[0,0,0], "z:", turbine_z[0,0,0])
            #print("Turbine coordinates:", turbine_x.shape, turbine_y.shape, turbine_z.shape)
            #print("Turbine x: ", turbine_x)
            #print("Turbine y: ", turbine_y)
            #print("Turbine z: ", turbine_z)
            # Pull values from u_waked_plane that are closest to the turbine locations
            #u_waked_plane_turbine = 10 * np.ones((3, 3)) # PLACEHOLDER
            #flow_field.u_sorted[0, t, :, :] = u_waked_plane_turbine
            u_rotor_values = u_freestream_plane[rotor_mask] + u_waked_plane[rotor_mask]
            u_rotor_disk = np.mean(u_rotor_values)
            flow_field.u_sorted[0, t, :, :] = u_rotor_disk  # or shape match if 2D needed


        #if turbines_in_plane[i]: # Just to avoid running unnecessarily if there are no turbines in plane
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
                ix_filter=[t],
                average_method=grid.average_method,
                cubature_weights=grid.cubature_weights,
                multidim_condition=flow_field.multidim_conditions,
            )
            print("C_T:", ct_i)

            a = (1. - np.sqrt(1. - ct_i * np.cos(0)**2)) / 2
            a = float(np.minimum(a, 0.35))  # force scalar  # Set a limit to guarantee numerical stability
            print("a:", a)

            # Apply induction to points inside the rotor disk
            u_waked_plane[rotor_mask] = - 2 * a * u_freestream_plane[rotor_mask]
            #u_waked_plane = gaussian_filter(u_waked_plane, 3) 
            # (this filtered the entire plane) u_waked[0, i, :, :] = gaussian_filter(u_waked_plane, 3)
            # Filter only around the rotor disk
            u_waked[:,i,:,:][rotor_mask_filt] = gaussian_filter(u_waked_plane[rotor_mask_filt], 2)

        
        # Ensure boundary conditions are satisfied
        u_waked[:, i, :, [0, -1]] = 0
        u_waked[:, i, [0, -1], :] = 0

    # This is the end of the loop that goes through the x planes.
    k = np.argmin(np.abs(z_1d - flow_field.reference_wind_height))
    #print("z_1d:", z_1d)    
    #print("dx:", dx)
    #print("k:", k)
    import matplotlib.pyplot as plt
    #plt.pcolormesh(x_1d, y_1d, u_freestream[0, :, :, k].T, label='Waked', shading='gouraud')
    plt.pcolormesh(x_1d, y_1d, u_waked[0, :, :, k].T, label='Waked', shading='gouraud',
                   #vmin=3, vmax=9,
                   )
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    print("Solve complete.")

    # Result: flow_field.u_sorted.
    return None