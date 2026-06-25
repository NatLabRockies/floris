"""
Farm and turbine reading functions for windIO integration.

This module handles extraction of wind farm layouts and turbine specifications
from windIO data formats.
"""

import numpy as np
from typing import Dict, Any, List

from floris.turbine_library import build_cosine_loss_turbine_dict
from .utils import TrackedDict


def create_generic_power_curve(
    rated_power: float,
    rated_wind_speed: float,
    cutin_wind_speed: float,
    cutout_wind_speed: float,
    n_points: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a generic power curve from basic turbine parameters.
    
    Uses a quadratic (omega-squared law) interpolation between cut-in and rated wind speed,
    plateau at rated power between rated and cut-out wind speed, and zero power elsewhere.
    
    Args:
        rated_power: Rated power in Watts
        rated_wind_speed: Rated wind speed in m/s
        cutin_wind_speed: Cut-in wind speed in m/s
        cutout_wind_speed: Cut-out wind speed in m/s
        n_points: Number of points in the power curve (default: 50)
        
    Returns:
        Tuple of (power_wind_speeds, power_data) as numpy arrays
    """
    # Generate wind speed array
    power_wind_speeds = np.linspace(0, cutout_wind_speed + 5, n_points)
    power_data = np.zeros(n_points)
    
    # Omega-squared law region (cutin to rated)
    for i, ws in enumerate(power_wind_speeds):
        if ws < cutin_wind_speed:
            power_data[i] = 0.0
        elif ws < rated_wind_speed:
            # Quadratic interpolation from cutin to rated
            power_data[i] = rated_power * ((ws - cutin_wind_speed) / (rated_wind_speed - cutin_wind_speed))**2
        elif ws <= cutout_wind_speed:
            # Plateau at rated power
            power_data[i] = rated_power
        else:
            # Beyond cutout
            power_data[i] = 0.0
    
    return power_wind_speeds, power_data


def read_single_turbine_type(turbine_windio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a single turbine type specification from windIO data.
    
    Args:
        turbine_windio: windIO turbine dictionary
        
    Returns:
        FLORIS turbine specification dictionary
    """
    turbine_floris = {}
    
    turbine_floris["TSR"] = turbine_windio.get("TSR", 7.0) 
    turbine_floris["name"] = turbine_windio["name"]
    turbine_floris["hub_height"] = turbine_windio["hub_height"]
    turbine_floris["rotor_diameter"] = turbine_windio["rotor_diameter"]
    
    # -- Performance ------------------------------------------------- #
    performance_windio = turbine_windio["performance"]
    
    power_data = None
    power_wind_speeds = None
    
    # -- Process power curve data ---------------------------------#
    if ("power_curve" in performance_windio.data):
        power_curve_windio = performance_windio["power_curve"]
        power_data = power_curve_windio["power_values"]
        power_wind_speeds = power_curve_windio["power_wind_speeds"]

    # -- Process Cp curve data ----------------------------------- #
    if ("Cp_curve" in performance_windio.data):
        if power_data is not None:
            raise ValueError(
                "Both 'power_curve' and 'Cp_curve' found in " \
                "performance data; only one should be provided"
            )

        cp_curve_windio = performance_windio["Cp_curve"]
        cp_values = cp_curve_windio["Cp_values"]
        cp_wind_speeds = cp_curve_windio["Cp_wind_speeds"]
        
        cp_curve = np.array(cp_values)
        wind_speed = np.array(cp_wind_speeds)
        mult_factor = 0.5 * 1.225 * (np.pi * (turbine_floris["rotor_diameter"]/2)**2 * wind_speed**3)
        
        power_data = cp_curve * mult_factor
        power_wind_speeds = cp_wind_speeds
        
    # synthetic power curve
    req =  ['rated_power', 'rated_wind_speed', 'cutin_wind_speed', 'cutout_wind_speed']
    
    if all (k in performance_windio.data for k in req):
        if power_data is not None:
            raise ValueError(
                "Both 'power_curve'/'Cp_curve' and synthetic power curve parameters found; " \
                "only one should be provided"
            )
        
        rated_power = performance_windio['rated_power']
        rated_wind_speed = performance_windio['rated_wind_speed']
        cutin_wind_speed = performance_windio['cutin_wind_speed']
        cutout_wind_speed = performance_windio['cutout_wind_speed']
        
        # Create generic power curve
        power_wind_speeds, power_data = create_generic_power_curve(
            rated_power, rated_wind_speed, cutin_wind_speed, cutout_wind_speed
        )
            
    # -- Thrust coefficient data --------------------------------- #
    thrust_data = None
    thrust_wind_speeds = None
    
    if "Ct_curve" in performance_windio.data:
        ct_curve_windio = performance_windio["Ct_curve"]
        thrust_data = ct_curve_windio["Ct_values"]
        thrust_wind_speeds = ct_curve_windio["Ct_wind_speeds"]

    # -- Validate presence of required curves -------------------- #
    if power_data is None:
        raise KeyError("Missing required 'power_curve' or 'Cp_curve' in performance data")  
        
    if thrust_data is None:
        raise KeyError("Missing required 'Ct_curve' in performance data")
    
    # Retrieve generator efficiency if provided
    turbine_floris["generator_efficiency"] = performance_windio.get("efficiency", 1.0)

    # -- Enforce common wind speeds for power and thrust curves ---------- #
    power_data = np.array(power_data)
    thrust_data = np.array(thrust_data)
    power_wind_speeds = np.array(power_wind_speeds)
    thrust_wind_speeds = np.array(thrust_wind_speeds)
    
    if not np.isclose(power_wind_speeds, thrust_wind_speeds).all():
        wind_speeds = np.union1d(power_wind_speeds, thrust_wind_speeds)

        power_data = np.interp(wind_speeds, 
                                 power_wind_speeds, 
                                 power_data, 
                                 left=0.0, right=0.0)
        thrust_data = np.interp(wind_speeds, 
                                 thrust_wind_speeds, 
                                 thrust_data, 
                                 left=0.0, right=0.0)

    else:
        power_data = power_data
        thrust_data = thrust_data

    # -- Power data stored as W, convert to kW ------------------------------- #
    power_data = power_data / 1e3  # W to kW

    # -- Store final curves in FLORIS format --------------------------------- #
    
    # Prepare data dict for build_cosine_loss_turbine_dict
    turbine_data_dict = {
        "power": power_data.tolist(),
        "thrust_coefficient": thrust_data.tolist(),
        "wind_speed": power_wind_speeds.tolist()
    }
    
    return build_cosine_loss_turbine_dict(
        turbine_data_dict,
        turbine_floris["name"],
        generator_efficiency=turbine_floris.get("generator_efficiency", 1.0),
        hub_height=turbine_floris["hub_height"],
        rotor_diameter=turbine_floris["rotor_diameter"],
        TSR=turbine_floris["TSR"]
    )


def process_turbine_types(farm_floris: Dict[str, Any]) -> None:
    """
    Process turbine types from temporary farm dictionary format.
    
    Args:
        farm_floris: Farm dictionary with turbine_types and _turbine_types_map keys
    """
    
    if "turbine_types" not in farm_floris:
        raise KeyError("Missing 'turbine_types' in farm_floris for turbine types")
    
    if "_turbine_types_map" not in farm_floris:
        raise KeyError("Missing '_turbine_types_map' in farm_floris for turbine type mapping")
    

def read_wind_farm(windio_dict: Dict[str, Any], logger) -> Dict[str, Any]:
    """
    Extract wind farm layout and turbine information from windIO data.
    
    Args:
        windio_dict: Validated windIO dictionary
        logger: Logger instance for warnings
        
    Returns:
        Dictionary with farm layout information
         - layout_x: List of x coordinates
         - layout_y: List of y coordinates
         - turbine_type: List of turbine specifications

        
    Note:
        _xxxxx keys are used for temporary storage and cannot be directly fed into FLORIS
    """
    farm_floris = {}
    farm_floris['_metadata'] = {}

    # -- Wind Farm ------------------------------------------------------- #
    with TrackedDict(windio_dict) as wind_farm_windio:
        # Unmapped variables : electrical_substations, electrical_collection_array
        #                      foundations, O_&_M
        name = wind_farm_windio["name"]
        
        # -- Layouts ----------------------------------------------------- #
        layouts_windio = wind_farm_windio["layouts"]

        if (isinstance(layouts_windio, list)):
            if (len(layouts_windio) > 1):
                logger.warning("Multiple layouts found, using the first layout only")
            layout_windio = TrackedDict.from_list(layouts_windio, context=wind_farm_windio.context + ".layouts")[0]
        else:
            layout_windio = wind_farm_windio["layouts"]

        # -- Coordinates --------------------------------------------- #
        coordinates_windio = layout_windio["coordinates"]
        # Unmapped variables : crs

        farm_floris['layout_x'] = coordinates_windio["x"]
        farm_floris['layout_y'] = coordinates_windio["y"]

        layout_z = coordinates_windio.get("z", None)

        dft_turbine_type = np.zeros(len(farm_floris["layout_x"]), dtype=int).tolist()
        turbine_types_map = layout_windio.get("turbine_types", dft_turbine_type)
        
        # Close the layout_windio TrackedDict
        layout_windio.close()
            
        turbine_type_is_unique = (len(set(turbine_types_map)) > 1)

        # Single-turbine type case
        if ('turbines' in wind_farm_windio.data):
            if ("turbine_types" in wind_farm_windio.data):
                raise ValueError("Both turbines and turbine_types defined in wind_farm, inconsistent data")
            
            if (turbine_type_is_unique):
                raise ValueError("turbines defined in wind_farm but multiple turbine types found, inconsistent data")

            turbine_types = wind_farm_windio["turbines"]
            turbine_types['_context'] = wind_farm_windio.context + ".turbines" 
            turbine_types = {0: turbine_types}
            
        # Multi-turbine type case
        elif ('turbine_types' in wind_farm_windio.data):
            turbine_types = wind_farm_windio["turbine_types"]
            if isinstance(turbine_types, list):
                turbine_types = {i_t: t for i_t, t in enumerate(turbine_types)} 
            for i_t, t in turbine_types.items():
                t['_context'] = wind_farm_windio.context + f".turbine_types[{i_t}]"
            
        # Error if neither turbines nor turbine_types defined
        else:
            raise KeyError("Missing required 'turbines' or 'turbine_types' in wind_farm")

    # -- Process individual turbine types ------------------------------------ #
        for idx, turbine_type in turbine_types.items():
            turbine_types[idx] = read_single_turbine_type(turbine_type)
        
        # -- Map turbine types to FLORIS format ------------------------------ #
        farm_floris["turbine_type"] = [None] * len(farm_floris["layout_x"])
        for idx in range(len(farm_floris["layout_x"])):
            turbine_type_idx = turbine_types_map[idx] # Index of the turbine type
            farm_floris["turbine_type"][idx] = dict(turbine_types[turbine_type_idx])
        
        # -- Make all turbine types unique if needed ----------------------------- #
        # This is a workaround for FLORIS which requires unique turbine definitions
        # for each turbine, even if they are identical. This should be removed
        # once FLORIS supports dictionary-based turbine libraries.
        unique_types = set(turbine_types_map)
        if len(unique_types) != len(farm_floris["turbine_type"]):
            logger.warning("Duplicate turbine types found in layout, ensuring unique turbine definitions for each turbine")
            for idx in range(len(farm_floris["layout_x"])):
                farm_floris["turbine_type"][idx]["turbine_type"] += f"_{idx}"
                
        # -- Set turbine heights if provided ------------------------------------- #
        if (layout_z is not None):
            print('WARNING: Direct altitude is not supported in FLORIS, adjusting hub heights accordingly.')
            for idx in range(len(farm_floris["layout_x"])):
                farm_floris["turbine_type"][idx]["hub_height"] = layout_z[idx] + farm_floris["turbine_type"][idx]["hub_height"]

        return farm_floris

def read_all_turbines_data(windio_dict: Dict[str, Any]) -> List[dict]:
    """
    Extract turbine definition information from windIO data. Returns an ordered
    list of turbine definitions.
    
    Args:
        windio_dict: Validated windIO dictionary
        
    Returns:
        List of turbine specifications
        
    Raises:
        ValueError: If critical turbine fields are missing
    """
    turbine_specs_list = []
    
    # -- Wind Farm ------------------------------------------------------- #
    wind_farm_windio = TrackedDict(windio_dict["wind_farm"], 'wind_farm')
    # Unmapped variables: electrical_substations, electrical_collection_array,
    #                    foundations, O_&_M, layouts (handled separately)
            
    # Determine turbines source
    turbines_model_list = None
    
    # CASE 1: turbine_types defined in wind_farm
    if "turbine_types" in wind_farm_windio.data:
        turbines_model_list = wind_farm_windio["turbine_types"]
        if not isinstance(turbines_model_list, list) or len(turbines_model_list) == 0:
            raise ValueError("turbine_types must be a non-empty list")
    
    # CASE 2: turbines defined in wind_farm (single type for whole farm)
    elif "turbines" in wind_farm_windio.data:
        turbine_data = wind_farm_windio["turbines"]
        if turbine_data is not None:
            turbines_model_list = [turbine_data]
    
    if turbines_model_list is None:
        raise KeyError("Missing required 'turbines' or 'turbine_types' section in wind_farm")
    
    # Extract each turbine
    for idx, turbine_data in enumerate(turbines_model_list):
        turbine_specs = read_single_turbine_type(turbine_data)
        turbine_specs_list.append(turbine_specs)
    
    # Close the TrackedDict
    wind_farm_windio.close()
    
    return turbine_specs_list


def map_indices_to_turbines(turbine_specs_list: List[dict], turbine_types: List[int]) -> List[dict]:
    """
    Map turbine type indices to turbine specifications.
    
    Args:
        turbine_specs_list: List of turbine specifications
        turbine_types: List of turbine type indices
        
    Returns:
        List of turbine specifications mapped to the indices
    """
    
    turbine_type = []
    for idx in turbine_types:
        if not isinstance(idx, int) or idx < 0 or idx >= len(turbine_specs_list):
            raise ValueError(f"Invalid turbine type index {idx} in layout")
        turbine_type.append(turbine_specs_list[idx])
    
    return turbine_type
