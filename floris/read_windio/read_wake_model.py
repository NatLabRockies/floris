"""
Wake model reading functions for windIO integration.

This module handles extraction of wake model configurations from windIO data.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from .utils import TrackedDict
from floris.utilities import load_yaml

# Load default values from default_inputs.yaml
_DEFAULT_CONFIG = None

def _get_default_config() -> Dict[str, Any]:
    """
    Load default FLORIS configuration from default_inputs.yaml.
    
    Returns:
        Dictionary containing default wake model configurations
    """
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        default_path = Path(__file__).parent.parent / "default_inputs.yaml"
        _DEFAULT_CONFIG = load_yaml(default_path)
    return _DEFAULT_CONFIG

# Mapping from windIO model names to FLORIS model names and parameters
# WindIO model not implemented in FLORIS are mapped to None
# parameter= {windio_name : floris_name}
# "none" refers to no model selected in FLORIS
    
WAKE_MODEL_MAPPING = {
    "wind_deficit_model": {
        "None": {
            "floris_name": "none",
            "parameters": {}
        },
        "Jensen": {
            "floris_name": "jensen",
            "parameters": {
                # wake_expansion_coefficient maps to 'we' in FLORIS Jensen model
                "wake_expansion_coefficient": {
                    "k_a": None,  # Not used in Jensen
                    "k_b": "we",  
                    "free_stream_ti": None,  # Not used in Jensen
                },
                "use_effective_ws": None,  # Not used in FLORIS Jensen
            }
        },
        "Bastankhah2014": {
            "floris_name": "gauss",
            "parameters": {
                # wake_expansion_coefficient parameters
                "wake_expansion_coefficient": {
                    "k_a": "ka",
                    "k_b": "kb",
                    "free_stream_ti": None,  # Not implemented in FLORIS
                },
                "ceps": None,  # Not used in Bastankhah2014
                "use_effective_ws": None,  # Not implemented in FLORIS
            }
        },
        "Bastankhah2016": {
            "floris_name": None,  # Not directly implemented in FLORIS
            "parameters": {
                "wake_expansion_coefficient": {
                    "k_a": "ka",
                    "k_b": "kb",
                    "free_stream_ti": None,
                },
                "ceps": "ceps",  # c_epsilon factor for Bastankhah2016
                "use_effective_ws": None,
            }
        },
        "TurbOPark": {
            "floris_name": "turboparkgauss",
            "parameters": {
                "wake_expansion_coefficient": {
                    "k_a": None,  
                    "k_b": "A",
                    "free_stream_ti": None,
                },
                "use_effective_ws": None,
            }
        },
        "SuperGaussian": {
            "floris_name": None,  # Not implemented in FLORIS
            "parameters": {}
        }
    },
    
    "deflection_model": {
        "None": {
            "floris_name": "none",
            "parameters": {}
        },
        "Jimenez": {
            "floris_name": "jimenez",
            "parameters": {
                "beta": "kd",  # WindIO uses beta, FLORIS uses kd for Jimenez
                "ad": "ad",
                "bd": "bd",
            }
        },
        "Bastankhah2016": {
            "floris_name": "gauss",  # Bastankhah2016 deflection maps to gauss in FLORIS
            "parameters": {
                "beta": "beta",
                "alpha": "alpha",
                "ad": "ad",
                "bd": "bd",
                "dm": "dm",
                "ka": "ka",
                "kb": "kb",
            }
        }
    },
    
    "turbulence_model": {
        "None": {
            "floris_name": "none",
            "parameters": {}
        },
        "STF2005": {
            "floris_name": None,  # Not implemented in FLORIS
            "parameters": {
                "coefficients": None,
            }
        },
        "STF2017": {
            "floris_name": None,  # Not implemented in FLORIS
            "parameters": {
                "coefficients": None,
            }
        },
        "IEC-TI-2019": {
            "floris_name": None,  # Not implemented in FLORIS
            "parameters": {
                "coefficients": None,
            }
        },
        "CrespoHernandez": {
            "floris_name": "crespo_hernandez",
            "parameters": {
                "coefficients": None,  # WindIO uses generic coefficients array
                # FLORIS uses specific named parameters:
                "initial": "initial",
                "constant": "constant",
                "ai": "ai",
                "downstream": "downstream",
            }
        },
        "GCL": {
            "floris_name": None,  # Not implemented in FLORIS
            "parameters": {
                "coefficients": None,
            }
        },
    },
    
    "superposition_model": {
        "None": {
            "floris_name": "none",
            "parameters": {}
        },
        "Linear": {
            "floris_name": "fls",
            "parameters": {}
        },
        "Squared": {
            "floris_name": "sosfs",  # Sum of squares freestream superposition
            "parameters": {}
        },
        "Max": {
            "floris_name": "max",
            "parameters": {}
        },
        "Product": {
            "floris_name": None,  # Not implemented in FLORIS
            "parameters": {}
        }
    }
}

def _extract_model_parameters(
    model_dict: TrackedDict,
    param_mapping: Dict[str, Any],
    floris_name: str,
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generic function to extract model parameters from windIO dict.
    
    Args:
        model_dict: TrackedDict containing model parameters
        param_mapping: Dictionary mapping windIO parameter names to FLORIS names
                      Can contain nested dicts for nested parameters
        floris_name: Name of the FLORIS model for default lookup
        defaults: Optional dictionary of default parameters
        
    Returns:
        Dictionary of extracted parameters with FLORIS names
    """
    parameters = {}
    
    # Extract mapped parameters
    for windio_param, floris_param in param_mapping.items():
        # Handle nested parameter mappings (e.g., wake_expansion_coefficient)
        if isinstance(floris_param, dict):

            # This is a nested structure in windIO
            if windio_param in model_dict:
                nested_dict = model_dict[windio_param]
                for nested_windio_param, nested_floris_param in floris_param.items():
                    
                    if nested_floris_param is None:
                        # Parameter exists in windIO but not in FLORIS
                        if nested_windio_param in nested_dict:
                            _ = nested_dict[nested_windio_param]  # Mark as visited
                            print(f"Warning: windIO parameter '{nested_windio_param}' has no FLORIS equivalent and will be ignored.")
                        continue
                    
                    if nested_windio_param in nested_dict:
                        param_value = nested_dict[nested_windio_param]
                        parameters[nested_floris_param] = param_value

                    elif defaults and nested_floris_param in defaults:
                        parameters[nested_floris_param] = defaults[nested_floris_param]
        else:
            # Simple parameter mapping
            if floris_param is None:
                # Parameter exists in windIO but not in FLORIS
                if windio_param in model_dict:
                    _ = model_dict[windio_param]  # Mark as visited
                    print(f"Warning: windIO parameter '{windio_param}' has no FLORIS equivalent and will be ignored.")
                continue
            
            if windio_param in model_dict:
                param_value = model_dict[windio_param]
                parameters[floris_param] = param_value
                
            elif defaults and floris_param in defaults:
                # Use default if available and not provided in windIO
                parameters[floris_param] = defaults[floris_param]
    
    return parameters

def _extract_generic_wake_model(
    analysis: TrackedDict,
    windio_section_name: str,
    floris_param_key: str,
    required: bool = False
) -> Dict[str, Any]:
    """
    Generic function to extract wake model configuration from windIO analysis section.
    
    Args:
        analysis: TrackedDict containing analysis section
        windio_section_name: Name of the windIO section (e.g., "wind_deficit_model")
        floris_param_key: FLORIS parameter key (e.g., "wake_velocity_parameters")
        required: Whether this model is required (raises error if missing)
        
    Returns:
        Dictionary with model configuration for FLORIS
    """
    if windio_section_name not in analysis:
        if required:
            raise ValueError(f"{windio_section_name} section missing in analysis")
        return {}
    
    model_dict = analysis[windio_section_name]
    model_name = model_dict["name"]
    
    # Look up FLORIS model name
    if model_name not in WAKE_MODEL_MAPPING[windio_section_name]:
        raise ValueError(
            f"{windio_section_name.replace('_', ' ').title()} '{model_name}' not found in mapping."
        )
    
    model_info = WAKE_MODEL_MAPPING[windio_section_name][model_name]
    floris_name = model_info["floris_name"]
    
    if floris_name is None:
        raise ValueError(
            f"{windio_section_name.replace('_', ' ').title()} '{model_name}' is not implemented in FLORIS."
        )
    
    # Get default parameters for this model
    defaults = _get_default_config().get("wake", {}).get(floris_param_key, {}).get(floris_name, {})
    
    # Extract parameters using the mapping
    parameters = _extract_model_parameters(
        model_dict,
        model_info["parameters"],
        floris_name,
        defaults
    )
    
    return {floris_param_key: {floris_name: parameters}}


def read_wake_model(windio_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract wake model configuration from windIO data.
    
    Args:
        windio_dict: Validated windIO dictionary containing attrs.analysis section
        
    Returns:
        Dictionary with wake model configuration compatible with FLORIS format
        
    Note:
        Maps windIO wake model specifications to FLORIS wake model parameters.
        WindIO models are in attributes.analysis section.
        Returns empty dict if no analysis section found.
    """

    wake_floris = {}

    with TrackedDict(windio_dict) as attrs:
        attrs.untrack('model_outputs_specification')

        flow_model = attrs.get('flow_model', None)
        if flow_model:
            soft = flow_model.get('name', 'floris').lower()
            if soft != 'floris':
                print(
                    f"WindIO flow_model.name is '{soft}', but expected 'floris'. "
                    "Proceeding to read attrs regardless."
                )

        if "analysis" not in attrs:
            raise ValueError("attrs.analysis section missing in windIO data")

        analysis = attrs["analysis"]
        model_strings = {}
        
        # Extract velocity model (required)
        velocity = _extract_generic_wake_model(
            analysis, "wind_deficit_model", 
            "wake_velocity_parameters", required=True
        )
        wake_floris.update(velocity)
        model_strings["velocity_model"] = list(velocity["wake_velocity_parameters"].keys())[0]
        
        # Extract deflection model (optional)
        deflection = _extract_generic_wake_model(
            analysis, "deflection_model", 
            "wake_deflection_parameters"
        )
        if deflection:
            wake_floris.update(deflection)
            model_strings["deflection_model"] = list(deflection["wake_deflection_parameters"].keys())[0]
        
        # Extract turbulence model (optional)
        turbulence = _extract_generic_wake_model(
            analysis, "turbulence_model", 
            "wake_turbulence_parameters"
        )
        if turbulence:
            wake_floris.update(turbulence)
            model_strings["turbulence_model"] = list(turbulence["wake_turbulence_parameters"].keys())[0]
        
        # Extract superposition/combination model (optional)
        superposition = _extract_superposition_model(analysis)
        if superposition:
            model_strings.update(superposition["model_strings"])
        
        wake_floris["model_strings"] = model_strings

        wake_floris['enable_secondary_steering'] = False
        print('WARNING: Setting enable_secondary_steering to False by default.')

        wake_floris['enable_yaw_added_recovery'] = False
        print('WARNING: Setting enable_yaw_added_recovery to False by default.')

        wake_floris['enable_transverse_velocities'] = False
        print('WARNING: Setting enable_transverse_velocities to False by default.')

        wake_floris['enable_active_wake_mixing'] = False
        print('WARNING: Setting enable_active_wake_mixing to False by default.')
        
    return {"wake": wake_floris}

def _extract_superposition_model(analysis: TrackedDict) -> Dict[str, Any]:
    """
    Extract superposition/combination model from windIO analysis section.
    
    Note: WindIO has separate ws_superposition and ti_superposition,
          but FLORIS only uses combination_model (for velocity deficit).
    """
    if "superposition_model" not in analysis:
        return {}
    
    superposition_model = analysis["superposition_model"]
    
    # Get ws_superposition for FLORIS combination_model
    ws_model_name = superposition_model.get("ws_superposition") if "ws_superposition" in superposition_model else None
    
    # Mark ti_superposition as visited (not used in FLORIS)
    if "ti_superposition" in superposition_model:
        _ = superposition_model["ti_superposition"]
    
    if not ws_model_name:
        return {}
    
    # Look up FLORIS model name
    if ws_model_name not in WAKE_MODEL_MAPPING["superposition_model"]:
        raise ValueError(f"Superposition model '{ws_model_name}' not found in mapping.")
    
    floris_name = WAKE_MODEL_MAPPING["superposition_model"][ws_model_name]["floris_name"]
    
    if floris_name is None:
        raise ValueError(f"Superposition model '{ws_model_name}' is not implemented in FLORIS.")
    
    return {"model_strings": {"combination_model": floris_name}}