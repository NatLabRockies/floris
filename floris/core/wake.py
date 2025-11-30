
import attrs
from attrs import define, field

from floris.core import BaseClass, BaseModel
from floris.core.wake_combination import (
    FLS,
    MAX,
    SOSFS,
)
from floris.core.wake_deflection import (
    EmpiricalGaussVelocityDeflection,
    GaussVelocityDeflection,
    JimenezVelocityDeflection,
    NoneVelocityDeflection,
)
from floris.core.wake_turbulence import (
    CrespoHernandez,
    NoneWakeTurbulence,
    WakeInducedMixing,
)
from floris.core.wake_velocity import (
    CumulativeGaussCurlVelocityDeficit,
    EmpiricalGaussVelocityDeficit,
    GaussVelocityDeficit,
    JensenVelocityDeficit,
    NoneVelocityDeficit,
    TurboparkgaussVelocityDeficit,
    TurbOParkVelocityDeficit,
)


MODEL_MAP = {
    "combination_model": {
        "fls": FLS,
        "max": MAX,
        "sosfs": SOSFS
    },
    "deflection_model": {
        "jimenez": JimenezVelocityDeflection,
        "gauss": GaussVelocityDeflection,
        "none": NoneVelocityDeflection,
        "empirical_gauss": EmpiricalGaussVelocityDeflection
    },
    "turbulence_model": {
        "none": NoneWakeTurbulence,
        "crespo_hernandez": CrespoHernandez,
        "wake_induced_mixing": WakeInducedMixing
    },
    "velocity_model": {
        "none": NoneVelocityDeficit,
        "cc": CumulativeGaussCurlVelocityDeficit,
        "gauss": GaussVelocityDeficit,
        "jensen": JensenVelocityDeficit,
        "turbopark": TurbOParkVelocityDeficit,
        "empirical_gauss": EmpiricalGaussVelocityDeficit,
        "turboparkgauss": TurboparkgaussVelocityDeficit,
    },
}


@define
class WakeModelManager(BaseClass):
    """
    WakeModelManager is a container class for the wake velocity, deflection,
    turbulence, and combination models.

    Args:
        wake (:obj:`dict`): The wake's properties input dictionary
            - velocity_model (str): The name of the velocity model to be instantiated.
            - turbulence_model (str): The name of the turbulence model to be instantiated.
            - deflection_model (str): The name of the deflection model to be instantiated.
            - combination_model (str): The name of the combination model to be instantiated.
    """
    model_strings: dict = field(converter=dict)
    enable_secondary_steering: bool = field(converter=bool)
    enable_yaw_added_recovery: bool = field(converter=bool)
    enable_active_wake_mixing: bool = field(converter=bool)
    enable_transverse_velocities: bool = field(converter=bool)

    wake_deflection_parameters: dict = field(converter=dict)
    wake_turbulence_parameters: dict = field(converter=dict)
    wake_velocity_parameters: dict = field(converter=dict, factory=dict)

    combination_model: BaseModel = field(init=False)
    deflection_model: BaseModel = field(init=False)
    turbulence_model: BaseModel = field(init=False)
    velocity_model: BaseModel = field(init=False)

    def __attrs_post_init__(self) -> None:
        velocity_model_string = self.model_strings["velocity_model"].lower()
        model: BaseModel = MODEL_MAP["velocity_model"][velocity_model_string]
        if velocity_model_string == "none":
            model_parameters = None
        else:
            model_parameters = self.wake_velocity_parameters[velocity_model_string]
        if model_parameters is None:
            # Use model defaults
            self.velocity_model = model()
        else:
            self.velocity_model = model.from_dict(model_parameters)

        deflection_model_string = self.model_strings["deflection_model"].lower()
        model: BaseModel = MODEL_MAP["deflection_model"][deflection_model_string]
        if deflection_model_string == "none":
            model_parameters = None
        else:
            model_parameters = self.wake_deflection_parameters[deflection_model_string]
        if model_parameters is None:
            self.deflection_model = model()
        else:
            self.deflection_model = model.from_dict(model_parameters)

        turbulence_model_string = self.model_strings["turbulence_model"].lower()
        model: BaseModel = MODEL_MAP["turbulence_model"][turbulence_model_string]
        if turbulence_model_string == "none":
            model_parameters = None
        else:
            model_parameters = self.wake_turbulence_parameters[turbulence_model_string]
        if model_parameters is None:
            self.turbulence_model = model()
        else:
            self.turbulence_model = model.from_dict(model_parameters)

        combination_model_string = self.model_strings["combination_model"].lower()
        model: BaseModel = MODEL_MAP["combination_model"][combination_model_string]
        self.combination_model = model()

    @model_strings.validator
    def validate_model_strings(self, instance: attrs.Attribute, value: dict) -> None:
        required_strings = [
            "velocity_model",
            "deflection_model",
            "combination_model",
            "turbulence_model"
        ]
        # Check that all required strings are given
        for s in required_strings:
            if s not in value.keys():
                raise KeyError(f"Wake: '{s}' not provided in the input but it is required.")

        # Check that no other strings are given
        for k in value.keys():
            if k not in required_strings:
                raise KeyError((
                    f"Wake: '{k}' was given as input but it is not a valid option."
                    f"Required inputs are: {', '.join(required_strings)}"
                ))

    @property
    def deflection_function(self):
        return self.deflection_model.function

    @property
    def velocity_function(self):
        return self.velocity_model.function

    @property
    def turbulence_function(self):
        return self.turbulence_model.function

    @property
    def combination_function(self):
        return self.combination_model.function
