
from typing import Callable

from attrs import define, field

from floris.core import (
    BaseClass,
    BaseLibrary,
    BaseModel,
)
from floris.core.wake_model import (
    BaseWakeModel,
    CumulativeCurl,
    EmpiricalGauss,
    Gauss,
    JensenJimenez,
    NoneWake,
    TurbOParkGauss,
)
from floris.core.wake_model.wake_combination import (
    fls,
    maximum,
    none_combination,
    sosfs,
)


MODEL_MAP = {
    "none": NoneWake,
    "cc": CumulativeCurl,
    "gauss": Gauss,
    "jensen": JensenJimenez,
    "empirical_gauss": EmpiricalGauss,
    "turboparkgauss": TurbOParkGauss,
}

COMBINATION_MAP = {
    "none": none_combination,
    "fls": fls,
    "max": maximum,
    "sosfs": sosfs,
}

def _wake_model_converter(model, model_parameters):
    # If model is a string, instantiate from MODEL_MAP using model_parameters
    if isinstance(model, str):
        if model == "none":
            return NoneWake()
        elif model not in MODEL_MAP:
            valid_models = list(MODEL_MAP.keys())
            raise ValueError(
                f"Unknown velocity model '{model}'. "
                f"Expected one of {valid_models}."
            )
        else:
            return MODEL_MAP[model](**model_parameters)

    # Handle dict representation of a wake model
    # Use existing parameters on model, unless model_parameters are different
    elif isinstance(model, dict):
        for k, v in model_parameters.items():
            model[k] = v
        return BaseLibrary.from_dict(model)

    # Otherwise, raise an error
    else:
        raise TypeError(
            "model must be a BaseWakeModel subclass (in dict representation), "
            "or a valid velocity-model string."
        )

@define
class WakeModelManager(BaseClass):
    """
    WakeModelManager is a container class for the wake velocity, deflection,
    turbulence, and combination models.

    Args:
        wake (:obj:`dict`): The wake's properties input dictionary
            - velocity_model (str): The name of the velocity model to be instantiated.
            - combination_model (str): The name of the combination model to be instantiated.
    """
    model: str | BaseWakeModel = field()
    parameters: dict = field(converter=dict)
    combination_model: str | Callable = field(default="sosfs")

    def __attrs_post_init__(self) -> None:

        self.model = _wake_model_converter(self.model, self.parameters)

        if isinstance(self.combination_model, str):
            self.combination_model = COMBINATION_MAP[self.combination_model]
        self.model.assign_combination_function(self.combination_model)

    def assign_user_defined_wake_model(self, wake_model: BaseWakeModel):
        self.parameters = {k: v for k, v in wake_model.as_dict().items() if k != "__classinfo__"}
        self.model = wake_model
