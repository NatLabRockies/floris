
from typing import Any, Dict

import numpy as np
from attrs import define, field

from floris.core import BaseModel


@define
class NoneWakeCombination(BaseModel):
    """
    The None wake turbulence model is a placeholder code that simple ignores
    any combination and returns None.
    """

    def prepare_function(self) -> dict:
        pass

    def function(
        self,
        ambient_TI: float,
        x: np.ndarray,
        x_i: np.ndarray,
        rotor_diameter: float,
        axial_induction: np.ndarray,
    ) -> None:
        """Return None"""
        return None
