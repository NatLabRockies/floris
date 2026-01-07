
import numpy as np
from attrs import define

from floris.core import BaseModel


@define
class SOSFS(BaseModel):
    """
    SOSFS uses sum of squares freestream superposition to combine the
    wake velocity deficits to the base flow field.

    For more information, refer to :cite:`katic_sos_1986`.
    """

    def prepare_function(self) -> dict:
        pass

    def function(self, wake_field: np.ndarray, velocity_field: np.ndarray):
        """
        Combines the base flow field with the velocity deficits
        using sum of squares.

        Args:
            u_field (np.array): The base flow field.
            u_wake (np.array): The wake to apply to the base flow field.

        Returns:
            np.array: The resulting flow field after applying the wake to the
                base.
        """
        return np.hypot(wake_field, velocity_field)
