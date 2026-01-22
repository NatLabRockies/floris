from abc import abstractmethod

import numpy as np
from attrs import define, field

from floris.core import BaseClass


@define
class BaseOperationModel(BaseClass):
    """
    Base class for turbine operation models. All turbine operation models must implement static
    power(), thrust_coefficient(), and axial_induction() methods, which are called by power() and
    thrust_coefficient() through the interface in the turbine.py module.

    Args:
        BaseClass (_type_): _description_

    Raises:
        NotImplementedError: _description_
        NotImplementedError: _description_
    """
    @staticmethod
    @abstractmethod
    def power() -> None:
        raise NotImplementedError("BaseOperationModel.power")

    @staticmethod
    @abstractmethod
    def thrust_coefficient() -> None:
        raise NotImplementedError("BaseOperationModel.thrust_coefficient")

    @staticmethod
    @abstractmethod
    def axial_induction() -> None:
        # TODO: Consider whether we can make a generic axial_induction method
        # based purely on thrust_coefficient so that we don't need to implement
        # axial_induction() in individual operation models.
        raise NotImplementedError("BaseOperationModel.axial_induction")
