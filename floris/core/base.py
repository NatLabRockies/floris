
from abc import abstractmethod
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
)

from attrs import (
    Attribute,
    define,
    field,
    fields,
    setters,
)

from floris.logging_manager import LoggingManager
from floris.type_dec import FromDictMixin


"""
Defines the BaseClass parent class for all models to be based upon.
"""


class State(Enum):
    UNINITIALIZED = 0
    INITIALIZED = 1
    USED = 2


@define
class BaseClass(FromDictMixin):
    """
    BaseClass object class. This class does the logging and MixIn class inheritance.
    """

    # Initialize `state` and ensure it is treated as an attribute rather than a constant parameter.
    # See https://www.attrs.org/en/stable/api-attr.html#attr.ib
    state = field(init=False, default=State.UNINITIALIZED)
    _logging_manager: LoggingManager = field(init=False, default=LoggingManager())

    @property
    def logger(self):
        """Returns the logger manager object."""
        return self._logging_manager.logger

@define
class BaseModel(BaseClass):
    """
    BaseModel is the generic class for any wake models. It defines the API required to
    create a valid model.
    """

    # This is a numerical epsilon to prevent divide by zeros
    NUM_EPS: Final[float] = field(init=False, default=0.001, on_setattr=setters.frozen)

    @abstractmethod
    def prepare_function() -> dict:
        raise NotImplementedError("BaseModel.prepare_function")

    @abstractmethod
    def function() -> None:
        raise NotImplementedError("BaseModel.function")

@define
class BaseLibrary(BaseClass):
    """
    Base class that writes the name and module of the class into the attrs dictionary.
    """
    def as_dict(self) -> dict:
        """Convert instance to dictionary including class information"""
        print("Im here")
        data = super().as_dict()
        data['__class__'] = {
            'module': self.__class__.__module__,
            'name': self.__class__.__name__
        }
        return data
    # __class__: dict = field(init=False)
    # def __attrs_post_init__(self) -> None:
    #     #import ipdb; ipdb.set_trace()
    #     self.__class__ = {
    #         'module': self.__class__.__module__,
    #         'name': self.__class__.__name__
    #     }
