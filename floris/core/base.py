
import importlib
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
    __classinfo__: dict = {"module": "", "name": ""}
    def __attrs_post_init__(self) -> None:
        #import ipdb; ipdb.set_trace()
        self.__classinfo__ = {
            "module": type(self).__module__,
            "name": type(self).__name__
        }

    @staticmethod
    def from_dict(data_dict):
        """Recreate instance from dictionary with class information"""
        if "__classinfo__" not in data_dict:
            raise ValueError(
                "Dictionary does not contain class information. ",
                "Insure inheritance from BaseLibrary."
            )
        data_noinfo = data_dict.copy()
        class_info = data_noinfo.pop("__classinfo__")

        # Import the module and get the class
        module = importlib.import_module(class_info["module"])
        cls = getattr(module, class_info["name"])

        return cls(**data_noinfo)
