
from pathlib import Path
from typing import List

import numpy as np
import pytest
from attrs import define, field

from floris.type_dec import (
    convert_to_path,
    floris_array_converter,
    floris_numeric_dict_converter,
    FromDictMixin,
    iter_validator,
)


@define
class AttrsDemoClass(FromDictMixin):
    w: int
    x: int = field(converter=int)
    y: float = field(converter=float, default=2.1)
    z: str = field(converter=str, default="z")
    non_initd: float = field(init=False)

    def __attrs_post_init__(self):
        self.non_initd = 1.1

    liststr: List[str] = field(
        factory=lambda:["qwerty", "asdf"],
        validator=iter_validator(list, str)
    )
    array: np.ndarray = field(
        factory=lambda:[1.0, 2.0],
        converter=floris_array_converter,
        # validator=iter_validator(np.ndarray, floris_float_type)
    )


def test_as_dict():
    # Non-initialized attributes should not be exported
    cls = AttrsDemoClass(w=0, x=1, liststr=["a", "b"])
    exported_dict = cls.as_dict()
    assert "non_initd" not in exported_dict


def test_FromDictMixin_defaults():
    # Test that the defaults set in the class definition are actually used
    inputs = {"w": 0, "x": 1}
    cls = AttrsDemoClass.from_dict(inputs)
    defaults = {a.name: a.default for a in AttrsDemoClass.__attrs_attrs__ if a.default}
    assert cls.y == defaults["y"]
    assert cls.z == defaults["z"]
    np.testing.assert_array_equal(cls.liststr, defaults["liststr"].factory())
    np.testing.assert_array_equal(cls.array, defaults["array"].factory())

    # Test that defaults can be overwritten
    inputs = {"w": 0, "x": 1, "y": 4.5}
    cls = AttrsDemoClass.from_dict(inputs)
    defaults = {a.name: a.default for a in AttrsDemoClass.__attrs_attrs__ if a.default}
    assert cls.y != defaults["y"]


def test_FromDictMixin_custom():

    inputs = {
        "w": 0,
        "x": 1,
        "y": 2.3,
        "z": "asdf",
        "liststr": ["a", "b"],
        "array": np.array([[1,2,3], [4,5,6]])
    }

    # Check that custom inputs are accepted
    AttrsDemoClass.from_dict(inputs)

    # Ensure extraneous inputs are not applied to the class
    inputs2 = {**inputs, "extra": [3, 4, 5.5]}
    with pytest.raises(AttributeError):
        AttrsDemoClass.from_dict(inputs2)

    # Test that missing required inputs raises an error
    inputs = {}
    with pytest.raises(AttributeError):
        AttrsDemoClass.from_dict(inputs)


def test_iter_validator():

    # Check the correct values work
    _ = AttrsDemoClass(w=0, x=1, liststr=["a", "b"])

    # Check wrong member type
    with pytest.raises(TypeError):
        AttrsDemoClass(w=0, x=1, liststr=[4.3, 1])

    # Check mixed member types
    with pytest.raises(TypeError):
        AttrsDemoClass(w=0, x=1, liststr=[4.3, "1"])

    # Check wrong iterable type
    with pytest.raises(TypeError):
        AttrsDemoClass(w=0, x=1, liststr=("a", "b"))


def test_array_converter():
    array_input = [[1, 2, 3], [4.5, 6.3, 2.2]]
    test_array = np.array(array_input)

    # Test conversion on initialization
    cls = AttrsDemoClass(w=0, x=1, array=array_input)
    np.testing.assert_allclose(test_array, cls.array)

    # Test conversion on reset
    cls.array = array_input
    np.testing.assert_allclose(test_array, cls.array)

    # Test that a non-iterable item like a scalar number fails
    with pytest.raises(TypeError):
        cls.array = 1


def test_numeric_dict_converter():
    """
    This function converts data in a dictionary to a numeric type.
    If it can't convert the data, it will raise a TypeError.
    It should support scalar, list, and numpy array types
    for values in the dictionary.
    """
    test_dict = {
        "scalar_string": "1",
        "scalar_int": 1,
        "scalar_float": 1.0,
        "list_string": ["1", "2", "3"],
        "list_int": [1, 2, 3],
        "list_float": [1.0, 2.0, 3.0],
        "array_string": np.array(["1", "2", "3"]),
        "array_int": np.array([1, 2, 3]),
        "array_float": np.array([1.0, 2.0, 3.0]),
    }
    numeric_dict = floris_numeric_dict_converter(test_dict)
    assert numeric_dict["scalar_string"] == 1
    assert numeric_dict["scalar_int"] == 1
    assert numeric_dict["scalar_float"] == 1.0
    np.testing.assert_allclose(numeric_dict["list_string"], [1, 2, 3])
    np.testing.assert_allclose(numeric_dict["list_int"], [1, 2, 3])
    np.testing.assert_allclose(numeric_dict["list_float"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(numeric_dict["array_string"], [1, 2, 3])
    np.testing.assert_allclose(numeric_dict["array_int"], [1, 2, 3])
    np.testing.assert_allclose(numeric_dict["array_float"], [1.0, 2.0, 3.0])

    test_dict = {"scalar_fail": "a"}
    with pytest.raises(TypeError):
        floris_numeric_dict_converter(test_dict)
    test_dict = {"list_fail": ["a", "2", "3"]}
    with pytest.raises(TypeError):
        floris_numeric_dict_converter(test_dict)
    test_dict = {"array_fail": np.array(["a", "2", "3"])}
    with pytest.raises(TypeError):
        floris_numeric_dict_converter(test_dict)

def test_convert_to_path():
    str_input = Path(__file__)
    expected_path = str_input.resolve()

    # Test that a string works
    test_str_input = convert_to_path(str_input)
    assert test_str_input == expected_path

    # Test that a pathlib.Path works
    path_input = Path(str_input)
    test_path_input = convert_to_path(path_input)
    assert test_path_input == expected_path

    # Test that both of those inputs are the same
    # NOTE These first three asserts tests the relative path search
    assert test_str_input == test_path_input

    # Test absolute path
    abs_path = expected_path
    test_abs_path = convert_to_path(abs_path)
    assert test_abs_path == expected_path

    # Test a file
    file_input = Path(__file__)
    test_file = convert_to_path(file_input)
    assert test_file == file_input

    # Test that a non-existent folder fails, now that the conversion has a multi-pronged search
    str_input = str(Path(__file__).parent / "bad_path")
    with pytest.raises(FileExistsError):
        convert_to_path(str_input)

    # Test that invalid data types fail
    with pytest.raises(TypeError):
        convert_to_path(1)

    with pytest.raises(TypeError):
        convert_to_path(1.2)

    with pytest.raises(TypeError):
        convert_to_path({"one": 1})

    with pytest.raises(TypeError):
        convert_to_path(["a", 1])
