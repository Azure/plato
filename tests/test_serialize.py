import json

import numpy as np
import pytest
from gymnasium import spaces

from platotk.serialize import GymEncoder, check_and_transform

states = (
    ({"float": np.float_(90)}, '{"float": 90.0}'),
    ({"int": np.int_(90)}, '{"int": 90}'),
    ({"bool": np.bool_(1)}, '{"bool": true}'),
    ({"list_float": np.arange(2, dtype=np.float_)}, '{"list_float": [0.0, 1.0]}'),
    ({"list_int": np.arange(2, dtype=np.int_)}, '{"list_int": [0, 1]}'),
)


@pytest.mark.parametrize("state, encoded", states)
def test_numpy_encoder(state, encoded):
    assert json.dumps(state, cls=GymEncoder) == encoded


samples = [
    # Test a valid instance for a Discrete space
    (spaces.Discrete(3), 1),
    # Test an invalid instance for a Discrete space
    (spaces.Discrete(3), 1.9),
    # Test a valid instance for a Box space
    (spaces.Box(0, 1, shape=(1,)), [0.5]),
    # Test an invalid instance for a Box space
    (spaces.Box(0, 3, shape=(1,), dtype=np.int32), 2.0),
    # Test an valid instance for a Dict space
    (
        spaces.Dict({"a": spaces.Discrete(3), "b": spaces.Box(0, 1, shape=(2,))}),
        {"a": 1, "b": [0.5, 0.5]},
    ),
]


@pytest.mark.parametrize("observation_space, observation", samples)
def test_transform(observation_space, observation):
    assert observation_space.contains(
        check_and_transform(observation_space, observation)
    )


def test_check_and_transform_valueerror():
    with pytest.raises(ValueError):
        check_and_transform(spaces.MultiBinary(5), np.array([1, 0, 1, 0]))
