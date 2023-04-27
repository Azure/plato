import numpy as np
import pytest
from gymnasium import spaces
from src.serve import check_and_transform

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
