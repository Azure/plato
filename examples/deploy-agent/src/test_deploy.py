import numpy as np
import pytest
from gymnasium import spaces
from src.serve import check_and_transform

samples = [
    (spaces.Discrete(3), 1, True),
    (spaces.Discrete(3), 1.9, False),
    (spaces.Box(0, 1, shape=(1,)), np.array([0.5], dtype="float32"), True),
    (spaces.Box(0, 1, shape=(1,)), 0.5, False),
    (
        spaces.Dict({"a": spaces.Discrete(3), "b": spaces.Box(0, 1, shape=(2,))}),
        {"a": 1, "b": np.array([0.5, 0.5], dtype="float32")},
        True,
    ),
]


@pytest.mark.parametrize("observation_space, observation, expected", samples)
def test_check(observation_space, observation, expected):
    assert observation_space.contains(observation) == expected


@pytest.mark.parametrize("observation_space, observation, expected", samples)
def test_transform(observation_space, observation, expected):
    assert observation_space.contains(
        check_and_transform(observation_space, observation)
    )


def check_and_transform_valueerror():
    with pytest.raises(ValueError):
        check_and_transform(spaces.MultiBinary(5), np.array([1, 0, 1, 0]))
