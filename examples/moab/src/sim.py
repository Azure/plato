"""
Simulator for the Moab plate+ball balancing device.
"""

import numpy as np
from gymnasium import Env, spaces
from model import MoabModel, clamp
from pyrr import matrix33, vector


def check_and_transform(observation_space, state):
    """
    Check and transform a observation state instance for a given observation space.

    This function takes a state instance and an observation space object that defines
    the valid shape and dtype of the state data for the simulation environment, then
    checks if the state instance is already valid for the observation space, and if
    not, it tries to transform it into a valid form.

    Args:
        observation_space (gym.Space): The observation space of the sim env.
        state (JSON): The state instance to check and transform.

    Returns:
        state: The transformed state instance that is valid for the observation space.

    Raises:
        ValueError: If the state instance cannot be transformed for the observation
        space.
    """
    # Check if the instance is valid for the space
    if isinstance(state, np.ndarray) and observation_space.contains(state):
        # Return the instance as it is
        return state
    # Try to transform the instance based on the space type
    elif isinstance(observation_space, spaces.Box):
        # Convert the instance to a numpy array with the same
        # shape and dtype as the space
        return np.array(state, dtype=observation_space.dtype).reshape(
            observation_space.shape
        )
    elif isinstance(observation_space, spaces.Discrete):
        # Convert the instance to an integer
        return int(state)
    elif isinstance(observation_space, spaces.Dict):
        # Convert the instance to a dictionary with valid values for each
        # key in the space
        return {
            key: check_and_transform(observation_space.spaces[key], state[key])
            for key in observation_space.spaces
        }
    # Add more cases for other types of spaces as needed
    else:
        # Raise an exception if the instance cannot be transformed
        raise ValueError(f"Cannot transform {state} for {observation_space}")


class MoabSim(Env):
    def __init__(self, env_config):
        """Define the observation and action spaces."""
        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(4,)
        )  # (ball_x, ball_y, ball_vel_x, ball_vel_y)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,)
        )  # (input_pitch, input_roll))

        # initialize the model
        self.model = MoabModel()
        self.reset()

    def _get_obs(self):
        """Get the observable state."""
        full_state = self.model.state()
        observable_state = [
            full_state["ball_x"],
            full_state["ball_y"],
            full_state["ball_vel_x"],
            full_state["ball_vel_y"],
        ]
        return check_and_transform(self.observation_space, observable_state)

    def _get_info(self):
        """Get additional info not needed by the agent's decision."""
        return {}

    def _set_velocity_for_speed_and_direction(self, speed: float, direction: float):
        # get the heading
        dx = self.model.target_x - self.model.ball.x
        dy = self.model.target_y - self.model.ball.y

        # direction is meaningless if we're already at the target
        if (dx != 0) or (dy != 0):
            # set the magnitude
            vel = vector.set_length([dx, dy, 0.0], speed)

            # rotate by direction around Z-axis at ball position
            rot = matrix33.create_from_axis_rotation([0.0, 0.0, 1.0], direction)
            vel = matrix33.apply_to_vector(rot, vel)

            # unpack into ball velocity
            self.model.ball_vel.x = vel[0]
            self.model.ball_vel.y = vel[1]
            self.model.ball_vel.z = vel[2]

    def reset(self, *, seed=None, options=None):
        # return to known good state to avoid accidental episode-episode dependencies
        self.model.reset()

        # initial control state. these are all [-1..1] unitless
        self.model.roll = np.random.uniform(low=-0.2, high=0.2)
        self.model.pitch = np.random.uniform(low=-0.2, high=0.2)

        # now we can update the initial plate metrics from the constants and the
        # controls
        self.model.update_plate(plate_reset=True)

        # initial ball state after updating plate
        self.model.set_initial_ball(
            self.model.ball.x,
            self.model.ball.y,
            self.model.ball.z,
        )

        # # velocity set as a speed/direction towards target
        # initial_speed = config.get("initial_speed", None)
        # initial_direction = config.get("initial_direction", None)
        # if initial_speed is not None and initial_direction is not None:
        #     self._set_velocity_for_speed_and_direction(initial_speed,
        #                                               initial_direction)

        # new episode, iteration count reset
        self.iteration_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        # use new syntax or fall back to old parameter names
        self.model.roll = action[0]  # input_roll
        self.model.pitch = action[1]  # input_pitch

        # clamp inputs to legal ranges
        self.model.roll = clamp(self.model.roll, -1.0, 1.0)
        self.model.pitch = clamp(self.model.pitch, -1.0, 1.0)

        self.model.step()
        self.iteration_count += 1

        state = self._get_obs()

        terminated = self.model.halted()  # ball falls off plate
        truncated = self.iteration_count >= 100

        return (
            check_and_transform(self.observation_space, state),
            self.reward(state),
            terminated,
            truncated,
            {},
        )

    def reward(self, state):
        distance_from_origin = np.sqrt(
            state[0] ** 2 + state[1] ** 2
        )  # np.sqrt(<ball_x> ** 2 + <ball_y> ** 2)
        max_distance = self.model.plate_radius
        return float(1 - distance_from_origin / max_distance)
