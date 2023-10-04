"""
Simulator for the Moab plate+ball balancing device.
"""
__author__ = "Mike Estee"
__copyright__ = "Copyright 2021, Microsoft Corp."

# We need to disable a check because the typeshed stubs for jinja are incomplete.
# pyright: strict, reportUnknownMemberType=false

import logging

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict
from model import MoabModel, clamp
from pyrr import matrix33, vector

from platotk.serialize import check_and_transform

# from bonsai_common import SimulatorSession, Schema
# from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface
# from microsoft_bonsai_api.simulator.client import BonsaiClientConfig

log = logging.getLogger(__name__)


class MoabSim(Env):
    def __init__(self, env_config):
        """Define the observation and action spaces."""
        self.observation_space = Dict(
            {
                # Ball X,Y position
                "ball_x": Box(low=-float("inf"), high=float("inf")),
                "ball_y": Box(low=-float("inf"), high=float("inf")),
                # Ball X,Y velocity
                "ball_vel_x": Box(low=-6.0, high=6.0),
                "ball_vel_y": Box(low=-6.0, high=6.0),
            }
        )
        self.action_space = Dict(
            {
                # Range -1 to 1 is a scaled value that represents
                # the full plate rotation range supported by the hardware.
                "input_pitch": Box(low=-1.0, high=1.0),  # rotate about x-axis
                "input_roll": Box(low=-1.0, high=1.0),  # rotate about y-axis
            }
        )
        self.model = MoabModel()
        self.reset()

    def _get_obs(self):
        """Get the observable state."""
        full_state = self.model.state()
        observable_state = {
            "ball_x": full_state["ball_x"],
            "ball_y": full_state["ball_y"],
            "ball_vel_x": full_state["ball_vel_x"],
            "ball_vel_y": full_state["ball_vel_y"],
        }
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
        self.model.roll = action["input_roll"]
        self.model.pitch = action["input_pitch"]

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
        distance_from_origin = np.sqrt(state["ball_x"] ** 2 + state["ball_y"] ** 2)
        max_distance = self.model.plate_radius
        return float(1 - distance_from_origin / max_distance)
