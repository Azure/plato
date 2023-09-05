"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R """

import math
import random

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete


def normalize_angle(x):
    """
    convert an angle to [-pi, pi]
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


class CartPole(Env):
    """
    Model for the dynamics of an inverted pendulum
    """

    def __init__(self, env_config):
        # Constants not exposed for changing for now
        self.GRAVITY = 9.8  # a classic...
        self.FORCE_MAG = 1.0
        self.STEP_DURATION = 0.02  # seconds between state updates (20ms)
        self.TRACK_WIDTH = 2.0  # m
        self.FORCE_NOISE = 0.02  # % of FORCE_MAG
        self.x_threshold = self.TRACK_WIDTH / 2
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        self.cart_mass = 0.31
        self.pole_mass = 0.055
        self.pole_length = 0.4
        self.initial_cart_position = 0
        self.initial_cart_velocity = 0
        self.initial_pole_angle = 0
        self.initial_angular_velocity = 0
        self.target_pole_position = 0

        # Define the observation space
        self.observation_space = Dict(
            {
                "cart_position": Box(
                    low=-float("inf"), high=float("inf")
                ),  # (m). 0 is the center of the track
                "cart_velocity": Box(low=-float("inf"), high=float("inf")),  # (m/s)
                "pole_angle": Box(
                    low=-float("inf"), high=float("inf")
                ),  # (rad). 0 is vertical.
                "pole_angular_velocity": Box(
                    low=-float("inf"), high=float("inf")
                ),  # (rad/s)
            }
        )

        # Define the action space
        self.action_space = Discrete(3, start=-1)

        if env_config["direction"] == "left":
            self.initial_cart_position = random.uniform(0, self.x_threshold)

        elif env_config["direction"] == "right":
            self.initial_cart_position = random.uniform(-self.x_threshold, 0)
        else:
            raise ValueError(
                f"Invalid direction {env_config['direction']}. \
                    Must be 'left' or 'right'."
            )
        self.reset()

    def _get_obs(self):
        """Get the observable state."""
        return {
            "cart_position": np.array(
                [self._cart_position]
            ),  # (m). 0 is the center of the track
            "cart_velocity": np.array([self._cart_velocity]),  # (m/s)
            "pole_angle": np.array([self._pole_angle]),  # (rad). 0 is vertical.
            "pole_angular_velocity": np.array([self._pole_angular_velocity]),  # (rad/s)
        }

    def _get_info(self):
        """Get additional info not needed by the agent's decision."""
        return {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self._cart_mass = self.cart_mass  # (kg)
        self._pole_mass = self.pole_mass  # (kg)
        self._pole_length = self.pole_length  # (m)
        self._cart_position = self.initial_cart_position  # (m)
        self._cart_velocity = self.initial_cart_velocity  # (m/s)
        self._pole_angle = normalize_angle(self.initial_pole_angle)  # (rad)
        self._pole_angular_velocity = self.initial_angular_velocity  # (rad/s)
        self._target_pole_position = self.target_pole_position  # (m)
        self._update_pole_center_state()
        self.iter = 0

        return self._get_obs(), self._get_info()

    def _update_pole_center_state(self):
        """
        Update _pole_center_position and _pole_center_velocity.
        """
        # Use the pole center, not the cart center, for tracking
        # pole center velocity.
        pole_half_length = self._pole_length / 2
        self._pole_center_position = (
            self._cart_position + math.sin(self._pole_angle) * pole_half_length
        )
        self._pole_center_velocity = (
            self._cart_velocity
            + math.sin(self._pole_angular_velocity) * pole_half_length
        )

    def step(self, action):
        """
        Move the state of the cartpole simulation forward one time unit.

        Args:
            action: a dict with a key 'command', with value between -1 and 1.
        """
        # We are expecting the input command to be -1 or 1,
        # but we'll support a continuous action space.
        # Add a small amount of random noise to the force so
        # the policy can't succeed by simply applying zero
        # force each time.
        force = self.FORCE_MAG * (action + random.uniform(-0.02, 0.02))

        # Precompute some helpful quantities
        total_mass = self._cart_mass + self._pole_mass
        pole_half_length = self._pole_length / 2
        pole_mass_length = self._pole_mass * pole_half_length

        cosTheta = math.cos(self._pole_angle)
        sinTheta = math.sin(self._pole_angle)

        temp = (
            force + pole_mass_length * self._pole_angular_velocity**2 * sinTheta
        ) / total_mass
        angularAccel = (self.GRAVITY * sinTheta - cosTheta * temp) / (
            pole_half_length
            * (4.0 / 3.0 - (self._pole_mass * cosTheta**2) / total_mass)
        )
        linearAccel = temp - (pole_mass_length * angularAccel * cosTheta) / total_mass

        self._cart_position = (
            self._cart_position + self.STEP_DURATION * self._cart_velocity
        )
        self._cart_velocity = self._cart_velocity + self.STEP_DURATION * linearAccel

        self._pole_angle = (
            self._pole_angle + self.STEP_DURATION * self._pole_angular_velocity
        )
        self._pole_angle = normalize_angle(self._pole_angle)

        self._pole_angular_velocity = (
            self._pole_angular_velocity + self.STEP_DURATION * angularAccel
        )

        self._update_pole_center_state()

        terminated = bool(
            self._cart_position < -self.x_threshold
            or self._cart_position > self.x_threshold
            or self._pole_angle < -self.theta_threshold_radians
            or self._pole_angle > self.theta_threshold_radians
        )

        self.iter += 1
        truncated = self.iter >= 100

        if not terminated:
            reward = 1.0
        else:
            reward = 0.0

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    @property
    def state(self):
        return {
            "cart_position": self._cart_position,
            "cart_velocity": self._cart_velocity,
            "pole_angle": self._pole_angle,
            "pole_angular_velocity": self._pole_angular_velocity,
            "pole_center_position": self._pole_center_position,
            "pole_center_velocity": self._pole_center_velocity,
            "target_pole_position": self._target_pole_position,
            # also return the constants from the config
            "cart_mass": self._cart_mass,
            "pole_mass": self._pole_mass,
            "pole_length": self._pole_length,
        }
