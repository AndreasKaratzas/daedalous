
import gym
import numpy

from gym import spaces


class AirSimEnv(gym.Env):
    """OpenAI GYM environment template for AirSim.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, image_shape):
        self.observation_space = spaces.Box(0,
                                            255,
                                            shape=image_shape,
                                            dtype=numpy.uint8)
        self.viewer = None

    def __del__(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()
