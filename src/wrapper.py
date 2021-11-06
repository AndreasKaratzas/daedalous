
import numpy

from gym import RewardWrapper


class ClipReward(RewardWrapper):
    """Clip reward to [min, max].

    Args:
        RewardWrapper ([type]): [description]
    """
    def __init__(self, env, min_r, max_r):
        super(ClipReward, self).__init__(env)
        self.min_r = min_r
        self.max_r = max_r

    def reward(self, reward):
        return numpy.clip(reward, self.min_r, self.max_r)
