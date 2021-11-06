
from gym.envs.registration import register

register(
    id='airsim-v6',
    entry_point='airgym.assets:AirSimEnv',
)

register(
    id='multirotor-v6',
    entry_point='airgym.assets:Multirotor',
)
