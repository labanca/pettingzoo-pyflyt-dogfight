from gymnasium.envs.registration import register
from pyflyt_dogfight.environments.multiagent.dogfight_env2 import DogfightEnv

register(
    id='DogfightEnv-v0',
    entry_point='pyflyt_dogfight.environments.multiagent.dogfight_env3:DogfightEnv',
)
