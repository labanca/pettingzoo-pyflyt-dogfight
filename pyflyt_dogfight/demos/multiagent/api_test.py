from pettingzoo.test import parallel_api_test
from pyflyt_dogfight.environments.multiagent.dogfight_parallel_env import DogfightEnv

env = DogfightEnv()
parallel_api_test(env, num_cycles=1000)