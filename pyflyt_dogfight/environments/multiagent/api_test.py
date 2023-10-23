from pettingzoo.test import parallel_api_test
from pettingzoo.butterfly import pistonball_v6
from pyflyt_dogfight.environments.multiagent.dogfight_env2 import DogfightEnv
env = DogfightEnv()
parallel_api_test(env, num_cycles=1000)