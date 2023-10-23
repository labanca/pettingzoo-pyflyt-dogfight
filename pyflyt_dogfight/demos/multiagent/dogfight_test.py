import gymnasium

from pyflyt_dogfight.environments.multiagent.dogfight_env2 import DogfightEnv

env = DogfightEnv(render='human')
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    if any(env.termination.values()) or any(env.truncation.values()):
        print(f'{env.termination=}- {env.truncation=}')
        break
env.close()


