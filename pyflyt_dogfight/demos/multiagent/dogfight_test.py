from pyflyt_dogfight.environments.multiagent.dogfight_env import DogfightEnv

env = DogfightEnv()
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
