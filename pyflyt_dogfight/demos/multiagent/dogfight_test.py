from pyflyt_dogfight.environments.multiagent.dogfight_parallel_env import DogfightEnv

env = DogfightEnv(render=True)
observations, infos = env.reset()

while env.agents:

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    if any(env.termination.values()) or any(env.truncation.values()):
        print(f'{env.termination=}- {env.truncation=}')
        break

env.close()


