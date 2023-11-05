from pyflyt_dogfight.environments.multiagent.dogfight_parallel_env import DogfightEnv


env = DogfightEnv(render=True)
observations, infos = env.reset()

while env.agents:

    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    if any(env.terminations.values()) or any(env.truncations.values()):
        print(f'{env.terminations=}- {env.truncations=}')
        break

env.close()


