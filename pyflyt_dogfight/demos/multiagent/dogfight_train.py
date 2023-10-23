import datetime
import time
import pathlib as plib

import gymnasium
import yaml
from multiprocessing import freeze_support
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv
from  stable_baselines3.common.env_checker import check_env

from pyflyt_dogfight.environments.multiagent.dogfight_env2 import DogfightEnv
from PyFlyt.core import Aviary

import supersuit as ss
import pettingzoo

def get_timestamp():
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%dT%H%M%S")
    return formatted_datetime


def train():
    buffer_size = 1_000_000
    buffer_size_deug = 128
    batch_size = 256
    n_envs = 1
    total_steps = n_envs * buffer_size
    repeats_per_buffer = 3
    critic_update_multiplier = 1
    actor_update_multiplier = 1
    lr = 0.00007
    discount_factor = 0.95
    max_duration_seconds = 10


    device = get_device(device='cuda')
    #print(device)
    #env = gymnasium.make_vec('DogfightEnv-v0')
    #env = DogfightEnv(max_duration_seconds=max_duration_seconds)
    #env = ss.pettingzoo_env_to_vec_env_v1(env)
    #env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

    env = DogfightEnv()
    #env = ss.pettingzoo_env_to_vec_env_v1(env)
    #env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

    #check_env(env)


    #nn_t = [128, 128, 256, 256, 256, 512, 512, 1024]
    nn_t = [128, 128, 128]
    policy_kwargs = dict(
        normalize_images=False,
        net_arch=dict(pi=nn_t, vf=nn_t)
    )

    algorithm = "PPO"
    study_name = 'dogfight_'+ get_timestamp()
    cwd_folder = plib.Path(__file__).absolute().parent
    output_dir = plib.Path.joinpath(cwd_folder, 'output', study_name)
    models_dir = plib.Path.joinpath(output_dir, "models")
    log_dir = plib.Path.joinpath(output_dir, "logs")





    policy = "MlpPolicy"
    model = PPO(policy=policy, env=env,  verbose=1,
                learning_rate=lr,
                device=device,
                tensorboard_log=str(log_dir),
                policy_kwargs=policy_kwargs
                #gamma=discount_factor,
                #batch_size=batch_size,
                )

    hparam_dict = {
        "env.action_space" : env.action_space.__repr__(),
        "env.angle_representation" : env.angle_representation if hasattr(env, "angle_representation") else None,
        "env.attitude_space" : env.attitude_space.__repr__() if hasattr(env, "attitude_space") else None,
        #"env.auxiliary_space" : env.auxiliary_space.__repr__(),
        # "env.camera_parameters" : env.camera_parameters if hasattr(env, "camera_parameters") else None ,
        # "env.combined_space" : env.combined_space.__repr__(),
        # "env.env_step_ratio" : env.env_step_ratio,
        # "env.flight_dome_size" : env.flight_dome_size,
        # "env.max_steps" : env.max_steps,
        # "env.metadata" : env.metadata,
        # "env.observation_space" : env.observation_space.__repr__(),
        # "env.render_mode" : env.render_mode,
        # "env.render_resolution" : env.render_resolution.__repr__(),
        # "env.reward_range" : env.reward_range.__repr__(),
        # "env.sparse_reward" : env.sparse_reward,
        # "env.spec" : env.spec,
        # "env.start_orn" : env.start_orn.__repr__(),
        # "env.start_pos" : env.start_pos.__repr__(),
        # "env.waypoints" : env.waypoints.__repr__(),
        # "model.action_noise": model.action_noise,
        # "model.action_space": model.action_space.__repr__(),
        # "model.algorithm": model.__class__.__name__,
        # "model.batch_size": model.batch_size,
        # "model.clip_range":model.clip_range,
        # "model.clip_range_vf": model.clip_range_vf.__repr__(),
        # "model.device": model.device,
        # "model.ent_coef": model.ent_coef,
        # "model.ep_info_buffer" : model.ep_info_buffer,
        # "model.ep_success_buffer": model.ep_success_buffer,
        # "model.gae_lambda": model.gae_lambda,
        # "model.gamma": model.gamma,
        # "model.learning rate": model.learning_rate,
        # "model.lr_schedule": model.lr_schedule,
        # "model.max_grad_norm" : model.max_grad_norm,
        # "model.n_epochs": model.n_epochs,
        # "model.n_steps": model.n_steps,
        # "model.normalize_advantage": model.normalize_advantage,
        # "model.num_envs": model.n_envs,
        # "model.num_timesteps": model.num_timesteps,
        # "model.observation_space": model.observation_space.__repr__(),
        # "model.policy": model.policy.__repr__(),
        # "model.policy_aliases": model.policy_aliases,
        # "model.policy_class": model.policy_class,
        # "model.policy_kwargs": model.policy_kwargs,
        # "model.rollout_buffer": model.rollout_buffer.__repr__(),
        # "model.sde_sample_freq": model.sde_sample_freq,
        # "model.seed": model.seed,
        # "model.start_time": model.start_time,
        # "model.study_name": study_name,
        # "model.target_kl": model.target_kl,
        # "model.tensorboard_log": model.tensorboard_log,
        # "model.total_timesteps": model._total_timesteps,
        # "model.use_sde": model.use_sde,
        # "model.vf_coef": model.vf_coef,
        #"Aviary.drone_type_mappings": Aviary.drone_type_mappings if hasattr(env, "drone_type_mappings") else None,
        #"Aviary.drone_type": Aviary.drone_type,
        #"Aviary.drone_options": Aviary.drone_options,
        #"Aviary.num_drones": Aviary.num_drones,
        #"Aviary.drones": Aviary.drones,
        #"Aviary.armed_drones": Aviary.armed_drones,
        "Aviary.print_all_bodies": Aviary.print_all_bodies,
        #"Aviary.physics_hz": Aviary.physics_hz,
        #"Aviary.update_period": Aviary.update_period,
        #"Aviary.physics_steps": Aviary.physics_steps,
        #"Aviary.updates_per_step": Aviary.updates_per_step,
        #"Aviary.aviary_steps": Aviary.aviary_steps,
        #"Aviary.contact_array": Aviary.contact_array,
        #"Aviary.wind_options": Aviary.wind_options,
        #"Aviary.wind_field": Aviary.wind_field,
        #"Aviary.wind_type": Aviary.wind_type,
        #"Aviary.world_scale": Aviary.world_scale,



    }

    if not plib.Path.exists(output_dir.parent):
        plib.Path.mkdir(output_dir.parent)

    if not plib.Path.exists(output_dir):
        plib.Path.mkdir(output_dir)

    if not plib.Path.exists(models_dir):
        plib.Path.mkdir(models_dir)


    yaml.dump(hparam_dict, open(plib.Path.joinpath(output_dir, 'hparam.yaml'), 'w'))

    start_time = time.time()

    model.learn(total_timesteps=total_steps,
                #progress_bar=False,
                #log_interval=1000,
                #reset_num_timesteps=False,
                tb_log_name=model.__class__.__name__
                )

    model.save(plib.Path.joinpath(models_dir, study_name))

    filename = plib.Path.joinpath(output_dir, "results.txt")

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time_seconds))
    with open(filename, "w") as file:
        file.write(f"\"start_time\": {time.strftime('%H:%M:%S', time.localtime(start_time))}\n")
        file.write(f"\"end_time\": {time.strftime('%H:%M:%S', time.localtime(end_time))}\n")
        file.write(f"\"elapsed_time\": {elapsed_time_formatted}\n")
        file.write(f"\"learning_rate\": {lr}\n")
        file.write(f"\"total_timesteps\": {total_steps}\n")
        file.write(f"\"algorithm\": {algorithm}\n")
        file.write(f"\"policy\": {policy}\n")
        file.write(f"\"nnt\": {nn_t}\n")
        file.write(f"\"device\": {model.policy.device}\n")
        file.write(f"\"sparse_reward\": {env.sparse_reward}\n")
        file.write(f"\"max_duration_seconds\": {max_duration_seconds}\n")
        file.write(f"\"policy_kwargs\": {policy_kwargs}\n")



if __name__ == '__main__':
    freeze_support()
    train()
