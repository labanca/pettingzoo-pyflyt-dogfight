import functools
from os import path

import numpy as np
from gymnasium import spaces
from PyFlyt.core import Aviary

# fix numpy buggy cross
np_cross = lambda x, y: np.cross(x, y)

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = DogfightEnv()
    env = parallel_to_aec(env)
    return env


class DogfightEnv(ParallelEnv):
    """Base Dogfighting Environment for the Aggressor model using custom environment API."""
    metadata = {"render_modes": ["human"], "name": "dogfight_parallel_v0"}

    def __init__(
            self,
            spawn_height: float = 15.0,
            flight_dome_size: float = 150.0,
            max_duration_seconds: float = 60.0,
            agent_hz: int = 30,
            damage_per_hit: float = 0.02,
            lethal_distance: float = 15.0,
            lethal_angle_radians: float = 0.1,
            assisted_flight: bool = True,
            render: bool = False,
            human_camera: bool = True,
    ):
        """__init__.

        Args:
            spawn_height (float): spawn_height
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            agent_hz (int): agent_hz
            damage_per_hit (float): damage_per_hit
            lethal_distance (float): lethal_distance
            lethal_angle_radians (float): how close must the nose of the aircraft be to the opponents body to be considered a hit
            assisted_flight (bool): whether to fly using RPYT controls or manual control of all actuators
            render (bool): whether to render the environment
            human_camera (bool): to allow the image from `render` to be available without the PyBullet display
        """

        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise AssertionError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        """SPACES"""
        self.high = np.ones(4 if assisted_flight else 6)
        self.low = self.high * -1.0
        self.low[-1] = 0.0

        self.state_shape = 13  # 12 states + health

        # MUDAR
        # self.observation_space = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=(2 * self.state_shape + self.action_space().shape[0],),
        # )

        # Parallel env stuff
        self.possible_agents = ['lm_' + str(r) for r in range(1,3)]  # 2 is the original dogfight value
        self.agent_id_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self.id_agent_mapping = dict(zip(list(range(len(self.possible_agents))), self.possible_agents ))
        self.render_mode = render

        """CONSTANTS"""
        self.env: Aviary
        self.num_drones = len(self.possible_agents)
        self.to_render = render
        self.human_camera = human_camera
        self.max_steps = int(agent_hz * max_duration_seconds) if not render else np.inf
        self.env_step_ratio = int(120 / agent_hz)
        self.flight_dome_size = flight_dome_size
        self.aggressor_filepath = path.join(path.dirname(__file__), "../../models")

        self.assisted_flight = assisted_flight
        self.damage_per_hit = damage_per_hit
        self.spawn_height = spawn_height
        self.lethal_distance = lethal_distance
        self.lethal_angle = lethal_angle_radians

    # parallel env
    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=self.low, high=self.high, dtype=np.float64)


    # @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * self.state_shape + self.action_space([self.agents[0]]).shape[0],),
        )


    def reset(self): #-> tuple[np.ndarray, dict]:
        """Resets the environment

        Args:

        Returns:
            tuple[np.ndarray, dict()]:
        """
        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

        # reset learning parameters
        # self.step_count = 0
        # self.termination = np.zeros((2), dtype=bool)
        # self.truncation = np.zeros((2), dtype=bool)
        # self.reward = np.zeros((2))
        # self.state = np.zeros((2, self.observation_space.shape[0]))
        # self.prev_actions = np.zeros((2, 4))
        # self.info = {}
        # self.info["out_of_bounds"] = False
        # self.info["collision"] = False
        # self.info["d1_win"] = False
        # self.info["d2_win"] = False
        # self.info["healths"] = np.ones((2))

        ## pettingzoo reset learning parameters
        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.termination = {agent: np.zeros((1), dtype=bool) for agent in self.agents}
        self.truncation = {agent: np.zeros((1), dtype=bool) for agent in self.agents}
        self.reward = {agent: np.zeros((1)) for agent in self.agents}
        self.state = {agent: np.zeros((1, self.observation_space(self.agents[0]).shape[0])) for agent in self.agents}
        self.prev_actions = {agent: np.zeros((1, 2)) for agent in self.agents}
        self.info = {agent:
            dict(
                out_of_bounds=False,
                collision=False,
                d1_win=False,
                d2_win=False,
                healths=np.ones((2))
            ) for agent in self.agents}


        # reset runtime parameters
        # self.health = np.ones((2))
        # self.in_cone = np.zeros((2,), dtype=bool)
        # self.in_range = np.zeros((2,), dtype=bool)
        # self.chasing = np.zeros((2,), dtype=bool)
        # self.current_hits = np.zeros((2), dtype=bool)
        # self.current_angles = np.zeros((2))
        # self.current_offsets = np.zeros((2))
        # self.current_distance = np.zeros((2))
        # self.previous_hits = np.zeros((2), dtype=bool)
        # self.previous_angles = np.zeros((2))
        # self.previous_offsets = np.zeros((2))
        # self.previous_distance = np.zeros((2))

        self.health = {agent: np.ones((1)) for agent in self.agents} # PROBABLY WRONG
        self.in_cone = {agent: np.zeros((1,), dtype=bool) for agent in self.agents}
        self.in_range = {agent: np.zeros((1,), dtype=bool) for agent in self.agents}
        self.chasing = {agent: np.zeros((1,), dtype=bool) for agent in self.agents}
        self.current_hits = {agent: np.zeros((1), dtype=bool) for agent in self.agents}
        self.current_angles = {agent: np.zeros((1)) for agent in self.agents}
        self.current_offsets = {agent: np.zeros((1)) for agent in self.agents}
        self.current_distance = {agent: np.zeros((1)) for agent in self.agents}
        self.previous_hits = {agent: np.zeros((1), dtype=bool) for agent in self.agents}
        self.previous_angles = {agent: np.zeros((1)) for agent in self.agents}
        self.previous_offsets = {agent: np.zeros((1)) for agent in self.agents}
        self.previous_distance = {agent: np.zeros((1)) for agent in self.agents}

        # # randomize starting position and orientation
        # # constantly regenerate starting position if they are too close
        # # fix height to 20 meters
        # start_pos = np.zeros((2, 3))
        # while np.linalg.norm(start_pos[0] - start_pos[1]) < self.flight_dome_size * 0.2:
        #     start_pos = (np.random.rand(2, 3) - 0.5) * self.flight_dome_size * 0.5
        #     start_pos[:, -1] = self.spawn_height
        # start_orn = (np.random.rand(2, 3) - 0.5) * 2.0 * np.array([1.0, 1.0, 2 * np.pi])

        ## pettingzoo conversion
        # randomize starting position and orientation
        # constantly regenerate starting position if they are too close
        # fix height to 20 meters
        start_pos = {agent: np.zeros((3)) for agent in self.agents}
        while np.linalg.norm(start_pos[self.agents[0]] - start_pos[self.agents[1]]) < self.flight_dome_size * 0.2:
            start_pos = {agent: (np.random.rand(3) - 0.5) * self.flight_dome_size * 0.5 for agent in self.agents}
            for agent in self.agents:
                start_pos[agent][-1] = self.spawn_height

        start_orn = {agent: (np.random.rand(3) - 0.5) * 2.0 * np.array([1.0, 1.0, 2 * np.pi]) for agent in
                     self.agents}

        # start_pos = np.array([[0, 0, 5], [5, 5, 10]])
        # start_orn = np.zeros_like(start_pos)
        # start_orn[1, -1] = np.pi
        # start_orn[1, 0] = np.pi / 2

        start_vec = dict()
        for agent in self.agents:
            _ , start_vec[agent] = self.compute_rotation_forward(start_orn[agent])
            start_vec[agent] = start_vec[agent] * 10.0

        # define all drone options
        # drone_options = [dict(), dict()]
        # for i in range(len(drone_options)):
        #     drone_options[i]["model_dir"] = self.aggressor_filepath
        #     drone_options[i]["drone_model"] = "aggressor"
        #     drone_options[i]["starting_velocity"] = start_vec[i]
        # drone_options[0]["use_camera"] = self.human_camera or self.to_render
        # drone_options[0]["camera_resolution"] = np.array([240, 320])

        ## pettingzoo
        # define all drone options
        drone_options = {agent: dict() for agent in self.agents}

        drone_options = {agent:
            dict(
                model_dir=self.aggressor_filepath,
                drone_model="aggressor",
                starting_velocity=start_vec[agent],
                use_camera=self.human_camera or self.to_render ,
                camera_resolution=np.array([240, 320]),
            ) for agent in self.agents}

        # start the environment
        self.env = Aviary(
            start_pos=np.array(list(start_pos.values())),
            start_orn=np.array(list(start_orn.values())),
            render=self.to_render,
            drone_type="fixedwing",
            drone_options=[value for value in drone_options.values()]
        )

        # render settings
        if self.to_render:
            self.env.drones[0].camera.camera_position_offset = [-10, 0, 5]

        # set flight mode and register all bodies
        self.env.register_all_new_bodies()
        self.env.set_mode(0 if self.assisted_flight else -1)

        # wait for env to stabilize
        for _ in range(3):
            self.env.step()

        return self.state, self.info

    def step(self, actions): # -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """step.

        Args:
            actions: an [n, 4] array of each drone's action

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """

        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}


        # set the actions, reset the reward
        self.env.set_all_setpoints(np.array(list(actions.values())))
        self.prev_actions = actions.copy()
        for agent in self.agents:
            self.reward[agent] *= 0.0

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if any(value for value in self.termination.values()) or any(value for value in self.truncation.values()):
                break

            self.env.step()

            # compute state and done
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        # colour the gunsights conditionally
        for agent in self.agents:
            if self.render and not np.all(self.previous_hits == self.current_hits[agent]):
                self.previous_hits[agent] = self.current_hits[agent].copy()
                hit_colour = np.array([1.0, 0.0, 0.0, 0.2])
                norm_colour = np.array([0.0, 0.0, 0.0, 0.2])

                self.env.changeVisualShape(
                    self.env.drones[self.agent_id_mapping[agent]].Id,
                    7,
                    rgbaColor=(hit_colour if self.current_hits[agent] else norm_colour),
                )

        return self.state, self.reward, self.termination, self.truncation, self.info

    def compute_state(self):
        # Get the states of both drones
        attitudes = np.array(self.env.all_states)

        """COMPUTE HITS"""
        # Get the rotation matrices and forward vectors
        # Offset the position to be on top of the main wing
        rotation, forward_vecs = self.compute_rotation_forward(attitudes[:, 1])
        attitudes[:, -1] -= forward_vecs * 0.35

        # Compute the vectors of each drone to each drone
        separation = attitudes[::-1, -1] - attitudes[:, -1]
        self.previous_distance = self.current_distance.copy()
        self.current_distance = np.linalg.norm(separation[0])

        # Compute engagement angles
        self.previous_angles = self.current_angles.copy()
        self.current_angles = np.arccos(
            np.sum(separation * forward_vecs, axis=-1) / self.current_distance
        )

        # Compute engagement offsets
        self.previous_offsets = self.current_offsets.copy()
        self.current_offsets = np.linalg.norm(
            np.cross(separation, forward_vecs), axis=-1
        )

        # Whether we're lethal or chasing or have an opponent in the cone
        for agent in self.agents:
            self.in_cone[agent] = self.current_angles[self.agent_id_mapping[agent]] < self.lethal_angle
            self.in_range[agent] = self.current_distance < self.lethal_distance
            self.chasing[agent] = np.abs(self.current_angles[self.agent_id_mapping[agent]]) < (np.pi / 2.0)

        # Compute whether anyone hit anyone
        for agent in self.agents:
            self.current_hits[agent] = self.in_cone[agent] & self.in_range[agent] & self.chasing[agent]

        # Update health based on hits
        for agent in self.agents:
            self.health[agent] -= self.damage_per_hit * self.current_hits[agent]

        """COMPUTE THE STATE VECTOR"""
        # form the opponent state matrix
        opponent_attitudes = np.zeros_like(attitudes)

        # opponent angular rate is unchanged
        opponent_attitudes[:, 0] = attitudes[::-1, 0]

        # oponent angular position is relative to ours
        opponent_attitudes[:, 1] = attitudes[::-1, 1] - attitudes[:, 1]

        # opponent velocity is relative to ours in our body frame
        ground_velocities = (
            rotation @ np.expand_dims(attitudes[:, -2], axis=-1)
        ).reshape(2, 3)
        opponent_velocities = (
            np.expand_dims(ground_velocities, axis=1)[::-1] @ rotation
        ).reshape(2, 3)
        opponent_attitudes[:, 2] = opponent_velocities - attitudes[:, 2]

        # opponent position is relative to ours in our body frame
        opponent_attitudes[:, 3] = (
            np.expand_dims(separation, axis=1) @ rotation
        ).reshape(2, 3)


        # flatten the attitude and opponent attitude, expand dim of health
        flat_attitude = attitudes.reshape(2, -1)
        flat_opponent_attitude = opponent_attitudes.reshape(2, -1)
        health = {agent: np.expand_dims(self.health[agent], axis=-1) for agent in self.agents}

        # Form the state vector
        for agent in self.agents:
            self.state[agent] = np.concatenate(
                [
                    flat_attitude[agent],
                    health[agent],
                    flat_opponent_attitude[agent],
                    health[self.agents[::-1].index(agent)],
                    self.prev_actions[agent],
                ],
                axis=-1,
            )


    def compute_term_trunc_reward(self, agent):
        """compute_term_trunc_reward."""
        collisions = self.env.contact_array.sum(axis=0) > 0.0
        collisions = collisions[np.array([d.Id for d in self.env.drones])]
        out_of_bounds = (
                np.linalg.norm(self.attitudes[:, -1], axis=-1) > self.flight_dome_size
        )
        out_of_bounds |= self.attitudes[:, -1, -1] <= 0.0

        # terminate if out of bounds, no health, or collision
        self.termination |= out_of_bounds
        self.termination |= self.health[agent] <= 0.0
        self.termination |= collisions

        # truncation is just end
        self.truncation |= self.step_count > self.max_steps

        # reward for closing the distance
        self.reward += (
                np.clip(
                    self.previous_distance - self.current_distance, a_min=0.0, a_max=None
                )
                * (~self.in_range & self.chasing)
                * 1.0
        )

        # reward for progressing to engagement
        for agent in self.agents:
            self.reward[agent] += (
                    (self.previous_angles[agent] - self.current_angles[agent]) * self.in_range * 10.0
            )

        # reward for engaging the enemy
        for agent in self.agents:
            self.reward += 3.0 / (self.current_angles[agent] + 0.1) * self.in_range[agent]

        # reward for hits
        for agent in self.agents:
            self.reward += 30.0 * self.current_hits[agent]

        # penalty for being hit
        for agent in self.agents:
            self.reward -= 20.0 * self.current_hits[agent][::-1]

        # penalty for crashing
        self.reward[agent] -= 3000.0 * collisions

        # penalty for out of bounds
        self.reward -= 3000.0 * out_of_bounds

        # all the info things
        self.info["out_of_bounds"] = out_of_bounds
        self.info["collision"] = collisions
        self.info["wins"] = self.health[agent] <= 0.0
        self.info["healths"] = self.health

    def render(self) -> np.ndarray:
        return self.env.drones[0].rgbaImg

    @staticmethod
    def compute_rotation_forward(orn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes the rotation matrix and forward vector of an aircraft given its orientation.

        Args:
            orn (np.ndarray): an [n, 3] array of each drone's orientation

        Returns:
            np.ndarray: an [n, 3, 3] rotation matrix of each aircraft
            np.ndarray: an [n, 3] forward vector of each aircraft
        """
        c, s = np.cos(orn), np.sin(orn)
        eye = np.stack([np.eye(3)] * orn.shape[0], axis=0)

        rx = eye.copy()
        rx[:, 1, 1] = c[..., 0]
        rx[:, 1, 2] = -s[..., 0]
        rx[:, 2, 1] = s[..., 0]
        rx[:, 2, 2] = c[..., 0]
        ry = eye.copy()
        ry[:, 0, 0] = c[..., 1]
        ry[:, 0, 2] = s[..., 1]
        ry[:, 2, 0] = -s[..., 1]
        ry[:, 2, 2] = c[..., 1]
        rz = eye.copy()
        rz[:, 0, 0] = c[..., 2]
        rz[:, 0, 1] = -s[..., 2]
        rz[:, 1, 0] = s[..., 2]
        rz[:, 1, 1] = c[..., 2]

        forward_vector = np.stack(
            [c[..., 2] * c[..., 1], s[..., 2] * c[..., 1], -s[..., 1]], axis=-1
        )

        # order of operations for multiplication matters here
        return rz @ ry @ rx, forward_vector
