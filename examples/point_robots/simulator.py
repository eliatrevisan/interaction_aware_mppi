from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.goals.static_sub_goal import StaticSubGoal
import gymnasium as gym
import torch
import numpy as np

class Simulator:
    def __init__(self, cfg) -> None:
        self._device = cfg["device"]
        self._dt = cfg["dt"]
        self._environment = self._initalize_environment(cfg)
        self._agents = {name: i for i, name in enumerate(cfg["agents"])}

    def _initalize_environment(self, cfg) -> UrdfEnv:
        """
        Initializes the simulation environment.

        Adds an obstacle and goal visualizaion to the environment and
        steps the simulation once.

        Params
        ----------
        render
            Boolean toggle to set rendering on (True) or off (False).
        """
        robots = []
        for _ in cfg['agents'].items():
            robots.append(GenericUrdfReacher(urdf=cfg["simulator"]["urdf"], mode=cfg["simulator"]["mode"]))
            
        env: UrdfEnv = gym.make("urdf-env-v0", dt=self._dt, robots=robots, render=cfg['simulator']['render'])

        # Extract initial positions from agents
        initial_positions = [agent_info['initial_pose'] for agent_info in cfg['agents'].values()]
        # Reset positions in the environment
        env.reset(pos=np.array(initial_positions))

        for agent_info in cfg['agents'].values():
            goal_dict = {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link": 0,
                "child_link": 1,
                "desired_position": agent_info['initial_goal'],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
            goal = StaticSubGoal(name="simpleGoal", content_dict=goal_dict)
            env.add_goal(goal)
        return env

    def step(self, action) -> torch.Tensor:
        # Extract the actions from the dictionary and concatenate them into a numpy array
        action_array = np.concatenate([a.cpu().numpy() for a in action.values()])

        observation_dict, _, terminated, _, info = self._environment.step(action_array)

        restructured_observation_dict = {}
        for agent_name, obs_name in zip(self._agents, observation_dict):
            position = observation_dict[obs_name]['joint_state']['position']
            velocity = observation_dict[obs_name]['joint_state']['velocity']
            state = torch.cat([torch.tensor(position, device=self._device), torch.tensor(velocity, device=self._device)])
            restructured_observation_dict[agent_name] = state

        return restructured_observation_dict