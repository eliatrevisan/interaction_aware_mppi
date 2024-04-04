from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.goals.static_sub_goal import StaticSubGoal
import gymnasium as gym
import torch
import numpy as np

class Simulator:
    def __init__(self, agents_cfg, sys_cfg) -> None:
        self._device = sys_cfg["device"]
        self._dt = sys_cfg["dt"]
        self._environment = self._initalize_environment(agents_cfg, sys_cfg)
        self._agents = agents_cfg.keys()

    def _initalize_environment(self, agents_cfg, sys_cfg) -> UrdfEnv:
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
        for agent_name, agent_cfg in agents_cfg.items():
            robot_type = agent_cfg["simulator"]["type"]
            if robot_type == 'GenericUrdfReacher':
                robots.append(GenericUrdfReacher(urdf=agent_cfg["simulator"]["urdf"], mode=agent_cfg["simulator"]["mode"]))
            # Add more elif conditions here for other robot types
            # elif robot_type == 'OtherRobotType':
            #     robots.append(OtherRobotType(...))
            
        env: UrdfEnv = gym.make("urdf-env-v0", dt=self._dt, robots=robots, render=sys_cfg['simulator']['render'])
        # Extract initial positions from agents_cfg
        initial_positions = [agent_cfg['initial_pose'] for agent_name, agent_cfg in agents_cfg.items()]
        # Reset positions in the environment
        env.reset(pos=np.array(initial_positions))
        for agent_name, agent_cfg in agents_cfg.items():
            goal_dict = {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link": 0,
                "child_link": 1,
                "desired_position": agent_cfg["initial_goal"],
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
            state = torch.cat([torch.tensor(position), torch.tensor(velocity)])
            restructured_observation_dict[agent_name] = state

        return restructured_observation_dict