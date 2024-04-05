from urdfenvs.robots.generic_urdf import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.goals.static_sub_goal import StaticSubGoal
import gymnasium as gym
import torch
import numpy as np

class Simulator:
    def __init__(self, cfg) -> None:
        self._device = cfg["device"]
        self._dt = cfg["dt"]
        self._agents = {name: i for i, name in enumerate(cfg["agents"])}
        self._environment = self._initalize_environment(cfg)
        self._first_plot = True

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
            robots.append(GenericDiffDriveRobot(
                urdf=cfg['simulator']["urdf"],
                mode=cfg['simulator']["mode"],
                actuated_wheels=[
                    "rear_right_wheel",
                    "rear_left_wheel",
                    "front_right_wheel",
                    "front_left_wheel",
                ],
                castor_wheels=[],
                wheel_radius = 0.098,
                wheel_distance = 2 * 0.187795 + 0.08,
        ))
            
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
    
    def plot_trajectories(self, traj_dict):
        if self._first_plot:
            marker_ids = []
            colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1]]  # Add more colors if needed
            iterations = len(traj_dict)/len(self._agents)
            for i in range(int(iterations)):
                color = colors[i % len(colors)]  # Cycle through colors if there are more agents than colors
                marker_ids.extend([self._environment.add_visualization(size=[0.02,0.02], rgba_color=color) for _ in range(len(self._agents)*traj_dict[next(iter(traj_dict))].size()[0])])
            self._first_plot = False


        positions_3d = []
        for agent_name in traj_dict.keys():
            trajectory_positions = traj_dict[agent_name].cpu().numpy()
            trajectory_positions_3d = [np.append(state[:2], 0) for state in trajectory_positions]
            positions_3d.extend(trajectory_positions_3d)
        self._environment.update_visualizations(positions_3d)
        return