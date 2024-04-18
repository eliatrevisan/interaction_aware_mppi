import pybullet as p
import pybullet_data
import torch
import numpy as np
import os
import sys

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, '..'))


class Agent:

    def __init__(self, urdf, device, agent_cfg):
        self.device = device
        self.initial_pose = agent_cfg['initial_pose']
        self.initial_goal = agent_cfg['initial_goal']

        self.urdf_path = f'{abs_path}/urdf_files/{urdf}'
        self.height = 0.075
        self.state = torch.tensor([self.initial_pose[0], self.initial_pose[1], self.initial_pose[2], 0, 0, 0], device=self.device)
        self.pos = torch.tensor([self.state[0], self.state[1], 0.075], device=self.device)
        self.rot = torch.tensor([0, 0, self.state[2]], device=self.device)
        self.lin_vel = torch.tensor([self.state[3], self.state[4], 0], device=self.device)
        self.ang_vel = torch.tensor([0, 0, self.state[5]], device=self.device)
        self.urdf_id = p.loadURDF(self.urdf_path, basePosition=self.pos, baseOrientation=p.getQuaternionFromEuler(self.rot))
        p.setCollisionFilterGroupMask(self.urdf_id, -1, 0, 0, physicsClientId=0)

        goal_urdf_path = f'{abs_path}/urdf_files/sphere.urdf'
        self.goal_id = p.loadURDF(goal_urdf_path, basePosition=[self.initial_goal[0], self.initial_goal[1], 0], useFixedBase=True)
        p.setCollisionFilterGroupMask(self.goal_id, -1, 0, 0, physicsClientId=0)
    
    def get_state(self):
        return self.state

    def update_state(self, state):
        self.state = state
        self.pos = torch.tensor([self.state[0], self.state[1], 0.075], device=self.device)
        self.rot = torch.tensor([0, 0, self.state[2]], device=self.device)
        self.lin_vel = torch.tensor([self.state[3], self.state[4], 0], device=self.device)
        self.ang_vel = torch.tensor([0, 0, self.state[5]], device=self.device)
        p.resetBasePositionAndOrientation(self.urdf_id, self.pos.cpu().numpy(), p.getQuaternionFromEuler(self.rot.cpu().numpy()))
        p.resetBaseVelocity(self.urdf_id, linearVelocity=self.lin_vel.cpu().numpy(), angularVelocity=self.ang_vel.cpu().numpy())
        return

        

class Simulator:
    def __init__(self, cfg, dynamics) -> None:
        self._device = cfg["device"]
        self._dt = cfg["dt"]
        self._environment = self._initalize_environment(cfg)
        self._first_plot = True
        self._dynamics = dynamics
        self._nx = cfg['mppi']['nx']

    def _initalize_environment(self, cfg):

        # Show GUI or not
        if cfg["simulator"]["render"]:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # This hides the GUI widgets

        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])

        # Load ground plane and goal position
        ground = p.loadURDF("plane.urdf")
        p.setCollisionFilterGroupMask(ground, -1, 0, 0, physicsClientId=0)
        self._agents = {name: Agent(cfg['simulator']['urdf'], cfg['device'], agent_cfg) for name, agent_cfg in cfg["agents"].items()}

        print("Simulator initialized")

    def step(self, action) -> torch.Tensor:
        # Extract the actions from the dictionary and stack them
        action_tensor = torch.stack([a for a in action.values()])

        state_tensor = torch.stack([agent.get_state() for agent in self._agents.values()])

        observation_tensor, action_tensor = self._dynamics(state_tensor.unsqueeze(0), action_tensor.unsqueeze(0))

        observation_dict = {}

        for i, agent_name in enumerate(self._agents.keys()):
            observation_dict[agent_name] = observation_tensor[:,i,:].squeeze()
            self._agents[agent_name].update_state(observation_tensor[:,i,:].squeeze())

        return observation_dict
    
    def plot_trajectories(self, traj_dict):
        if self._first_plot:
            self.marker_ids = []
            colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1]]  # Add more colors if needed
            iterations = len(traj_dict)/len(self._agents)
            for i in range(int(iterations)):
                color = colors[i % len(colors)]  # Cycle through colors if there are more agents than colors
                for _ in range(len(self._agents)*traj_dict[next(iter(traj_dict))].size()[0]):
                    visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=color)
                    body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id)
                    self.marker_ids.append(body_id)
            self._first_plot = False

        for i, agent_name in enumerate(traj_dict.keys()):
            trajectory_positions = traj_dict[agent_name].cpu().numpy()
            for j, state in enumerate(trajectory_positions):
                position_3d = np.append(state[:2], 0)
                p.resetBasePositionAndOrientation(self.marker_ids[i*len(trajectory_positions) + j], position_3d, (0, 0, 0, 1))
        return