import torch


class OmnidirectionalPointRobotDynamics:
    def __init__(self, dt=0.05, device="cuda:0") -> None:
        self._dt = dt
        self._device = device

    def step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x, y, theta = states[:, :, 0], states[:, :, 1], states[:, :, 2]

        new_x = x + actions[:, :, 0] * self._dt
        new_y = y + actions[:, :, 1] * self._dt
        new_theta = theta + actions[:, :, 2] * self._dt

        new_states = torch.cat([new_x.unsqueeze(2), new_y.unsqueeze(2), new_theta.unsqueeze(2), actions], dim=2)
        return new_states, actions
    
class JackalDynamics:
    def __init__(self, dt=0.05, device="cuda:0") -> None:
        self._dt = dt
        self._device = device
    
    def step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x, y, theta, vx, vy, omega = states[:, :, 0], states[:, :, 1], states[:, :, 2], states[:, :, 3], states[:, :, 4], states[:, :, 5]

        # Update velocity and position using bicycle model
        new_vx = actions[:, :,0] * torch.cos(theta)
        new_vy = actions[:, :,0] * torch.sin(theta)
        new_omega = actions[:, :,1]

        new_x = x + new_vx * self._dt
        new_y = y + new_vy * self._dt
        new_theta = theta + new_omega * self._dt

        new_states = torch.stack([new_x, new_y, new_theta, new_vx, new_vy, new_omega], dim=2)
        return new_states, actions
    
class QuarterRoboatDynamics:
    def __init__(self, cfg) -> None:
        self.aa = 0.45
        self.bb = 0.90

        self.m11 = 12
        self.m22 = 24
        self.m33 = 1.5
        self.d11 = 6
        self.d22 = 8
        self.d33 = 1.35

        self.cfg = cfg
        self.dt = cfg["dt"]
        self.n_samples = cfg["mppi"]["num_samples"]
        self.device = cfg["device"]


        ## Dynamics
        self.D = torch.tensor([		[self.d11	    ,0		,0      ],
                                    [0		,self.d22	    ,0	    ],
                                    [0		,0	    ,self.d33	    ]], device=self.device)

        self.M = torch.tensor([	    [self.m11		,0		,0		],
                                    [0		,self.m22	    ,0		],
                                    [0		,0		,self.m33       ]], device=self.device)
        
        self.B = torch.tensor([	    [1		,1		,0		,0],
                                    [0		,0	    ,1		,1],
                                    [self.aa/2		,-self.aa/2		,self.bb/2    ,-self.aa/2    ]], device=self.device)

        # Inverse of inertia matrix (precalculated for speed)
        self.Minv = torch.inverse(self.M)

    def rot_matrix(self, heading):
        stacked = torch.stack([torch.cos(heading), -torch.sin(heading), torch.zeros_like(heading), torch.sin(heading), torch.cos(heading), torch.zeros_like(heading), torch.zeros_like(heading), torch.zeros_like(heading), torch.ones_like(heading)], dim=1).reshape(heading.size(0), 3, 3, heading.size(1)).to(self.device)

        return stacked.permute(0, 3, 1, 2)
        
    def coriolis(self, vel):
        stacked = torch.stack([torch.zeros_like(vel[:, :, 0]), torch.zeros_like(vel[:, :, 0]), -self.m22  * vel[:, :,1], torch.zeros_like(vel[:, :, 0]), torch.zeros_like(vel[:, :, 0]), self.m11 * vel[:, :,0], self.m22 * vel[:, :,1], -self.m11 * vel[:, :,0], torch.zeros_like(vel[:, :, 0]),], dim=1).reshape(vel.size(0), 3, 3, vel.size(1)).to(self.device)

        return stacked.permute(0, 3, 1, 2)
        
    def step(self, states: torch.Tensor, actions: torch.Tensor, t: int = -1) -> torch.Tensor:

        # # Change dt if the horizon cutoff is reached to extend the predicted time horizon
        # if t < self.cfg["mppi"]["horizon_cutoff"]:
        #     self.dt = self.cfg["mppi"]["dt"]
        # else:
        #     self.dt = self.cfg["mppi"]["dt_horizon"]


        # Set u 
        u = actions

        # Set current pose and velocity
        pose_enu = states[:,:,0:3]
        
        # Convert from ENU to NED
        pose = torch.zeros_like(pose_enu)
        pose[:, :, 0] = pose_enu[:, :, 1]
        pose[:, :, 1] = pose_enu[:, :, 0]
        pose[:, :, 2] = torch.pi/2 -pose_enu[:, :, 2]

        vel_enu = states[:,:,3:6]
        # Convert from ENU to NED
        vel = torch.zeros_like(vel_enu)
        vel[:, :, 0] = vel_enu[:, :, 1]
        vel[:, :, 1] = vel_enu[:, :, 0]
        vel[:, :, 2] = -vel_enu[:, :, 2]

        # Rotate velocity to the body frame
        vel_body = torch.bmm(self.rot_matrix(-pose[:,:,2]).reshape(-1,3,3), vel.reshape(-1,3).unsqueeze(2)).reshape(vel.size(0), vel.size(1), vel.size(2))

        # print(pose_enu[0,0,:])
        # print(vel_body[0,0,:])

        # Compute new velocity
        Minv_batch = self.Minv.repeat(vel.reshape(-1,3).size(0), 1, 1)
        B_batch = self.B.repeat(vel.reshape(-1,3).size(0), 1, 1)
        D_batch = self.D.repeat(vel.reshape(-1,3).size(0), 1, 1)
        C_batch = self.coriolis(vel_body).reshape(-1,3,3)
        
        new_vel_body = torch.bmm(Minv_batch, (torch.bmm(B_batch, u.reshape(-1,4).unsqueeze(2))- torch.bmm(C_batch, vel_body.reshape(-1,3).unsqueeze(2)) - torch.bmm(D_batch, vel_body.reshape(-1,3).unsqueeze(2)))).reshape(vel.size(0), vel.size(1), vel.size(2)) * self.dt + vel_body

        # Rotate velocity to the world frame
        vel = torch.bmm(self.rot_matrix(pose[:,:,2]).reshape(-1,3,3), new_vel_body.reshape(-1,3).unsqueeze(2)).reshape(vel.size(0), vel.size(1), vel.size(2))

        # Compute new pose
        pose += self.dt * vel

        # Convert from NED to ENU
        new_pose = torch.zeros_like(pose)
        new_pose[:, :, 0] = pose[:, :, 1]
        new_pose[:, :, 1] = pose[:, :, 0]
        new_pose[:, :, 2] = torch.pi/2 -pose[:, :, 2]
        new_vel = torch.zeros_like(vel)
        new_vel[:, :, 0] = vel[:, :, 1]
        new_vel[:, :, 1] = vel[:, :, 0]
        new_vel[:, :, 2] = -vel[:, :, 2]

        # Set new state
        new_states = torch.concatenate((new_pose, new_vel),2)

        return new_states, actions