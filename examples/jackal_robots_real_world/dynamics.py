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