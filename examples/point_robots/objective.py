import torch


class OmnidirectionalPointRobotObjective(object):
    def __init__(self, goals, device="cuda:0"):
        self.nav_goals = torch.tensor(goals, device=device).unsqueeze(0)

    def compute_running_cost(self, state: torch.Tensor):
        positions = state[:, :, 0:2]
        goal_dist = torch.linalg.norm(positions - self.nav_goals, axis=2)
        return goal_dist * 1.0

class SocialNavigationObjective(object):
    def __init__(self, device="cuda:0"):
        self._device = device
        self._min_dist = 0.5
        self._coll_weight = 10.0
        self._rule_radius = 2.0
        self._rule_angle = torch.pi/6.0
        self._rule_min_vel = 0.1
        self._rule_weight = 1.0


    def compute_running_cost(self, agents_states, init_agent_state):
        return self._rule_cost(agents_states, init_agent_state) + self._dynamic_collision_cost(agents_states)
    
    def _dynamic_collision_cost(self, agents_states):
        # Compute collision cost for each pair of agents
        n = agents_states.shape[1]
        i, j = torch.triu_indices(n, n, 1)  # Get indices for upper triangle of matrix
        agent_i_states = torch.stack([agents_states[:,index,:] for index in i])
        agent_j_states = torch.stack([agents_states[:,index,:] for index in j])

        # Compute the distance between each pair of agents
        dist = torch.linalg.norm(agent_i_states[:, :, :2] - agent_j_states[:, :, :2], dim=2)
        # Compute the cost for each sample
        cost = torch.sum((dist < self._min_dist).float() * self._coll_weight, dim=0)

        return cost
    
    def _rule_cost(self, agents_states, init_agent_states):
        # Compute cost for head-on collisions
        n = agents_states.shape[1]
        a, b = torch.triu_indices(n, n, 1)  # Get indices for upper triangle of matrix
        i = torch.concat([a,b])
        j = torch.concat([b,a])
        agent_i_states = torch.stack([agents_states[:,index,:] for index in i])
        agent_j_states = torch.stack([agents_states[:,index,:] for index in j])

        # From here on, we assume the velocities are in the world frame
        # if not, rotate them before continuing!
        # Also, we assume EastNorthUp
        # test_i = torch.tensor([-0.5,0,0,0,1,0]).unsqueeze(0).unsqueeze(0)
        # test_j = torch.tensor([0.5,0,0,0,-1,0]).unsqueeze(0).unsqueeze(0)
        # self._check_right_side(test_i, test_j)
        # self._check_vel_headon(test_i, test_j)
        right_side = self._check_right_side(agent_i_states, agent_j_states)
        headon = self._check_vel_headon(agent_i_states, agent_j_states)

        return torch.sum((right_side & headon) * self._rule_weight, dim=0)

    def _check_right_side(self, agent_i_states, agent_j_states):
        # Get the positions of the agents
        pos_i = agent_i_states[:, :, :2]
        pos_j = agent_j_states[:, :, :2]
        vel_i = agent_i_states[:, :, 3:5]

        # Compute the vector from the first agent to the second agent
        vij = pos_j - pos_i

        # Compute the angle between vij and vel_i
        angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle = angle_vij - angle_vel_i

        # compute angle diff and nomalize to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle + torch.pi/2), torch.cos(angle + torch.pi/2))
        magnitude_vij = torch.sqrt((vij ** 2).sum(dim=2))

        # Check if the magnitude of vij is greater than the rule radius and the absolute difference between angle and pi/4 is less than the rule angle
        is_right_side = (magnitude_vij < self._rule_radius) & (torch.abs(angle_diff) < self._rule_angle)

        return is_right_side
    
    def _check_vel_headon(self, agent_i_states, agent_j_states):
        # Get the velocities of the agents
        vel_i = agent_i_states[:, :, 3:5]
        vel_j = agent_j_states[:, :, 3:5]

        # Compute the angle between vel_i and vel_j
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle_vel_j = torch.atan2(vel_j[:, :, 1], vel_j[:, :, 0])
        angle = angle_vel_i - angle_vel_j

        # compute angle diff and normalize to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle - torch.pi), torch.cos(angle - torch.pi))

        # Compute the magnitudes of vel_i and vel_j
        magnitude_vel_i = torch.sqrt((vel_i ** 2).sum(dim=2))
        magnitude_vel_j = torch.sqrt((vel_j ** 2).sum(dim=2))

        # Check if the absolute difference between angle and pi is less than the rule angle
        is_headon = (torch.abs(angle_diff) < self._rule_angle) & (magnitude_vel_i > self._rule_min_vel) & (magnitude_vel_j > self._rule_min_vel)

        return is_headon
    
    def _check_front_right(self, agent_i_states, agent_j_states):
        pos_i = agent_i_states[:, :, :2]
        pos_j = agent_j_states[:, :, :2]
        vel_i = agent_i_states[:, :, 3:5]

        # Compute the vector from the first agent to the second agent
        vij = pos_j - pos_i

        # Compute the dot product of vij and vel_i
        dot_product = (vij * vel_i).sum(dim=2)

        # Compute the magnitudes of vij and vel_i
        magnitude_vij = torch.sqrt((vij ** 2).sum(dim=2))
        magnitude_vel_i = torch.sqrt((vel_i ** 2).sum(dim=2))

        # Compute the angle between vij and vel_i
        angle = torch.acos(dot_product / (magnitude_vij * magnitude_vel_i))
        is_front_right = (magnitude_vij < self._rule_radius) & (angle < 0) & (angle > -torch.pi/2)
        return is_front_right  
    
    def _check_vel_giveway(self):
        return False
    def _check_crossed_constvel(self):
        return False