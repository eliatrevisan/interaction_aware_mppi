import torch


class OmnidirectionalPointRobotObjective(object):
    def __init__(self, goals, device="cuda:0"):
        self.nav_goals = torch.tensor(goals, device=device).unsqueeze(0)

    def compute_running_cost(self, state: torch.Tensor):
        positions = state[:, :, 0:2]
        goal_dist = torch.linalg.norm(positions - self.nav_goals, axis=2)
        return goal_dist * 1.0
    
    def get_goals(self):
        return self.nav_goals.squeeze(0)
    
    def set_goals(self, goals):
        self.nav_goals = torch.tensor(goals, device=self.nav_goals.device).unsqueeze(0)
        return None

class SocialNavigationObjective(object):
    def __init__(self, device="cuda:0"):
        self._device = device
        self._min_dist = 0.5
        self._coll_weight = 10.0
        self._rule_cross_radius = 5.0
        self._rule_headon_radius = 2.0
        self._rule_angle = torch.pi/4.0
        self._rule_min_vel = 0.1
        self._headon_weight = 0.0
        self._crossing_weight = 5.0


    def compute_running_cost(self, agents_states, init_agent_state, t):
        return self._rule_cost(agents_states, init_agent_state, t) + self._dynamic_collision_cost(agents_states)
    
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
    
    def _rule_cost(self, agents_states, init_agent_states, t):
        # Compute cost for head-on collisions
        n = agents_states.shape[1]
        a, b = torch.triu_indices(n, n, 1)  # Get indices for upper triangle of matrix
        i = torch.concat([a,b])
        j = torch.concat([b,a])
        agent_i_states = torch.stack([agents_states[:,index,:] for index in i])
        agent_j_states = torch.stack([agents_states[:,index,:] for index in j])
        init_agent_i_states = torch.stack([init_agent_states[:,index,:] for index in i])
        init_agent_j_states = torch.stack([init_agent_states[:,index,:] for index in j])

        # From here on, we assume the velocities are in the world frame
        # if not, rotate them before continuing!
        # Also, we assume EastNorthUp
        # test_i = torch.tensor([-0.5,0,0,0,1,0]).unsqueeze(0).unsqueeze(0)
        # test_j = torch.tensor([0.5,0,0,0,-1,0]).unsqueeze(0).unsqueeze(0)
        # self._check_right_side(test_i, test_j)
        # self._check_vel_headon(test_i, test_j)
        right_side = self._check_right_side(agent_i_states, agent_j_states)
        headon = self._check_vel_headon(agent_i_states, agent_j_states)
        priority = self._check_priority(init_agent_i_states, init_agent_j_states, agent_i_states.shape[1])
        crossed_constvel = self._check_crossed_constvel(agent_i_states, init_agent_j_states, t)

        return torch.sum((right_side & headon) * self._headon_weight, dim=0) + torch.sum((priority & crossed_constvel) * self._crossing_weight, dim=0)

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
        is_right_side = (magnitude_vij < self._rule_headon_radius) & (torch.abs(angle_diff) < self._rule_angle)

        return is_right_side
    
    def _check_vel_headon(self, agent_i_states, agent_j_states):
        # Get the velocities of the agents
        vel_i = agent_i_states[:, :, 3:5]
        vel_j = agent_j_states[:, :, 3:5]

        # Compute the angle between vel_i and vel_j
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle_vel_j = torch.atan2(vel_j[:, :, 1], vel_j[:, :, 0])
        angle = angle_vel_j - angle_vel_i

        # compute angle diff and normalize to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle - torch.pi), torch.cos(angle - torch.pi))

        # Compute the magnitudes of vel_i and vel_j
        magnitude_vel_i = torch.sqrt((vel_i ** 2).sum(dim=2))
        magnitude_vel_j = torch.sqrt((vel_j ** 2).sum(dim=2))

        # Check if the absolute difference between angle and pi is less than the rule angle
        is_headon = (torch.abs(angle_diff) < self._rule_angle) & (magnitude_vel_i > self._rule_min_vel) & (magnitude_vel_j > self._rule_min_vel)

        return is_headon
    
    def _check_priority(self, init_agent_i_states, init_agent_j_states, k):
        pos_i = init_agent_i_states[:, :, :2]
        pos_j = init_agent_j_states[:, :, :2]
        vel_i = init_agent_i_states[:, :, 3:5]
        vel_j = init_agent_j_states[:, :, 3:5]

        # Compute the vector from the first agent to the second agent
        vij = pos_j - pos_i

        # Compute the angle between vel_i and vij 
        angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle = angle_vij - angle_vel_i

        # Nomalize to [-pi, pi]
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))

        # Compute the magnitudes of vij
        magnitude_vij = torch.sqrt((vij ** 2).sum(dim=2))

        is_front_right = (magnitude_vij < self._rule_cross_radius) & (angle < 0) & (angle > -torch.pi/2)

        # Compute the angle between vel_i and vel_j
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle_vel_j = torch.atan2(vel_j[:, :, 1], vel_j[:, :, 0])
        angle_2 = angle_vel_j - angle_vel_i

        # compute angle diff and normalize to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle_2 - torch.pi/2), torch.cos(angle_2 - torch.pi/2))

        # Compute the magnitudes of vel_i and vel_j
        magnitude_vel_i = torch.sqrt((vel_i ** 2).sum(dim=2))
        magnitude_vel_j = torch.sqrt((vel_j ** 2).sum(dim=2))

        is_giveway_vel = (torch.abs(angle_diff) < self._rule_angle) & (magnitude_vel_i > self._rule_min_vel) & (magnitude_vel_j > self._rule_min_vel)

        return is_front_right.expand(-1, k) & is_giveway_vel.expand(-1, k)
    
    def _check_crossed_constvel(self, agent_i_states, init_agent_j_states, t):
        pos_i = agent_i_states[:, :, :2]
        init_pos_j = init_agent_j_states[:, :, :2]
        vel_i = agent_i_states[:, :, 3:5]
        init_vel_j = init_agent_j_states[:, :, 3:5]

        # find current position of agent j
        pos_j = init_pos_j + init_vel_j * t
        # pos_j = init_pos_j
        # make pos_j same size as pos_i
        pos_j = pos_j.expand(-1, pos_i.shape[1], -1)

        # Compute the vector from the first agent to the second agent
        vij = pos_j - pos_i

        # Compute the angle between vij and vel_i
        angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle = angle_vij - angle_vel_i

        # # Nomalize to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle + torch.pi/2), torch.cos(angle + torch.pi/2))
        # angle = torch.atan2(torch.sin(angle), torch.cos(angle))

        crossed_constvel = (angle_diff < 0) & (angle_diff > -torch.pi/2)
        # crossed_constvel = (angle < 0)

        return crossed_constvel