import torch


class RoboatObjective(object):
    def __init__(self, goals, device="cuda:0"):
        self.nav_goals = torch.tensor(goals, device=device).unsqueeze(0)
        self._goal_weight = 2.0
        self._back_vel_weight = 1.0
        self._rot_vel_weight = 0.01
        self._lat_vel_weight = 0.05
        self._heading_to_goal_weight = 1.0
        self._max_speed = 0.3 # m/s
        self._max_speed_weight = 5.0

    def compute_running_cost(self, state: torch.Tensor):
        return self._goal_cost(state[:, :, 0:2])  + self._vel_cost(state) + self._heading_to_goal(state)
    
    def _goal_cost(self, positions):
        return torch.linalg.norm(positions - self.nav_goals, axis=2) * self._goal_weight
    
    def _vel_cost(self, state):
        # convert velocities to body frame
        cos = torch.cos(state[:, :, 2])
        sin = torch.sin(state[:, :, 2])
        vel_body = torch.stack([state[:, :, 3] * cos + state[:, :, 4] * sin, -state[:, :, 3] * sin + state[:, :, 4] * cos], dim=2)
        
        # penalize velocities in the back, lateral and rotational directions
        back_vel_cost = torch.relu(-vel_body[:, :, 0]) * self._back_vel_weight
        lat_vel_cost = vel_body[:, :, 1] ** 2 * self._lat_vel_weight
        rot_vel_cost = state[:, :, 5] ** 2 * self._rot_vel_weight

        # Calculate the magnitude of the velocity
        vel_magnitude = torch.norm(vel_body, dim=2)

        # Penalize velocity magnitude exceeding max speed
        exceed_max_speed_cost = (vel_magnitude - self._max_speed) ** 2 * self._max_speed_weight

        return back_vel_cost + lat_vel_cost + rot_vel_cost + exceed_max_speed_cost
    
    def _heading_to_goal(self, state):
        # Get the heading of the agent
        theta = state[:, :, 2]
        # Get the vector pointing to the goal
        goal = self.nav_goals - state[:, :, 0:2]
        # Compute the angle between the heading and the goal
        angle = torch.atan2(goal[:, :, 1], goal[:, :, 0]) - theta
        # Normalize the angle to [-pi, pi]
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))
        cost = torch.where(torch.linalg.norm(state[:,:,0:2] - self.nav_goals, axis=2)>0.5, angle**2, torch.zeros_like(angle))
        return cost * self._heading_to_goal_weight
    
    def get_goals(self):
        return self.nav_goals.squeeze(0)
    
    def set_goals(self, goals):
        self.nav_goals = torch.tensor(goals, device=self.nav_goals.device).unsqueeze(0)
        return None

class SocialNavigationObjective(object):
    def __init__(self, device="cuda:0"):
        self._device = device
        self._min_dist = 1.0
        self._width = 0.45
        self._height = 0.9
        self._coll_weight = 100.0
        self._rule_cross_radius = 5.0
        self._rule_headon_radius = 2.0
        self._rule_angle = torch.pi/4.0
        self._rule_min_vel = 0.05
        self._headon_weight = 5.0
        self._crossing_weight = 5.0


    def compute_running_cost(self, agents_states, init_agent_state, t):
        return self._rule_cost(agents_states, init_agent_state, t) + self._dynamic_collision_cost(agents_states)
    
    def _dynamic_collision_cost(self, agents_states):
        # Compute collision cost for each pair of agents
        n = agents_states.shape[1]
        i, j = torch.triu_indices(n, n, 1)  # Get indices for upper triangle of matrix
        agent_i_states = torch.stack([agents_states[:,index,:] for index in i])
        agent_j_states = torch.stack([agents_states[:,index,:] for index in j])

        # grid = self.create_occupancy_grid(agent_i_states)

        # Compute the distance between each pair of agents
        dist = torch.linalg.norm(agent_i_states[:, :, :2] - agent_j_states[:, :, :2], dim=2)
        # Compute the cost for each sample
        cost = torch.sum((dist < self._min_dist).float() * self._coll_weight, dim=0)

        return cost
    
    def create_occupancy_grid(self, agent_i_states):
        # Create a 1000x1000 pixel grid initialized with zeros
        grid = torch.zeros((1000, 1000))

        # Convert the agent's position from world coordinates to pixel coordinates
        # Assume that the world frame is centered at (500, 500) in pixel coordinates
        agent_position_pixel = (agent_i_states[:, :, :2] * 10 + 500).long()

        # Convert the agent's size from meters to pixels
        agent_size_pixel = (torch.tensor([self._height, self._width]) * 10).long()

        # Get the agent's heading
        theta = agent_i_states[:, :, 2]

        # Fill in the grid with ones where the agent is located
        for i in range(agent_position_pixel.shape[0]):
            x, y = agent_position_pixel[i]
            half_length, half_width = agent_size_pixel // 2

            # Create a rectangle representing the agent
            rectangle = torch.zeros((half_width * 2, half_length * 2))
            rectangle[half_width-half_width:half_width+half_width, half_length-half_length:half_length+half_length] = 1

            # Rotate the rectangle according to the agent's heading
            rotation_matrix = torch.tensor([[torch.cos(theta[i]), -torch.sin(theta[i])], [torch.sin(theta[i]), torch.cos(theta[i])]])
            rectangle = torch.einsum('ij,jkl->ikl', rotation_matrix, rectangle)

            # Add the rectangle to the grid
            grid[y-half_width:y+half_width, x-half_length:x+half_length] += rectangle

        return grid

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
        theta_i = agent_i_states[:, :, 2]

        # Compute the vector from the first agent to the second agent
        vij = pos_j - pos_i

        # Compute the angle between vij and vel_i
        angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle = angle_vij - angle_vel_i

        # # Compute the angle between vij and heading of agent i
        # angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        # angle = angle_vij - theta_i

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
        theta_i = agent_i_states[:, :, 2]
        theta_j = agent_j_states[:, :, 2]

        # Compute the angle between vel_i and vel_j
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle_vel_j = torch.atan2(vel_j[:, :, 1], vel_j[:, :, 0])
        angle = angle_vel_j - angle_vel_i

        # # Compute the angle between agents' headings
        # angle = theta_j - theta_i

        # compute angle diff and normalize to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle - torch.pi), torch.cos(angle - torch.pi))

        # Compute the magnitudes of vel_i and vel_j
        magnitude_vel_i = torch.sqrt((vel_i ** 2).sum(dim=2))
        magnitude_vel_j = torch.sqrt((vel_j ** 2).sum(dim=2))

        # Check if the absolute difference between angle and pi is less than the rule angle
        is_headon = (torch.abs(angle_diff) < self._rule_angle) & (magnitude_vel_i >= self._rule_min_vel) & (magnitude_vel_j >= self._rule_min_vel)

        return is_headon
    
    def _check_priority(self, init_agent_i_states, init_agent_j_states, k):
        pos_i = init_agent_i_states[:, :, :2]
        pos_j = init_agent_j_states[:, :, :2]
        vel_i = init_agent_i_states[:, :, 3:5]
        vel_j = init_agent_j_states[:, :, 3:5]
        theta_i = init_agent_i_states[:, :, 2]
        theta_j = init_agent_j_states[:, :, 2]

        # Compute the vector from the first agent to the second agent
        vij = pos_j - pos_i

        # Compute the angle between vel_i and vij 
        angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle = angle_vij - angle_vel_i

        # # Compute the angle between vij and heading of agent i
        # angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        # angle = angle_vij - theta_i

        # Nomalize to [-pi, pi]
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))

        # Compute the magnitudes of vij
        magnitude_vij = torch.sqrt((vij ** 2).sum(dim=2))

        is_front_right = (magnitude_vij < self._rule_cross_radius) & (angle < 0) & (angle > -torch.pi/2)

        # Compute the angle between vel_i and vel_j
        angle_vel_i = torch.atan2(vel_i[:, :, 1], vel_i[:, :, 0])
        angle_vel_j = torch.atan2(vel_j[:, :, 1], vel_j[:, :, 0])
        angle_2 = angle_vel_j - angle_vel_i

        # # Compute the angle between agents' headings
        # angle_2 = theta_j - theta_i

        # compute angle diff and normalize to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle_2 - torch.pi/2), torch.cos(angle_2 - torch.pi/2))

        # Compute the magnitudes of vel_i and vel_j
        magnitude_vel_i = torch.sqrt((vel_i ** 2).sum(dim=2))
        magnitude_vel_j = torch.sqrt((vel_j ** 2).sum(dim=2))

        is_giveway_vel = (torch.abs(angle_diff) < self._rule_angle) & (magnitude_vel_i >= self._rule_min_vel) & (magnitude_vel_j >= self._rule_min_vel)

        return is_front_right.expand(-1, k) & is_giveway_vel.expand(-1, k)
    
    def _check_crossed_constvel(self, agent_i_states, init_agent_j_states, t):
        pos_i = agent_i_states[:, :, :2]
        init_pos_j = init_agent_j_states[:, :, :2]
        vel_i = agent_i_states[:, :, 3:5]
        init_vel_j = init_agent_j_states[:, :, 3:5]
        theta_i = agent_i_states[:, :, 2]

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

        # # Compute the angle between vij and heading of agent i
        # angle_vij = torch.atan2(vij[:, :, 1], vij[:, :, 0])
        # angle = angle_vij - theta_i

        # # Nomalize to [-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle + torch.pi/2), torch.cos(angle + torch.pi/2))
        # angle = torch.atan2(torch.sin(angle), torch.cos(angle))

        crossed_constvel = (angle_diff < 0) & (angle_diff > -torch.pi/2)
        # crossed_constvel = (angle < 0)

        return crossed_constvel