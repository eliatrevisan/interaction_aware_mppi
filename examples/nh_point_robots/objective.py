import torch


class OmnidirectionalPointRobotObjective(object):
    def __init__(self, goal=[1.0, 1.0], device="cuda:0"):
        self.nav_goal = torch.tensor(goal, device=device)

    def compute_running_cost(self, state: torch.Tensor):
        positions = state[:, 0:2]
        goal_dist = torch.linalg.norm(positions - self.nav_goal, axis=1)
        return goal_dist * 1.0

class SocialNavigationObjective(object):
    def __init__(self, device="cuda:0"):
        self.min_dist = 0.5
        self.coll_cost = 10.0

    def compute_configuration_cost(self, agents_states):
        return self.dynamic_collision_cost(agents_states)
    
    def dynamic_collision_cost(self, agents_states):
        # Compute collision cost for each pair of agents
        n = len(agents_states)
        i, j = torch.triu_indices(n, n, 1)  # Get indices for upper triangle of matrix
        agent_i_states = torch.stack([agents_states[index] for index in i])
        agent_j_states = torch.stack([agents_states[index] for index in j])

        # Compute the distance between each pair of agents
        dist = torch.linalg.norm(agent_i_states[:, :, :2] - agent_j_states[:, :, :2], dim=2)
        # Compute the cost for each sample
        cost = torch.sum((dist < self.min_dist).float() * self.coll_cost, dim=0)

        return cost