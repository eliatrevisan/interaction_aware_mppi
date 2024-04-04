from mppi_torch.mppi import MPPIPlanner
import torch


class IAMPPIPlanner:
    def __init__(self, agents, configuration_cost, sys_cfg) -> None:
        self._device = sys_cfg["device"]
        self._dt = sys_cfg["dt"]
        self._agents = agents
        self._configuration_cost = configuration_cost

        # Create a new configuration dictionary
        sys_cfg = self._create_config(sys_cfg)
     
        self.planner = MPPIPlanner(
            cfg=sys_cfg["mppi"],
            nx=self.nx,
            dynamics=self._step,
            running_cost=self._compute_running_cost,
        )
    def _get_noise_sigma(self):
        noise_sigmas = []  # List to hold the noise_sigma matrices of each agent
        total_size = 0  # Total size of the resulting matrix

        for agent_name, agent in self._agents.items():
            noise_sigma = torch.tensor(agent.agent_cfg['mppi']['noise_sigma'])


            # Check if the matrix is square
            if noise_sigma.shape[0] != noise_sigma.shape[1]:
                raise ValueError(f"The noise_sigma matrix of agent {agent_name} is not square.")

            noise_sigmas.append(noise_sigma)
            total_size += noise_sigma.shape[0]

        # Stack the noise_sigma matrices diagonally
        noise_sigma_matrix = torch.block_diag(*noise_sigmas)

        return noise_sigma_matrix, total_size
    
    def _create_config(self, sys_cfg):
        self.noise_sigma_matrix, self.nu = self._get_noise_sigma()
        self.nx = sum([agent.agent_cfg['mppi']['nx'] for agent in self._agents.values()])
        self.u_min = torch.concat([torch.tensor(agent.agent_cfg['mppi']['u_min']) for agent in self._agents.values()])
        self.u_max = torch.concat([torch.tensor(agent.agent_cfg['mppi']['u_max']) for agent in self._agents.values()])

        # Create a new configuration dictionary
        new_config = {
            'noise_sigma': self.noise_sigma_matrix.tolist(),
            'u_min': self.u_min.tolist(),
            'u_max': self.u_max.tolist(),
        }

        # Update the sys_cfg['mppi'] dictionary with the new values
        sys_cfg['mppi'].update(new_config)

        return sys_cfg
    
    def _step(self, system_state, action, t):
        # Split the system_state tensor into individual states for each agent
        start = 0
        system_states = []
        for agent in self._agents.values():
            nx = agent.agent_cfg['mppi']['nx']  # Number of states for this agent
            agent_state = system_state[:, start:start+nx]
            system_states.append(agent_state)
            start += nx

        # Split the system_state tensor into individual states for each agent
        start = 0
        actions = []
        for agent in self._agents.values():
            nu = len(agent.agent_cfg['mppi']['noise_sigma'])  # Number of states for this agent
            agent_action = action[:, start:start+nu]
            actions.append(agent_action)
            start += nu

            new_states_list = []
            new_actions_list = []

        for agent, state, action in zip(self._agents.values(), system_states, actions):
            new_state, new_action = agent.dynamics.step(state, action)
            new_states_list.append(new_state)
            new_actions_list.append(new_action)

        new_states = torch.cat(new_states_list, dim=1)
        new_actions = torch.cat(new_actions_list, dim=1)

        return new_states, new_actions
    
    def _compute_running_cost(self, system_state):
        # Split the system_state tensor into individual states for each agent
        start = 0
        agents_states = []
        for agent in self._agents.values():
            nx = agent.agent_cfg['mppi']['nx']  # Number of states for this agent
            agent_state = system_state[:, start:start+nx]
            agents_states.append(agent_state)
            start += nx

        # Compute running cost for each agent
        agents_cost = torch.sum(torch.stack([agent.objective.compute_running_cost(state) 
                                            for agent, state in zip(self._agents.values(), agents_states)]), dim=0)

        conf_cost = self._configuration_cost(agents_states)

        return conf_cost + agents_cost
    
    def command(self, system_state):
        # Extract the states from the dictionary and convert them to tensors
        states = [torch.tensor(state) for state in system_state.values()]

        # Concatenate the states into a single tensor
        system_state_tensor = torch.cat(states, dim=0)

        # Get the actions as a concatenated tensor
        actions_tensor = self.planner.command(system_state_tensor)

        # Split the actions tensor into individual actions for each agent
        actions = {}
        start = 0
        for agent_name, agent in self._agents.items():
            n = len(agent.agent_cfg['mppi']['noise_sigma'])  # Number of inputs for this agent
            actions[agent_name] = actions_tensor[start:start+n]
            start += n

        return actions
    
    def zero_command(self):
        actions = {}
        for agent_name, agent in self._agents.items():
            n = len(agent.agent_cfg['mppi']['noise_sigma'])  # Number of inputs for this agent
            actions[agent_name] = torch.zeros(n)

        return actions