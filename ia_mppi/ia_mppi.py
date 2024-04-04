from mppi_torch.mppi import MPPIPlanner
import torch


class IAMPPIPlanner:
    def __init__(self, cfg, dynamics, agent_cost, config_cost) -> None:
        self._device = cfg["device"]
        self._dt = cfg["dt"]
        self._dynamic = dynamics
        self._agent_cost = agent_cost
        self._config_cost = config_cost

        self._agents = {name: i for i, name in enumerate(cfg["agents"])}
        self._nu_agent = cfg["mppi"]["nu"]
        self._nx_agent = cfg["mppi"]["nx"]
        self._num_agents = len(cfg["agents"])
        self._nu_sys = self._nu_agent * self._num_agents
        self._nx_sys = self._nx_agent * self._num_agents
        self._noise_sigma_sys = torch.block_diag(*([torch.tensor(cfg['mppi']['noise_sigma'], device=self._device)] * self._num_agents))
        self._u_min_sys = torch.cat([torch.tensor(cfg['mppi']['u_min'], device=self._device)] * self._num_agents)
        self._u_max_sys = torch.cat([torch.tensor(cfg['mppi']['u_max'], device=self._device)] * self._num_agents)

        self._action_sys = torch.zeros(self._nu_sys, device=self._device)
        
        # Create a new configuration dictionary
        mppi_cfg = self._update_config(cfg)
     
        self.mppi = MPPIPlanner(
            cfg=mppi_cfg["mppi"],
            nx=self._nx_sys,
            dynamics=self._step,
            running_cost=self._compute_running_cost,
        )
    
    def _update_config(self, cfg):

        # Create a new configuration dictionary
        new_config = {
            'nu': self._nu_sys,
            'noise_sigma': self._noise_sigma_sys.tolist(),
            'u_min': self._u_min_sys.tolist(),
            'u_max': self._u_max_sys.tolist(),
            'device': self._device
        }

        # Update the sys_cfg['mppi'] dictionary with the new values
        cfg['mppi'].update(new_config)
        # Remove entries from the dictionary to make MPPIPlanner happy
        cfg['mppi'].pop('nx', None)
        cfg['mppi'].pop('nu', None)

        return cfg
    
    def _step(self, system_state, action, t):
        # Reshape the system_state and action tensors
        reshaped_state = system_state.view(-1, self._num_agents, self._nx_agent)
        reshaped_action = action.view(-1, self._num_agents, self._nu_agent)
        
        new_states, new_actions = self._dynamic(reshaped_state, reshaped_action)
        
        new_states = new_states.view(-1, self._num_agents * self._nx_agent)
        new_actions = new_actions.view(-1, self._num_agents * self._nu_agent)

        return new_states, new_actions
    
    def _compute_running_cost(self, system_state):
        reshaped_state = system_state.view(-1, self._num_agents, self._nx_agent)
        reshaped_init_state = self.intial_state_tensor.view(-1, self._num_agents, self._nx_agent)
        
        agents_cost = torch.sum(self._agent_cost(reshaped_state), dim=1)

        conf_cost = self._config_cost(reshaped_state, reshaped_init_state)

        return conf_cost + agents_cost
    
    def make_plan(self, system_state):
        # Extract the states from the dictionary and convert them to tensors
        states = [torch.tensor(state) for state in system_state.values()]

        # Concatenate the states into a single tensor
        system_state_tensor = torch.cat(states, dim=0)

        # Save the initial state tensor for cost computation
        self.intial_state_tensor = system_state_tensor

        # Get the actions as a concatenated tensor
        self._action_sys = self.mppi.command(system_state_tensor)

        return None
    
    def zero_command(self, agent_name=None):
        actions = {}
        if agent_name is None:
            for agent_name in self._agents.keys():
                actions[agent_name] = torch.zeros(self._nu_agent, device=self._device)
        else:
            for name in agent_name:
                if name in self._agents:
                    actions[name] = torch.zeros(self._nu_agent, device=self._device)
                else:
                    raise ValueError(f"Agent name {name} not found")
        return actions
    
    def get_command(self, agent_name=None):
        actions = {}
        if agent_name is None:
            for i, agent_name in enumerate(self._agents.keys()):
                actions[agent_name] = self._action_sys[i*self._nu_agent:(i+1)*self._nu_agent]
        else:
            for name in agent_name:
                if name in self._agents:
                    i = self._agents[name]
                    actions[name] = self._action_sys[i*self._nu_agent:(i+1)*self._nu_agent]
                else:
                    raise ValueError(f"Agent name {name} not found")
        return actions