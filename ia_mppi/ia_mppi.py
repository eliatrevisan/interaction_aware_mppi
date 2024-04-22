from mppi_torch.mppi import MPPIPlanner
import torch
torch.set_grad_enabled(False)
import os
os.environ["TORCHDYNAMO_DYNAMIC_SHAPES"] = "1"



class IAMPPIPlanner:
    def __init__(self, cfg, dynamics, agent_cost, config_cost, ego_agent=None) -> None:
        self._device = cfg["device"]
        self._dt = torch.tensor(cfg["dt"], device=self._device)
        self._horizon = cfg["mppi"]["horizon"]
        self._dynamic = dynamics
        self._agent_cost = agent_cost
        self._config_cost = config_cost
        self._agents = {name: i for i, name in enumerate(cfg["agents"])}

        self._ego_agent = ego_agent
        if self._ego_agent is not None and self._ego_agent not in self._agents:
            raise ValueError("Ego agent is not in the agents dictionary.")
        
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
            dynamics=torch.compile(self._step),
            running_cost=torch.compile(self._compute_running_cost),
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
        cfg['mppi'].pop('horizon_cutoff', None)
        cfg['mppi'].pop('dt_cutoff', None)

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
        
        agents_cost = torch.sum(self._agent_cost.compute_running_cost(reshaped_state), dim=1)

        if self._num_agents <=1:
            conf_cost = torch.zeros(1, device=self._device)
        else:
            conf_cost = self._config_cost.compute_running_cost(reshaped_state, reshaped_init_state, self.t)
        self.t += self._dt

        return conf_cost + agents_cost
    
    def make_plan(self, system_state):
        # Extract the states from the dictionary and convert them to tensors
        states = [torch.tensor(state) for state in system_state.values()]

        # Concatenate the states into a single tensor
        system_state_tensor = torch.cat(states, dim=0)

        # Save the initial state tensor for cost computation
        self.intial_state_tensor = system_state_tensor
        # Reset timestep
        self.t = torch.tensor(0, device=self._device)

        # Get the actions as a concatenated tensor
        self._action_seq_sys = self.mppi.command(system_state_tensor)
        self._action_sys = self._action_seq_sys[0,:]
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
    
    def get_planned_traj(self, agent_name=None):
        
        trajectory_sys = []
        trajectory_sys.append(self.intial_state_tensor)
        for i in range(self._horizon):
            state, _ = self._step(trajectory_sys[-1], self._action_seq_sys[i], i)
            trajectory_sys.append(state.squeeze(0))
        # Throw away the initial state
        trajectory_sys = trajectory_sys[1:]
        # Convert trajectory_sys to a tensor
        trajectory_sys = torch.stack(trajectory_sys)

        trajectories = {}
        if agent_name is None:
            for i, agent_name in enumerate(self._agents.keys()):
                trajectories[agent_name] = trajectory_sys[:,i*self._nx_agent:(i+1)*self._nx_agent]
        else:
            for name in agent_name:
                if name in self._agents:
                    i = self._agents[name]
                    trajectories[name] = trajectory_sys[:,i*self._nx_agent:(i+1)*self._nx_agent]
                else:
                    raise ValueError(f"Agent name {name} not found")
        return trajectories
    
    def update_other_goals(self, system_state):
        # Get the current goals
        current_goals = self._agent_cost.get_goals()

        # Create a mask where all elements are True except for the row corresponding to the ego agent
        mask = torch.ones(current_goals.shape[0], dtype=torch.bool)
        mask[self._agents[self._ego_agent]] = False

        predicted_goals = self._compute_constvel_goals(system_state)

        # Update the goals of all agents except the ego agent
        current_goals[mask] = predicted_goals.float()[mask]

        # Set the new goals
        self._agent_cost.set_goals(current_goals)

        return None
    
    def _compute_constvel_goals(self, system_state):
        predicted_goals = []
        prediction_radius = 3.0

        for agent_name, state in system_state.items():
            position = state[:2]
            velocity = 1.0*state[3:5]
            terminal_state = position + velocity * self._dt * self._horizon

            # Compute the distance between the terminal state and the initial state
            distance = torch.norm(terminal_state - position)

            # If the distance is greater than the prediction radius, project the terminal state back
            if distance > prediction_radius:
                direction = (terminal_state - position) / distance
                terminal_state = position + direction * prediction_radius

            predicted_goals.append(terminal_state)

        return torch.stack(predicted_goals)
    
    def get_goals(self):
        return self._agent_cost.get_goals()
    
    def update_ego_goal(self, observation):
        current_ego_pos = observation[self._ego_agent][0:2]
        current_goals = self._agent_cost.get_goals()
        current_ego_goal = current_goals[self._agents[self._ego_agent]]

        # Compute the distance between the current ego position and the current ego goal
        distance = torch.norm(current_ego_pos - current_ego_goal)

        # If the distance is less than a threshold, update the ego goal
        if distance < 0.5:
            current_goals[self._agents[self._ego_agent]][1] = -current_ego_goal[1]
        
        self._agent_cost.set_goals(current_goals)
        return None