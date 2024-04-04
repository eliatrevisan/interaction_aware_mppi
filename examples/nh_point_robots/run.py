import sys
sys.path.append('/root/dev/ia_mppi/ia_mppi/')
import torch
import os
from simulator import Simulator
from objective import OmnidirectionalPointRobotObjective, SocialNavigationObjective
from dynamics import OmnidirectionalPointRobotDynamics
from nh_ia_mppi import IAMPPIPlanner
import yaml
from tqdm import tqdm

abs_path = os.path.dirname(os.path.abspath(__file__))

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load system config
    config['system'] = yaml.safe_load(open(f"{abs_path}/cfg_system.yaml"))

    # for agent, agent_config in config.get('agents', {}).items():
    #     config['agents'][agent].update(config['system'])

    # Check if there are any other yaml files to import in the 'agents' section
    for agent, agent_config in config.get('agents', {}).items():
        if 'config' in agent_config and agent_config['config'].endswith('.yaml'):
            # Load the agent-specific config and merge it into the main config
            agent_specific_config = load_config(f"{abs_path}/{agent_config['config']}")
            # config['agents'][agent].update(agent_specific_config)
            deep_update(config['agents'][agent], agent_specific_config)

    return config

def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            # Get node or create one
            node = source.setdefault(key, {})
            deep_update(node, value)
        else:
            source[key] = value
    return source

# Load the main config file
CONFIG = load_config(f"{abs_path}/cfg_agents.yaml")


class PointRobotAgent:
    def __init__(self, agent_config, system_config):
        self.agent_cfg = agent_config
        self.sys_cfg = system_config
        self.dynamics = OmnidirectionalPointRobotDynamics(
            dt=self.sys_cfg["dt"], device=self.sys_cfg["device"]
        )
        self.objective = OmnidirectionalPointRobotObjective(goal=self.agent_cfg["initial_goal"], device=self.sys_cfg["device"])

def run_point_robot_example():

    agents = {}

    for agent_name, agent_config in CONFIG.get('agents', {}).items():
        agent = PointRobotAgent(agent_config, CONFIG["system"])
        agents[agent_name] = agent

    # Now you have a dictionary of Agent objects. You can access each agent by its name.
        
    configuration_cost = SocialNavigationObjective()
        
    simulator = Simulator(agents_cfg=CONFIG["agents"], sys_cfg=CONFIG["system"])
    planner = IAMPPIPlanner(agents, configuration_cost.compute_configuration_cost, sys_cfg=CONFIG["system"])

    initial_action = planner.zero_command()
    observation = simulator.step(initial_action)

    for _ in tqdm(range(CONFIG['system']['simulator']['steps'])):

        action = planner.command(observation)
        observation = simulator.step(action)


if __name__ == "__main__":
    run_point_robot_example()