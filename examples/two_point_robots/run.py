import torch
import os
from simulator import Simulator
from objective import OmnidirectionalPointRobotObjective
from dynamics import OmnidirectionalPointRobotDynamics
from mppi_torch.mppi import MPPIPlanner
import yaml
from tqdm import tqdm

abs_path = os.path.dirname(os.path.abspath(__file__))

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Load system config
    system_config = yaml.safe_load(open(f"{abs_path}/cfg_system.yaml"))

    for agent, agent_config in config.get('agents', {}).items():
        config['agents'][agent].update(system_config)

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


class Agent:
    def __init__(self, config):
        self.cfg = config
        self.simulator = Simulator(
            cfg=self.cfg["simulator"],
            dt=self.cfg["dt"],
            goal=self.cfg["goal"],
            device=self.cfg["device"],
        )
        self.dynamics = OmnidirectionalPointRobotDynamics(
            dt=self.cfg["dt"], device=self.cfg["device"]
        )
        self.objective = OmnidirectionalPointRobotObjective(goal=self.cfg["goal"], device=self.cfg["device"])

def run_point_robot_example():

    agents = {}

    for agent_name, agent_config in CONFIG.get('agents', {}).items():
        agent = Agent(agent_config)
        agents[agent_name] = agent

    # Now you have a dictionary of Agent objects. You can access each agent by its name.

    initial_action = torch.zeros(3, device=CONFIG["device"])
    observation = agents["agent1"].simulator.step(initial_action)

    for _ in tqdm(range(CONFIG["steps"])):
        action = agents[0].planner.command(observation)

        observation = agents[0].simulator.step(action)


if __name__ == "__main__":
    run_point_robot_example()