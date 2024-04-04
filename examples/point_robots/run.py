import sys
sys.path.append('/root/dev/ia_mppi/ia_mppi/')
import torch
import os
from simulator import Simulator
from objective import OmnidirectionalPointRobotObjective, SocialNavigationObjective
from dynamics import OmnidirectionalPointRobotDynamics
from ia_mppi import IAMPPIPlanner
import yaml
from tqdm import tqdm

# Load the config file
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_point_robot.yaml"))

def run_point_robot_example():

    dynamics = OmnidirectionalPointRobotDynamics(
        dt=CONFIG["dt"], device=CONFIG["device"]
    )
    agent_cost = OmnidirectionalPointRobotObjective(
        goals=torch.tensor([agent_info['initial_goal'] for agent_info in CONFIG['agents'].values()], device=CONFIG["device"]),
        device=CONFIG["device"]
    )
    configuration_cost = SocialNavigationObjective(
        device=CONFIG["device"]
    )
        
    simulator = Simulator(cfg=CONFIG)
    planner = IAMPPIPlanner(
        cfg=CONFIG,
        dynamics=dynamics.step,
        agent_cost=agent_cost.compute_running_cost,
        config_cost=configuration_cost.compute_running_cost
    )

    initial_action = planner.zero_command()
    # initial_action['agent1'] = torch.tensor([1.0, 0.0, 0.0], device=CONFIG["device"])
    # initial_action['agent2'] = torch.tensor([1.0, 0.0, 0.0], device=CONFIG["device"])
    observation = simulator.step(initial_action)

    for _ in tqdm(range(CONFIG['simulator']['steps'])):

        planner.make_plan(observation)
        action = planner.get_command()
        observation = simulator.step(action)


if __name__ == "__main__":
    run_point_robot_example()