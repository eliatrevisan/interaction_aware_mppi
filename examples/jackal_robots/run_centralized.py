import torch
import os
from simulator import Simulator
from objective import JackalRobotObjective, SocialNavigationObjective
from dynamics import JackalDynamics
from ia_mppi.ia_mppi import IAMPPIPlanner
import yaml
from tqdm import tqdm
import copy

# Load the config file
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_jackals.yaml"))

def run_point_robot_example():

    dynamics = JackalDynamics(
        dt=CONFIG["dt"], device=CONFIG["device"]
    )
    agent_cost = JackalRobotObjective(
        goals=torch.tensor([agent_info['initial_goal'] for agent_info in CONFIG['agents'].values()], device=CONFIG["device"]),
        device=CONFIG["device"]
    )
    configuration_cost = SocialNavigationObjective(
        device=CONFIG["device"]
    )
        
    simulator = Simulator(cfg=CONFIG)
    planner = IAMPPIPlanner(
        cfg=copy.deepcopy(CONFIG),
        dynamics=dynamics.step,
        agent_cost=copy.deepcopy(agent_cost),
        config_cost=copy.deepcopy(configuration_cost),
    )

    initial_action = planner.zero_command()
    observation = simulator.step(initial_action)

    for _ in tqdm(range(CONFIG['simulator']['steps'])):

        planner.make_plan(observation)
        action = planner.get_command()
        plans = planner.get_planned_traj()
        simulator.plot_trajectories(plans)
        observation = simulator.step(action)


if __name__ == "__main__":
    run_point_robot_example()