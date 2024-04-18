import sys
sys.path.append('/root/dev/ia_mppi/ia_mppi/')
import torch
import os
from simulator import Simulator
from objective import RoboatObjective, SocialNavigationObjective
from dynamics import QuarterRoboatDynamics
from ia_mppi import IAMPPIPlanner
import yaml
from tqdm import tqdm
import copy
import time

# Load the config file
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_roboats.yaml"))

def run_point_robot_example():

    dynamics = QuarterRoboatDynamics(
        CONFIG
    )
    agent_cost = RoboatObjective(
        goals=torch.tensor([agent_info['initial_goal'] for agent_info in CONFIG['agents'].values()], device=CONFIG["device"]),
        device=CONFIG["device"]
    )
    configuration_cost = SocialNavigationObjective(
        device=CONFIG["device"]
    )
        
    simulator = Simulator(
        cfg=CONFIG,
        dynamics=dynamics.step
    )

    planner = IAMPPIPlanner(
        cfg=copy.deepcopy(CONFIG),
        dynamics=dynamics.step,
        agent_cost=copy.deepcopy(agent_cost),
        config_cost=copy.deepcopy(configuration_cost),
    )

    initial_action = planner.zero_command()
    observation = simulator.step(initial_action)

    for _ in tqdm(range(CONFIG['simulator']['steps'])):
        start_time = time.time()

        # print(observation['agent1'])

        planner.make_plan(observation)
        action = planner.get_command()
        # action = planner.zero_command()
        # action['agent0'] = torch.tensor([1.0, 1.5, 0., 0.], device=CONFIG["device"])
        plans = planner.get_planned_traj()
        simulator.plot_trajectories(plans)
        observation = simulator.step(action)

        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_time = CONFIG['dt'] - elapsed_time

        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    run_point_robot_example()