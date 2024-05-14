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
import copy

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
    
    for i in range(10):
        cfg = copy.deepcopy(CONFIG)
        cfg['mppi']['seed_val'] = i
        planner = IAMPPIPlanner(
            cfg=cfg,
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
        simulator.reset()


if __name__ == "__main__":
    run_point_robot_example()