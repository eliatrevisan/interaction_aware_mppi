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
    planners = {}
    for agent_name, agent_info in CONFIG['agents'].items():
        planners[agent_name] = IAMPPIPlanner(
            cfg=copy.deepcopy(CONFIG),
            dynamics=dynamics.step,
            agent_cost=copy.deepcopy(agent_cost),
            config_cost=copy.deepcopy(configuration_cost),
            ego_agent=agent_name
        )
    initial_actions = {}
    for agent_name, planner in planners.items():
        initial_actions[agent_name] = planner.zero_command([agent_name])[agent_name]
    observation = simulator.step(initial_actions)

    for _ in tqdm(range(CONFIG['simulator']['steps'])):

        actions = {}
        all_plans = {}
        i = 0
        for agent_name, planner in planners.items():
            planner.update_other_goals(observation)
            planner.make_plan(observation)
            plans = planner.get_planned_traj()
            for key, value in plans.items():
                all_plans[f"{key}_{i}"] = value
            actions[agent_name] = planner.get_command([agent_name])[agent_name]
            i += 1
        simulator.plot_trajectories(all_plans)
        observation = simulator.step(actions)


if __name__ == "__main__":
    run_point_robot_example()