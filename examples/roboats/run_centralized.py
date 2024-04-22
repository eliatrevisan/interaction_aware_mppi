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
from torch.profiler import profile, record_function, ProfilerActivity

# Load the config file
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_roboats.yaml"))

def run_point_robot_example():

    dynamics = QuarterRoboatDynamics(
        cfg=CONFIG
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

        # with profile(with_stack=True, profile_memory=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        #         planner.make_plan(observation)
        # # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=50))
        # prof.export_stacks("/tmp/profiler_stacks.txt", "self_cpu_time_total")

        planner.make_plan(observation)

        # end_time = time.time()
        # print(f"Planning time: {end_time - start_time}")

        action = planner.get_command()
        plans = planner.get_planned_traj()
        simulator.plot_trajectories(plans)
        observation = simulator.step(action)

        end_time = time.time()
        elapsed_time = end_time - start_time
        sleep_time = CONFIG['dt'] - elapsed_time

        # if sleep_time > 0:
        #     time.sleep(sleep_time)


if __name__ == "__main__":
    run_point_robot_example()