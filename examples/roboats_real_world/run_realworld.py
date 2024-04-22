import sys
sys.path.append('/root/dev/ia_mppi/ia_mppi/')
import torch
import os
from realworld import Realworld
from objective import RoboatObjective, SocialNavigationObjective
from dynamics import QuarterRoboatDynamics
from ia_mppi import IAMPPIPlanner
import yaml
import copy
import rospy
import time
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Run the node with the given agent name.')

# Add an argument for the agent name
parser.add_argument('--agent_name', type=str, default='roboat1', help='The name of the agent')

# Parse the arguments
args = parser.parse_args()

# Now you can use args.agent_name to get the name of the agent
print(f'Running control node with agent {args.agent_name}')


abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_roboats.yaml"))


class mppi_controller():
    def __init__(self):
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
        self.realworld = Realworld(
            cfg=copy.deepcopy(CONFIG),
            ego_agent=args.agent_name
        )
        print(f'{args.agent_name}')
        self.planner = IAMPPIPlanner(
            cfg=copy.deepcopy(CONFIG),
            dynamics=dynamics.step,
            agent_cost=copy.deepcopy(agent_cost),
            config_cost=copy.deepcopy(configuration_cost),
            ego_agent=args.agent_name
        )

    def control_robot(self):
        rate = rospy.Rate(10) # 10hz
        while not (self.realworld.got_data):
            print("not have data yet")
            rate.sleep()
        while not rospy.is_shutdown():
            observation = copy.deepcopy(self.realworld.get_states())
            # print(observation)

            self.planner.update_other_goals(observation)
            self.planner.update_ego_goal(observation)

            start_time = time.time()
            self.planner.make_plan(observation)
            end_time = time.time()

            elapsed_time = end_time - start_time

            action = self.planner.get_command([args.agent_name])
            # print(f"Elapsed time: {elapsed_time} seconds")

            self.realworld.send_command(action[args.agent_name])
            # self.realworld.send_command(torch.tensor([1,-1,0,0]))

            planned_traj = self.planner.get_planned_traj()
            # Extract the trajectories from the planned_traj dictionary
            trajectories = [traj for traj in planned_traj.values()]
            # Stack the trajectories into a single tensor
            stacked_trajectories = torch.stack(trajectories)
            self.realworld.publish_trajectories(stacked_trajectories)

            goals = self.planner.get_goals()
            self.realworld.publish_goals(goals)

            rate.sleep()
        

if __name__ == '__main__':
    controller_node = mppi_controller()
    try:
        controller_node.control_robot()
    except rospy.ROSInterruptException:
        pass