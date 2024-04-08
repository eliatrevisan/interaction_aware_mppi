import sys
sys.path.append('/root/dev/ia_mppi/ia_mppi/')
import torch
import os
from realworld import Realworld
from objective import JackalRobotObjective, SocialNavigationObjective
from dynamics import JackalDynamics
from ia_mppi import IAMPPIPlanner
import yaml
import copy
import rospy
import time


abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/cfg_jackal.yaml"))


class mppi_controller():
    def __init__(self):
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
        self.planner = IAMPPIPlanner(
            cfg=copy.deepcopy(CONFIG),
            dynamics=dynamics.step,
            agent_cost=copy.deepcopy(agent_cost),
            config_cost=copy.deepcopy(configuration_cost),
            ego_agent='jackal0'
        )
        self.realworld = Realworld(
            cfg=copy.deepcopy(CONFIG),
        )

    def control_robot(self):
        rate = rospy.Rate(10) # 10hz
        while not (self.realworld.got_data):
            print("not have data yet")
            rate.sleep()
        while not rospy.is_shutdown():
            observation = copy.deepcopy(self.realworld.get_states())
            print(observation['agent0'])

            self.planner.update_other_goals(observation)
            self.planner.update_ego_goal(observation)

            start_time = time.time()
            self.planner.make_plan(observation)
            end_time = time.time()

            elapsed_time = end_time - start_time

            action = self.planner.get_command(['jackal0'])
            # print(f"Elapsed time: {elapsed_time} seconds")

            self.realworld.send_command(action['jackal0'])

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