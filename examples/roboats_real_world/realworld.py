import sys
sys.path.append('/root/dev/ros_ws/src')
import torch
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Point
from roboat_core.msg import Force
from nav_msgs.msg import Odometry
import tf
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation as R

class Realworld:
      def __init__(self, cfg, ego_agent) -> None:
            rospy.init_node('mppi_node', anonymous=True)
            self._device = cfg["device"]
            self._agents_indexes = {name: i for i, name in enumerate(cfg["agents"])}
            self._agents = {agent: None for agent in self._agents_indexes.keys()}
            self.force_cmd = Force()
            self.got_data = False
            self.pub_command = rospy.Publisher(f'{ego_agent}/mpc_force', Force, queue_size=1)
            self.pub_plans = rospy.Publisher(f'{ego_agent}/planned_trajectories', MarkerArray, queue_size=1)
            # Create a publisher for the goals
            self.goal_pub = rospy.Publisher(f'{ego_agent}/planned_goals', MarkerArray, queue_size=1)
            # Create a subscriber for each agent
            self._agent_subs = {}
            # Create a subscriber for each agent
            for agent_name in self._agents_indexes.keys():
                callback = self._create_callback(agent_name)
                self._agent_subs[agent_name] = rospy.Subscriber(f'{agent_name}/odometry/filtered', Odometry, callback, queue_size=1)



      def send_command(self, action):
            self.force_cmd.data = action.cpu().tolist()
            self.pub_command.publish(self.force_cmd)

      def publish_trajectories(self, trajectories, cost=None):
        marker_array = MarkerArray()

        if cost is not None:
            # Convert the cost to a numpy array if it's not already
            if isinstance(cost, torch.Tensor):
                cost = cost.cpu().numpy()

            # Normalize the cost to the range [0, 1]
            cost_min = np.min(cost)
            cost_range = np.max(cost) - cost_min
            normalized_cost = (cost - cost_min) / cost_range
        else:
            # Set a default color if cost is not provided
            default_color = ColorRGBA(0.0, 1.0, 0.0, 1.0)  # Green color

        for i, trajectory in enumerate(trajectories):
            # Convert the trajectory to a numpy array if it's not already
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.cpu().numpy()

            # Create a marker for the trajectory
            marker = Marker()
            marker.header.frame_id = "map"
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.id = i
            marker.scale.x = 0.02

            # Use the normalized cost to create a color gradient from green to red
            # or use the default color if cost is not provided
            marker.color = default_color if cost is None else ColorRGBA(normalized_cost[i], 1.0 - normalized_cost[i], 0.0, 1.0)

            for point in trajectory:
                p = Point()
                p.x = point[1]
                p.y = -point[0]
                marker.points.append(p)

            marker_array.markers.append(marker)

        # Publish the marker array
        self.pub_plans.publish(marker_array)



      def _create_callback(self, agent_name):
        def callback(data):
            state = torch.zeros(6, device=self._device)
            state[0] = -data.pose.pose.position.y
            state[1] = data.pose.pose.position.x

            # Orientation
            quaternion = (
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w
            )
            euler = torch.tensor(tf.transformations.euler_from_quaternion(quaternion))
            state[2] = torch.atan2(torch.sin(euler[2] + torch.pi/2), torch.cos(euler[2] + torch.pi/2))
            yaw = state[2] 

            # Create the rotation matrix
            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])

            # Get the agent's linear velocity in the body frame
            linear_velocity_body = np.array([data.twist.twist.linear.x, data.twist.twist.linear.y])

            # Transform the linear velocity to the world frame
            linear_velocity_world = np.dot(rotation_matrix, linear_velocity_body)

            # Assign the transformed linear velocities to the state
            state[3] = linear_velocity_world[0]
            state[4] = linear_velocity_world[1]

            # Angular velocity
            state[5] = data.twist.twist.angular.z

            self._agents[agent_name] = state
            self.got_data = all(self._agents[agent] is not None for agent in self._agents_indexes.keys())

        return callback

      def get_states(self):
        return self._agents
        
      def publish_goals(self, goals):
            # Create a MarkerArray message
            marker_array = MarkerArray()

            for i, goal in enumerate(goals):
                  # Create a Marker for each goal
                  marker = Marker()
                  marker.header.frame_id = "map"
                  marker.type = marker.SPHERE
                  marker.action = marker.ADD
                  marker.scale.x = 0.1
                  marker.scale.y = 0.1
                  marker.scale.z = 0.1
                  marker.color.a = 1.0
                  marker.color.r = 1.0
                  marker.color.g = 0.0
                  marker.color.b = 0.0

                  # Convert the goal to a Point and assign it to marker.pose.position
                  point = Point()
                  point.x = goal[1].item()
                  point.y = -goal[0].item()
                  point.z = 0  # assuming the goal is 2D
                  marker.pose.position = point

                  marker.id = i

                  # Add the Marker to the MarkerArray
                  marker_array.markers.append(marker)

            # Publish the MarkerArray
            self.goal_pub.publish(marker_array)