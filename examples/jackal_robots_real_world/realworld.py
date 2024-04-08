import torch
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Joy
from derived_object_msgs.msg import ObjectArray
import tf
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation as R

class Realworld:
      def __init__(self, cfg) -> None:
            rospy.init_node('mppi_node', anonymous=True)
            self._device = cfg["device"]
            self._agents_indexes = {name: i for i, name in enumerate(cfg["agents"])}
            self._agents = {}
            self.vel_cmd = Twist()
            self.vel_cmd.linear.x = 0.0
            self.vel_cmd.angular.z = 0.0
            self.got_data = False
            self._enable_output_ = False
            self.pub_command = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            self.pub_plans = rospy.Publisher('/planned_trajectories', MarkerArray, queue_size=1)
            # Create a publisher for the goals
            self.goal_pub = rospy.Publisher('/planned_goals', MarkerArray, queue_size=1)
            self._agentssub = rospy.Subscriber("/vicon_util/dynamic_objects", ObjectArray, self._agentsCallback, queue_size=1)
            self._bluetooth_sub_ = rospy.Subscriber("/bluetooth_teleop/joy", Joy,
                                            lambda msg: self.bluetooth_callback(msg), queue_size=1)

      def bluetooth_callback(self,msg):
            if msg.axes[2] <0.9:
                  self._enable_output_= 1
            if msg.axes[5] <0.9:
                  self._enable_output_= 0

      def send_command(self, action):
         self.vel_cmd.linear.x = action[0].cpu()
         self.vel_cmd.angular.z = action[1].cpu()
         if self._enable_output_:
            self.pub_command.publish(self.vel_cmd)

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
                p.x = point[0]
                p.y = point[1]
                marker.points.append(p)

            marker_array.markers.append(marker)

        # Publish the marker array
        self.pub_plans.publish(marker_array)



      def _agentsCallback(self, msg):

        for agent_name in self._agents_indexes:
            state = torch.zeros(6, device=self._device)
            agent_object = next((obj for obj in msg.objects if obj.id == self._agents_indexes[agent_name]), None)
            state[0] = agent_object.pose.position.x
            state[1] = agent_object.pose.position.y

            # Orientation
            quaternion = (
                  agent_object.pose.orientation.x,
                  agent_object.pose.orientation.y,
                  agent_object.pose.orientation.z,
                  agent_object.pose.orientation.w
            )
            euler = tf.transformations.euler_from_quaternion(quaternion)
            state[2] = euler[2]
            yaw = euler[2]

            # Create the rotation matrix
            rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
            ])

            # Get the agent's linear velocity in the body frame
            linear_velocity_body = np.array([agent_object.twist.linear.x, agent_object.twist.linear.y])

            # Transform the linear velocity to the world frame
            linear_velocity_world = np.dot(rotation_matrix, linear_velocity_body)

            # Assign the transformed linear velocities to the state
            state[3] = linear_velocity_world[0]
            state[4] = linear_velocity_world[1]

            # # Save orientation as the direction of the velocity vector
            # state[2] = torch.arctan2(state[4], state[3])

            # Angular velocity
            state[5] = agent_object.twist.angular.z

            self._agents[agent_name] = state

        self.got_data = True

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
                  point.x = goal[0].item()
                  point.y = goal[1].item()
                  point.z = 0  # assuming the goal is 2D
                  marker.pose.position = point

                  marker.id = i

                  # Add the Marker to the MarkerArray
                  marker_array.markers.append(marker)

            # Publish the MarkerArray
            self.goal_pub.publish(marker_array)