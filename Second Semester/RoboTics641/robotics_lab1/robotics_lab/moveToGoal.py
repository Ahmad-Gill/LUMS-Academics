import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math

def main():
    rclpy.init()  # Initialize ROS 2

    node = Node('move_to_goal_node')  # Create a node named 'move_to_goal_node'

    publisher = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)  # Publisher to send velocity commands
    current_pose = Pose()  # Create a Pose message to store current position

    goal_x = 1.0  # Goal X position
    goal_y = 1.0  # Goal Y position

    def pose_callback(msg):
        nonlocal current_pose
        current_pose = msg  # Update current pose with the feedback

        dx = goal_x - current_pose.x
        dy = goal_y - current_pose.y

        distance = math.sqrt(dx**2 + dy**2)

        move_cmd = Twist()

        # If the turtle is close to the goal, stop moving
        if distance < 0.1:
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.0
            node.get_logger().info("Goal reached!")
        else:
            move_cmd.linear.x = 1.0  # Move forward with speed 1
            move_cmd.angular.z = 4.0 * math.atan2(dy, dx)  # Rotate towards the goal

        publisher.publish(move_cmd)  # Publish the movement command

    # Subscribe to the /turtle1/pose topic to get feedback on the turtle's position
    node.create_subscription(Pose, '/turtle1/pose', pose_callback, 10)

    rclpy.spin(node)  # Keep spinning to process callbacks

    node.destroy_node()  # Clean up the node
    rclpy.shutdown()  # Shutdown ROS 2

if __name__ == '__main__':
    main()
