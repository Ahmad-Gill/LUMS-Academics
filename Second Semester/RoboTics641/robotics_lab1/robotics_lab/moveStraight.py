import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from time import sleep

def main():
    rclpy.init()  # Initialize ROS 2

    node = Node('move_straight_node')  # Create a node named 'move_straight_node'

    publisher = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)  # Publisher to send velocity commands

    move_cmd = Twist()  # Create a Twist message

    move_cmd.linear.x = 1.0  # Set linear velocity (moving forward)
    move_cmd.angular.z = 0.0  # No rotation

    # Publish the velocity for 2 seconds (duration = distance / speed)
    publisher.publish(move_cmd)
    sleep(2)  # Move for 2 seconds

    # Stop the turtle after the movement
    stop_cmd = Twist()  # Create a new Twist message to stop the turtle
    publisher.publish(stop_cmd)  # Stop the turtle

    node.get_logger().info("Turtle moved straight for 2 seconds.")

    node.destroy_node()
    rclpy.shutdown()  # Shutdown ROS 2

if __name__ == '__main__':
    main()
