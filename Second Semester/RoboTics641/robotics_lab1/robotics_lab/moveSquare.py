import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math
import time

def main():
    rclpy.init()
    node = Node('move_square_node')
    pub = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)

    side_length = 2.0  # each side of the square in turtlesim units
    linear_speed = 1.0  # units/sec
    angular_speed = math.pi / 2  # 90 degrees in radians/sec
    turn_time = 1.57 / angular_speed  # time to turn 90 degrees

    move_cmd = Twist()
    turn_cmd = Twist()
    turn_cmd.angular.z = math.pi / 2  # 90 degrees turn

    for _ in range(4):
        # Move forward precise distance
        move_cmd.linear.x = linear_speed
        move_cmd.angular.z = 0.0
        start = time.time()
        while time.time() - start < side_length / linear_speed:
            pub.publish(move_cmd)
            rclpy.spin_once(node, timeout_sec=0.1)

        # Stop before turning
        move_cmd.linear.x = 0.0
        pub.publish(move_cmd)
        time.sleep(0.1)

        # Turn 90 degrees
        turn_cmd.linear.x = 0.0
        turn_cmd.angular.z = angular_speed
        start = time.time()
        while time.time() - start < math.pi/2 / angular_speed:
            pub.publish(turn_cmd)
            rclpy.spin_once(node, timeout_sec=0.1)

        # Stop after turn
        turn_cmd.angular.z = 0.0
        pub.publish(turn_cmd)
        time.sleep(0.1)

    node.get_logger().info("Turtle completed the square!")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
