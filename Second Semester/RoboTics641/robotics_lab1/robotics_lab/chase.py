import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
import math

def main():
    rclpy.init()
    node = Node('turtle2_chaser')

    turtle1_pose = None
    turtle2_pose = None

    # Callback functions to update poses
    def turtle1_cb(msg):
        nonlocal turtle1_pose
        turtle1_pose = msg

    def turtle2_cb(msg):
        nonlocal turtle2_pose
        turtle2_pose = msg

    # Subscribers
    node.create_subscription(Pose, '/turtle1/pose', turtle1_cb, 10)
    node.create_subscription(Pose, '/turtle2/pose', turtle2_cb, 10)

    # Publisher
    pub = node.create_publisher(Twist, '/turtle2/cmd_vel', 10)

    node.get_logger().info("Turtle2 chaser node started.")

    # Control loop
    rate = node.create_rate(10)  # 10 Hz
    while rclpy.ok():
        rclpy.spin_once(node)

        if turtle1_pose is None or turtle2_pose is None:
            continue

        # Compute distance and heading
        dx = turtle1_pose.x - turtle2_pose.x
        dy = turtle1_pose.y - turtle2_pose.y
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)

        cmd = Twist()

        if distance < 0.5:
            # Stop when caught
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            node.get_logger().info("Turtle2 caught Turtle1! Stopping.")
        else:
            # Proportional control to chase
            cmd.linear.x = min(1.5 * distance, 2.0)
            cmd.angular.z = 4.0 * (angle_to_target - turtle2_pose.theta)

        pub.publish(cmd)
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
