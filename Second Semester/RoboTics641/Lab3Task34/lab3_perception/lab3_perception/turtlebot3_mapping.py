import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy

class ObstacleFollower(Node):

    def __init__(self):
        super().__init__('obstacle_follower')

        # Publisher to /cmd_vel
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # QoS for LaserScan
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # Subscriber to /scan
        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos
        )

        # Distance to stop near obstacle
        self.stop_distance = 0.3  # meters

    def scan_callback(self, msg):
        twist = Twist()

        # Take front laser readings only
        mid = len(msg.ranges) // 2
        front_ranges = msg.ranges[mid-10: mid+10]  # ~20 degrees in front

        # Filter valid readings
        valid_ranges = [r for r in front_ranges if not math.isinf(r) and not math.isnan(r)]
        if not valid_ranges:
            return

        min_dist = min(valid_ranges)

        # Move forward until close to obstacle
        if min_dist > self.stop_distance:
            twist.linear.x = 0.2  # forward speed
            twist.angular.z = 0.0
        else:
            twist.linear.x = 0.0  # stop
            twist.angular.z = 0.0

        self.pub.publish(twist)


def main():
    rclpy.init()
    node = ObstacleFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()