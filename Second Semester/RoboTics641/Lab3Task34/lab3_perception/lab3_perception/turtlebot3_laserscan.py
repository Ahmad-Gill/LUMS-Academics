import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math

class LaserScanNode(Node):
    def __init__(self):
        super().__init__('turtlebot3_laserscan')

        # QoS for Gazebo Lidar (Best Effort)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos)

        self.get_logger().info("Waiting for Lidar data...")

    def scan_callback(self, msg):
        valid_count = 0
        closest_range = float('inf')
        closest_angle = 0.0
        farthest_range = 0.0
        farthest_angle = 0.0

        # loop over all readings
        for i, r in enumerate(msg.ranges):
            # ignore invalid values (inf or nan)
            if math.isinf(r) or math.isnan(r):
                continue

            valid_count += 1
            angle = msg.angle_min + (i * msg.angle_increment)

            # closest point
            if r < closest_range:
                closest_range = r
                closest_angle = angle

            # farthest point
            if r > farthest_range:
                farthest_range = r
                farthest_angle = angle

        # Print info
        self.get_logger().info("\n================ LIDAR INFO ================")
        self.get_logger().info(f"Valid readings: {valid_count}")
        self.get_logger().info(f"Closest point:  {closest_range:.2f} m  at angle {math.degrees(closest_angle):.2f} deg")
        self.get_logger().info(f"Farthest point: {farthest_range:.2f} m  at angle {math.degrees(farthest_angle):.2f} deg")
        self.get_logger().info("============================================")


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
