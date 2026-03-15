import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
import time

def main():
    rclpy.init()
    node = Node('circle_motion_rviz')

    # Publishers
    marker_pub = node.create_publisher(Marker, '/robot_marker', 10)
    path_pub = node.create_publisher(Marker, '/robot_path', 10)
    cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)

    # Circle parameters
    radius = 5.0
    angular_speed = 0.2
    linear_speed = radius * angular_speed

    start_time = time.time()
    path_points = []  # store points for the trail

    cmd = Twist()
    cmd.linear.x = linear_speed
    cmd.angular.z = angular_speed

    def timer_callback():
        t = time.time() - start_time

        # Compute robot position
        x = radius * math.cos(angular_speed * t)
        y = radius * math.sin(angular_speed * t)
        yaw = angular_speed * t + math.pi/2

        # Publish velocity (optional)
        cmd_pub.publish(cmd)

        # Cube marker for robot
        cube_marker = Marker()
        cube_marker.header.frame_id = 'odom'
        cube_marker.header.stamp = node.get_clock().now().to_msg()
        cube_marker.ns = "robot"
        cube_marker.id = 0
        cube_marker.type = Marker.CUBE
        cube_marker.action = Marker.ADD
        cube_marker.pose.position.x = x
        cube_marker.pose.position.y = y
        cube_marker.pose.position.z = 0.1
        cube_marker.pose.orientation.w = 1.0
        cube_marker.scale.x = 0.3
        cube_marker.scale.y = 0.3
        cube_marker.scale.z = 0.3
        cube_marker.color.r = 0.0
        cube_marker.color.g = 0.0
        cube_marker.color.b = 1.0
        cube_marker.color.a = 1.0
        marker_pub.publish(cube_marker)

        # Add current point to path
        point = Point()
        point.x = x
        point.y = y
        point.z = 0.1
        path_points.append(point)

        # Line strip marker for path
        path_marker = Marker()
        path_marker.header.frame_id = 'odom'
        path_marker.header.stamp = node.get_clock().now().to_msg()
        path_marker.ns = "path"
        path_marker.id = 1
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05  # line width
        path_marker.color.r = 1.0
        path_marker.color.g = 0.0
        path_marker.color.b = 0.0
        path_marker.color.a = 1.0
        path_marker.points = path_points
        path_pub.publish(path_marker)

        # Log
        node.get_logger().info(f'Robot at x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}')

    node.create_timer(0.1, timer_callback)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
