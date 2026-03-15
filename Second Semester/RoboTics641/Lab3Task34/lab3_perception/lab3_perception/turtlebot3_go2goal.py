import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose, PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import math

def main():
    rclpy.init()
    node = Node('turtlebot3_go2goal')

    # Publishers
    cmd_pub = node.create_publisher(Twist, '/cmd_vel', 10)
    odom_pub = node.create_publisher(Odometry, '/odom', 10)
    path_pub = node.create_publisher(Marker, '/path_marker', 10)
    start_marker_pub = node.create_publisher(Marker, '/start_marker', 10)
    goal_marker_pub = node.create_publisher(Marker, '/goal_marker', 10)

    # Start and Goal
    x = 0.0
    y = 0.0
    theta = 0.0
    goal_x = 5.0
    goal_y = 5.0

    dt = 0.1
    linear_speed = 0.5
    angular_speed = 1.0

    # Path marker
    path_marker = Marker()
    path_marker.header.frame_id = 'odom'
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    path_marker.scale.x = 0.05
    path_marker.color.r = 1.0
    path_marker.color.g = 0.0
    path_marker.color.b = 0.0
    path_marker.color.a = 1.0
    path_marker.points = []

    # Start marker
    start_marker = Marker()
    start_marker.header.frame_id = 'odom'
    start_marker.type = Marker.SPHERE
    start_marker.action = Marker.ADD
    start_marker.pose.position.x = x
    start_marker.pose.position.y = y
    start_marker.pose.position.z = 0.1
    start_marker.scale.x = 0.2
    start_marker.scale.y = 0.2
    start_marker.scale.z = 0.2
    start_marker.color.r = 0.0
    start_marker.color.g = 1.0
    start_marker.color.b = 0.0
    start_marker.color.a = 1.0

    # Goal marker
    goal_marker = Marker()
    goal_marker.header.frame_id = 'odom'
    goal_marker.type = Marker.SPHERE
    goal_marker.action = Marker.ADD
    goal_marker.pose.position.x = goal_x
    goal_marker.pose.position.y = goal_y
    goal_marker.pose.position.z = 0.1
    goal_marker.scale.x = 0.2
    goal_marker.scale.y = 0.2
    goal_marker.scale.z = 0.2
    goal_marker.color.r = 1.0
    goal_marker.color.g = 0.0
    goal_marker.color.b = 0.0
    goal_marker.color.a = 1.0

    # Publish start and goal once
    start_marker_pub.publish(start_marker)
    goal_marker_pub.publish(goal_marker)

    def timer_callback():
        nonlocal x, y, theta

        # Compute distance and angle to goal
        dx = goal_x - x
        dy = goal_y - y
        distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)

        # Simple proportional control
        linear = linear_speed if distance > 0.05 else 0.0
        angular = angular_speed * (angle_to_goal - theta)

        # Update robot pose (simulate odometry)
        x += linear * math.cos(theta) * dt
        y += linear * math.sin(theta) * dt
        theta += angular * dt

        # Publish cmd_vel (optional)
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        cmd_pub.publish(cmd)

        # Publish simulated odometry
        odom = Odometry()
        odom.header.stamp = node.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(theta / 2.0)
        odom_pub.publish(odom)

        # Update path
        point = Point()
        point.x = x
        point.y = y
        point.z = 0.0
        path_marker.points.append(point)
        path_marker.header.stamp = node.get_clock().now().to_msg()
        path_pub.publish(path_marker)

        if distance < 0.05:
            node.get_logger().info('Goal Reached!')
            rclpy.shutdown()

    node.create_timer(dt, timer_callback)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
