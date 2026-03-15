#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler


def main(args=None):
    rclpy.init(args=args)

    # Create node
    node = Node('dynamic_tf_broadcaster')

    # Create broadcaster
    broadcaster = TransformBroadcaster(node)

    start_time = node.get_clock().now()

    def timer_callback():
        t = TransformStamped()

        now = node.get_clock().now()
        time_sec = (now - start_time).nanoseconds * 1e-9

        # Circle motion
        radius = 2.0
        angular_speed = 0.5

        x = radius * math.cos(angular_speed * time_sec)
        y = radius * math.sin(angular_speed * time_sec)

        t.header.stamp = now.to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'virtual_robot'

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        # Orientation (facing forward)
        yaw = angular_speed * time_sec
        q = quaternion_from_euler(0, 0, yaw)

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        broadcaster.sendTransform(t)

    # Run at 10 Hz
    node.create_timer(0.1, timer_callback)

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()