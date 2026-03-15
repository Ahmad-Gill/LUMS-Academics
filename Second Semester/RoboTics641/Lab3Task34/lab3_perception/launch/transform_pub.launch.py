from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.2',
            '--y', '0',
            '--z', '0.1',
            '--yaw', '0',
            '--pitch', '0',
            '--roll', '0',
            '--frame-id', 'virtual_robot',
            '--child-frame-id', 'lidar_sensor'
        ]
    )

    dynamic_tf = Node(
        package='lab3_perception',
        executable='dynamic_tf_broadcaster',
        output='screen'
    )

    return LaunchDescription([
        static_tf,
        dynamic_tf
    ])