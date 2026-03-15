import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch turtlesim_node
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='turtlesim',
            output='screen'
        ),

        # Run keyboard teleop for Turtle 1
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop_turtle1',
            output='screen'
        ),

        # Run Turtle 2 chasing node
        Node(
            package='robotics_lab',
            executable='chase',  # Name from setup.py entry point
            name='turtle2_chaser',
            output='screen'
        )
    ])
