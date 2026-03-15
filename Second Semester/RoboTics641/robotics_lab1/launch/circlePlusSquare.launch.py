import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch the turtlesim_node
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='turtlesim',
            output='screen',
        ),

        # Start the move_square node for turtle1
        Node(
            package='robotics_lab',  # your package
            executable='move_square',  # moveSquare.py
            name='move_square_node',
            output='screen',
        ),

        # Start the move_circle node for turtle2
        Node(
            package='robotics_lab',
            executable='move_circle',  # moveCircle.py
            name='move_circle_node',
            output='screen',
        )
    ])
