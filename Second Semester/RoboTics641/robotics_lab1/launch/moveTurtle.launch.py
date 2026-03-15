import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start the move_straight node
        Node(
            package='robotics_lab',            # The package where the move_straight.py file is located
            executable='moveStraight', # The Python file to run (moveStraight.py)
            name='move_straight_node',
            output='screen',
        ),

        # Start the move_to_goal node
        Node(
            package='robotics_lab',            # The package where the move_to_goal.py file is located
            executable='moveToGoal', # The Python file to run (move_to_goal.py)
            name='move_to_goal_node',
            output='screen',
        )
    ])
