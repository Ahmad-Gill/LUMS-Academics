from launch import LaunchDescription
from launch.actions import ExecuteProcess
import os

def generate_launch_description():

    turtlebot3_model = os.environ.get('TURTLEBOT3_MODEL', 'waffle')

    return LaunchDescription([

        # Launch empty Gazebo world
        ExecuteProcess(
            cmd=['ros2', 'run', 'gazebo_ros', 'gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),

        # Spawn TurtleBot3 Waffle
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                '-entity', 'turtlebot3',
                '-file', os.path.expanduser(f'~/ros2_ws/src/turtlebot3/turtlebot3_description/urdf/turtlebot3_{turtlebot3_model}.urdf')
            ],
            output='screen'
        ),

        # Run obstacle follower node
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'lab3_perception', 'turtlebot3_mapping'
            ],
            output='screen'
        )
    ])