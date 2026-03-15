from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os

def generate_launch_description():
    # Path to TurtleBot3 URDF
    turtlebot3_model_path = os.path.join(
        os.getenv('HOME'), 'ros2_ws', 'src', 'turtlebot3', 'turtlebot3_description', 'urdf', 'turtlebot3_burger.urdf'
    )

    return LaunchDescription([

        # 1️⃣ Launch Gazebo empty world
        ExecuteProcess(
            cmd=['gazebo', '--verbose', 'empty_world.sdf', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),

        # 2️⃣ Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(turtlebot3_model_path).read()}]
        ),

        # 3️⃣ Obstacle follower node
        Node(
            package='lab3_perception',
            executable='turtlebot3_mapping',  # your obstacle_follower.py code
            name='obstacle_follower',
            output='screen'
        ),

    ])