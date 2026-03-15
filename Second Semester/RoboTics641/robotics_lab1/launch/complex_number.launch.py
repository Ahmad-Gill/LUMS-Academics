import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generateLaunchDescription():
    return LaunchDescription([
        DeclareLaunchArgument('real', defaultValue='0.0' ),
        DeclareLaunchArgument('imaginary', defaultValue='0.0'),
        Node(
            package='robotics_lab1',            # The package where the node is located
            executable='publisher',    # The name of the node executable (publisher.py)
            name='complexNumberPublisher',
            output='screen',   #all the erros outputs show here 
            parameters=[{
                'real': LaunchConfiguration('real'),        # Pass the 'real' argument to the node
                'imaginary': LaunchConfiguration('imaginary')  # Pass the 'imaginary' argument to the node
            }]
        ),

        # Start the 'subscriber' node
        Node(
            package='robotics_lab1',            # The package where the subscriber node is located
            executable='subscriber',   # The name of the node executable (subscriber.py)
            name='complexNumberSubscriber',
            output='screen'
        )
    ])
