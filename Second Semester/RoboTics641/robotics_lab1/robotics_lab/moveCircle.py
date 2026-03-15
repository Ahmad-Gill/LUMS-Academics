import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn
from time import sleep

def main():
    rclpy.init()
    node = Node('move_circle_node')

    # Spawn turtle2
    client = node.create_client(Spawn, 'spawn')
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for spawn service...')

    request = Spawn.Request()
    request.x = 5.0
    request.y = 5.0
    request.theta = 0.0
    request.name = 'turtle2'

    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    node.get_logger().info(f"Turtle spawned: {future.result().name}")

    # Publisher to move turtle2
    pub = node.create_publisher(Twist, '/turtle2/cmd_vel', 10)
    move_cmd = Twist()
    move_cmd.linear.x = 2.0
    move_cmd.angular.z = 2.0

    # Move in circle for 10 seconds
    for _ in range(10):
        pub.publish(move_cmd)
        sleep(1)

    # Stop turtle
    pub.publish(Twist())
    node.get_logger().info("Turtle2 finished circle.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
