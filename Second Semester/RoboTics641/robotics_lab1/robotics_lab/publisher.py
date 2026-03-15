import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray 

def main():
    rclpy.init()  # Initialize ROS 2

    node = Node('simplePublisher')  # Create a node

    # Declare parameters (real, imaginary)
    node.declare_parameter('real', 0.0)
    node.declare_parameter('imaginary', 0.0)

    # Read the parameters
    real = node.get_parameter('real').get_parameter_value().double_value
    imaginary = node.get_parameter('imaginary').get_parameter_value().double_value

    # Create a publisher for Float64MultiArray message type
    publisher = node.create_publisher(Float64MultiArray, '/complexnumbers', 10)

    # Create the message to publish (real and imaginary in an array)
    msg = Float64MultiArray()
    msg.data = [real, imaginary]  # Store real and imaginary in the data array

    # Publish the message every second
    while rclpy.ok():
        publisher.publish(msg)  # Publish the message
        node.get_logger().info(f'Publishing: {msg.data[0]} + {msg.data[1]}j')  # Log the published message
        rclpy.spin_once(node)  # Spin once to process callbacks

    node.destroy_node()  # Clean up the node
    rclpy.shutdown()  # Shutdown ROS 2

if __name__ == '__main__':
    main()
