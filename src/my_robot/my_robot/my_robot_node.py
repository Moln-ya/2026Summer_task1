#python
import rclpy

#Node class
from rclpy.node import Node

class MyRobotNode(Node):

	def init(self):
		super().init('my_robot_node')
		self.timer = self.create_timer(
			1.0,
			self.timer_callback
		)
		self.get_logger().info(
			'My robot node started'
		)
	def timer_callback(self):

		self.get_logger().info(
			'Hello ROS2'
		)
def main(args=None):
	rclpy.init(args=args)
	node = MyRobotNode()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
if name == 'main':
	main()
