import rospy
from std_msgs.msg import Bool

class Test_Caught(object):

	def __init__(self, num_robbers):
		rospy.init_node('test_caught')
		rospy.Subscriber('/caught_zhora', Bool, self.caught_callback)
		self.num_robbers = num_robbers
		print("Test Caught Ready!")
		rospy.spin()

	def caught_callback(self, msg):
		if msg.data == True:
			print("Zhora Caught!")
			self.num_robbers -= 1
		if self.num_robbers == 0:
			print("All Robbers Caught")
			rospy.signal_shutdown("All Robbers Caught!")

if __name__ == '__main__':
    a = Test_Caught(1)
