import rospy
from std_msgs.msg import Bool
import pymsgbox

class Test_Caught(object):

	def __init__(self, num_robbers):
		rospy.init_node('test_caught')
		rospy.Subscriber('/caught_zhora', Bool, self.zhora_callback)
		self.pub_zhora = rospy.Publisher('/caught_confirm_zhora', Bool, queue_size=10)
		self.num_robbers = num_robbers
		print("Test Caught Ready!")
		rospy.spin()

	def zhora_callback(self, msg):
		if msg.data == True:

			res = pymsgbox.confirm("Did I catch zhora?" , title="Robber Caught?", buttons=["Yes", "No"])
			if res == "Yes":
				msg = Bool()
				msg.data = True
				self.pub_zhora.publish(msg)
				rospy.signal_shutdown("Zhora Caught!")


if __name__ == '__main__':
    a = Test_Caught(1)
