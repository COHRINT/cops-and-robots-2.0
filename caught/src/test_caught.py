#!/usr/bin/env python
""" Simple interface to test caught_robber.py
subscribes to the /caught topic and publishes to
/caught_confirm """


__author__ = "LT"
__copyright__ = "Copyright 2017, Cohrint"
__license__ = "GPL"
__version__ = "1.3"
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Stable"

import rospy
from caught.msg import Caught
import pymsgbox

DEFAULT_NUM_ROBBERS = 1

class Test_Caught(object):

	def __init__(self):
		rospy.init_node('test_caught')
		rospy.Subscriber('/caught', Caught, self.robber_callback)
		self.pub = rospy.Publisher('/caught_confirm', Caught, queue_size=10)
		self.num_robbers = rospy.get_param('~num_robbers', DEFAULT_NUM_ROBBERS)
		rospy.loginfo("Test Caught Ready!")
		rospy.spin()

	def robber_callback(self, msg):
		msg_res = Caught()
		msg_res.robber = msg.robber
		res = pymsgbox.confirm("Did I catch "+ msg.robber.capitalize() +" ?" , title="Robber Caught?", buttons=["Yes", "No"])
		if res == "Yes":
			rospy.loginfo("Test Caught publishing " + msg.robber.capitalize() + " as caught!")
			msg_res.confirm = True
			self.num_robbers -= 1
		else:
			msg_res.confirm = False
		self.pub.publish(msg_res)
		if self.num_robbers == 0:
			rospy.loginfo("All Robbers Caught!")
			rospy.signal_shutdown("All Robbers Caught!")



if __name__ == '__main__':
    a = Test_Caught()
