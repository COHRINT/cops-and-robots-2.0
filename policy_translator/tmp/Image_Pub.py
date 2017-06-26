import rospy
from sensor_msgs.msg import Image
import os

import cv2
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('interface_map')
pub = rospy.Publisher('/interface_map', Image, queue_size=10)

bridge = CvBridge()

r = rospy.Rate(1) # 1Hz
while not rospy.is_shutdown():
	bgr = cv2.imread(os.path.abspath('./tmpBelief.png'))
	rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
	msg = bridge.cv2_to_imgmsg(rgb, encoding="passthrough")
	pub.publish(msg)
	r.sleep()
