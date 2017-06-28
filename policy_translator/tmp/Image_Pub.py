#!/usr/bin/env python

'''
-Publishes an empty message for the sole purpose of indicating to the observation_interface (interface_elements.py)
to update the belief map (tmpBelief.png)
-AN EMPTY IMAGE IS PUBLISHED
 	tmpBelief.png IS ACTUALLY READ IN interface_elements.py-MapDisplay-ros_update()
'''

__author__ = ["Luke Barbier", "Ian Loefgren"]
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Luke Barbier", "Ian Loefgren"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Luke Barbier"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"

import rospy
from sensor_msgs.msg import Image
import os

import cv2
from cv_bridge import CvBridge, CvBridgeError

rospy.init_node('interface_map')
pub = rospy.Publisher('/interface_map', Image, queue_size=10)

#bridge = CvBridge()

r = rospy.Rate(1) # 1Hz
while not rospy.is_shutdown():
	#bgr = cv2.imread(os.path.abspath('./tmpBelief.png'))
	# rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
	# msg = bridge.cv2_to_imgmsg(rgb, encoding="passthrough")
	msg = Image() # pass an empty msg
	pub.publish(msg)
	r.sleep()
