#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

"""
    *** COMPUTER VERSION ***

    -To run, $ roslaunch caught caught.launch
    See Wiki for additional running instructions

    -Node that publishes if a cop has caught a robber
    using OpenCV's color recognition of the robber

    -To add or update colors of a robber visit the
        robber_colors.yaml file
"""

__author__ = "LT"
__copyright__ = "Copyright 2017, Cohrint"
__license__ = "GPL"
__version__ = "1.4.0"
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Stable"

import rospy
import rospkg
import roslib
import cv2
import numpy as np
import numpy.polynomial.polynomial as poly
import os
import yaml

from sensor_msgs.msg import Image
from caught.msg import Caught
from cv_bridge import CvBridge, CvBridgeError

# calculated from find_cam_calib.py
# coefs = [ 33518.96832373, -48636.45623814,  18884.04641427] # for 0 - 1.5m
coefs = [ 34706.81758805, -58782.89189293,  27970.47650456] # for 0 - 1.0m
WAIT_TIME = 50
DEFAULT_CATCH_DIST = 0.5

class Caught_Robber(object):


    def __init__(self):

        rospy.init_node('caught_robber')

        # Identify benchmark pixels => "a catch"
        dist = rospy.get_param('~catch_dist', DEFAULT_CATCH_DIST)
        rospy.loginfo("Desired catch distance: " + str(dist))
        self.caught_val = self.calibrate_caught_distance(dist)

        # Identify Cops' image topics to subscribe to
        # All topics have same callback "self.caught_callback"
        copList = rospy.get_param('~cops', ['pris'])
        rospy.logdebug("CopList: " + str(copList))
        for cop in copList:
            video_feed = "/" + cop + '/camera/rgb/image_color'
            rospy.Subscriber(video_feed, Image ,self.caught_callback)
            rospy.loginfo("Caught subscribed to: "+ video_feed)

        self.num_robbers = 0
        self.pub = rospy.Publisher('/caught' , Caught, queue_size=10)
        rospy.Subscriber('/caught_confirm', Caught, self.jail_robber)
        # Open color config file of robbers
        try:
            yaml_cfg_file = os.path.dirname(__file__) \
                + '/robber_colors.yaml'
            with open(yaml_cfg_file, 'r') as color_cfg:
                self.robber_info = yaml.load(color_cfg) # load color info as a dict
                for rob in self.robber_info:
                    self.robber_info[rob]['caught'] = False
                    self.num_robbers += 1

        except IOError as ioerr:
            rospy.logerr("Error in reading robber_colors.yaml")
            print(ioerr)

        self.bridge = CvBridge()
        self.publishing = False
        self.wait_time = 0

        self.counter = 0 # Counter for blob detection consistency
        self.caught_count = 1 # The number counter must reach for a catch (filter consistency in reading)
        rospy.loginfo("Caught Node ready")
        self.show_mask = rospy.get_param('~show_mask', 'True')
        rospy.spin()



    def calibrate_caught_distance(self, dist):
    """ Returns the pixel area for a catch given a distance """

        if (dist > 1.0):
            rospy.logerr("Invalid distance, Please enter a value" +
                " below 1.0")
            return 0
        if (dist < .15):
            dist = .15
        return poly.polyval(dist, coefs)

    def caught_callback(self, ros_image):
    """  Analyzes cop video feed for robber color values specificed
    in the yaml file. Color masks image then compares largest contour
    against the caught_val to verify a catch """

        rospy.logdebug("Entering Caught Callback: ")
        if self.publishing == True:
            rospy.logdebug("Already Published")
            return
        elif self.wait_time > 0:
            rospy.logdebug("Wait: " + str(self.wait_time))
            self.wait_time -= 1
            if self.wait_time == 0:
                rospy.loginfo("Ready to Catch again!")
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")

            for rob in self.robber_info:
                if self.robber_info[rob]['caught'] == True: # check if the robbers already been caught
                    continue

                r_min = self.robber_info[rob]['r']['MIN']
                g_min = self.robber_info[rob]['g']['MIN']
                b_min = self.robber_info[rob]['b']['MIN']

                r_max = self.robber_info[rob]['r']['MAX']
                g_max = self.robber_info[rob]['g']['MAX']
                b_max = self.robber_info[rob]['b']['MAX']

                # Make lower and upper lists in BGR format
                lower = [b_min, g_min, r_min] # BGR
                upper = [b_max, g_max, r_max]

                # Convert to cv2's required numpy arrays
                lower_np = np.array(lower, dtype= "uint8")
                upper_np = np.array(upper, dtype= "uint8")

                # Analyze and manipulate the image
                mask = cv2.inRange(cv_image, lower_np, upper_np) # Color mask image according to min and max
                image, cont, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                output = cv2.bitwise_and(cv_image, cv_image, mask = mask)

                # Find the largest contour
                c = max(cont, key=cv2.contourArea)
                area = cv2.contourArea(c)
                rospy.logdebug("Max Contour: " + str(area))

                # Check publish caught msg
                if area > self.caught_val:
                    self.counter += 1
                    if (self.counter >= self.caught_count):
                        self.publishing = True
                        msg = Caught()
                        msg.robber = rob
                        msg.confirm = True
                        self.pub.publish(msg)
                        self.counter = 0 # restart
                        rospy.loginfo("Caught Node publishing catch of: " + rob.capitalize())
                else:
                    self.counter = 0

                if self.show_mask:
                    # cv2.imshow("image", cv_image) #For un affected image view
                    cv2.imshow("mask", output)
                    cv2.waitKey(5)

        except CvBridgeError as e:
            print(e)



    def jail_robber(self, msg):
        if msg.confirm == True:
            self.robber_info[msg.robber]['caught'] = True
            self.num_robbers -= 1
            rospy.loginfo(msg.robber.capitalize() + " jailed!")
        else:
            rospy.loginfo("It wasn't %s...", msg.robber.capitalize())
            self.wait_time = WAIT_TIME
            rospy.loginfo("Beginning wait of " + str(self.wait_time) + " cycles")
	    self.publishing = False
        if self.num_robbers == 0:
            rospy.loginfo("***All Robbers Caught!***")
            rospy.signal_shutdown("All Robbers Caught!")

if __name__ == '__main__':
    a = Caught_Robber()
