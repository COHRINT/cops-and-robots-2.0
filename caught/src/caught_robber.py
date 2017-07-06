from __future__ import division

"""
    Node that publishes if a cop has caught a robber
    using OpenCV's color recognition of the robber

    To add or update colors of a robber visit the
        robber_colors.yaml file

    To run:
    $ ROS_NAMESPACE=<cop name>/camera/rgb rosrun image_proc image_proc
    9
    topic_name: /<cop_name>/camera/rgb/image_color

    To view video feed from cop:
    $ rosrun image_view image_view image:=/<cop_name>/camera/rgb/image_color
"""

__author__ = "LT"
__copyright__ = "Copyright 2017, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"

import rospy
import roslib
import cv2
import numpy as np
import pymsgbox
import os
import yaml

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class Caught_Robber(object):

    def __init__(self, copList):


        self.caught_val = 12000 # Identify benchmark pixels => "a catch"

        rospy.init_node('caught_robber')

        # Identify Cops' image topics to subscribe to
        # All topics have same callback "self.caught_callback"
        for cop in copList:
            video_feed = "/" + cop + '/camera/rgb/image_color'
            rospy.Subscriber(video_feed, Image ,self.caught_callback)

        self.num_robbers = 0
        # Open color config file of robbers
        try:
            yaml_cfg_file = os.path.dirname(__file__) \
                + 'robber_colors.yaml'
            with open(yaml_cfg_file, 'r') as color_cfg:
                self.robber_info = yaml.load(color_cfg) # load color info as a dict
                for rob in self.robber_info:
                    # Add a publisher object to each robber's info
                    # Topic name "/caught_zhora" or "/caught_roy"
                    self.robber_info[rob]['pub'] = rospy.Publisher('/caught_' + rob, Bool, queue_size=10)
                    self.robber_info[rob]['caught'] = False
                    self.num_robbers += 1

        except IOError as ioerr:
            print(ioerr)

        self.bridge = CvBridge()
        self.counter = 0 # Counter for blob detection consistency
        self.caught_count = 5 # The number counter must reach for a catch (filter consistency in reading)
        print("Caught_Robber callback ready")
        rospy.spin()

    def caught_callback(self, ros_image):
        print("Entering Caught Callback")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")

            for rob in self.robber_info:
                if self.robber_info[rob]['caught'] == True: # check if the robbers already been caught
                    continue


                pub = self.robber_info[rob]['pub'] # Each robber's corresponding caught publisher

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
                print(area)

                # Check publish caught msg
                if area > self.caught_val:
                    self.counter += 1
                    if (self.counter >= self.caught_count):
                        res = pymsgbox.confirm("Did I catch " + rob + "?", title="Robber Caught?", buttons=["Yes", "No"])
                        if res == "Yes":
                            print("Caught " + rob)
                            msg = Bool()
                            msg.data = True
                            pub.publish(msg)
                            self.robber_info[rob]['caught'] = True
                            self.num_robbers -= 1
                        else:
                            print("Not Caught")
                            print("Sleeping")
                            rospy.sleep(8)
                        self.counter = 0 # restart
                else:
                    self.counter = 0

                #cv2.imshow("image", cv_image) For un affected image view

                # cv2.imshow("mask", output)
                # cv2.waitKey(10)

        except CvBridgeError as e:
            print(e)

        if self.num_robbers == 0:
            print("\n******************")
            print("All Robbers Caught!")
            print("******************\n")
            rospy.signal_shutdown("All Robbers Caught!")

if __name__ == '__main__':
    cop = ["pris"] # for multiple cops
    a = Caught_Robber(cop)
