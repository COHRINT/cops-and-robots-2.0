#!/usr/bin/env python
"""Provides pose ...

"""
__author__ = "LT" 
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Matthew Aitken", "Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "2.0.0"
__maintainer__ = "Luke Barbier"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"

import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
import tf

class Pose(object):
    """
    Contains:
    1) an update callback to the vicon topic, update_pose_callback()
    2) self.pose = [x,y,degrees] floats
    """
    precision = 2 # number of decimal places to round to
    pose = [0,0,0] # random initial pose to be overriden in the update method
    
    received = False # update the pose once before we continue
    
    def __init__(self, robot_name=None):
        
        if robot_name is None:
            print("No robot name given to the Pose Object")
            print("Check the instantiation of Pose()")
            raise
        else:
            bf_topic = "/" + robot_name.lower() + "/base_footprint"
        
        # Transform Listener in order to perform transforms
        self.listener = tf.TransformListener()

        rospy.Subscriber(bf_topic, TransformStamped, self.update_pose_callback)

        """
        Note if you just calibrated, then the robot will be on [0,0,0] and the program will hang up here.
        The solution is simple, move the robot anywhere in the continuous space except exactly [0,0,0]
        """
        print("Waiting for /" + robot_name +"/base_footprint to become available or start " + robot_name + " not on [0,0,0]")

        # self.pose == [0,0,0] for some reason the first pose returned from vicon is [0,0,0]
        # so let's update it
        while self.received is False or self.pose == [0,0,0]: # update the pose once before continuing
            pass
        print(robot_name + " initial pose: " + str(self.pose))
        
    def update_pose_callback(self, msg):
        """
        Updates the pose using the transform topic from vicon
        """
        self.received = True

        # Receive the msg
        x = msg.transform.translation.x
        y = msg.transform.translation.y

        xr = msg.transform.rotation.x
        yr = msg.transform.rotation.y
        zr = msg.transform.rotation.z
        wr = msg.transform.rotation.w
        quat = [xr, yr, zr, wr]
        (_, _, theta) = tf.transformations.euler_from_quaternion(quat)

        # Round the positions
        x = round(x, self.precision)
        y = round(y, self.precision)
        t = round(theta, self.precision)
        self.pose = [x, y, t]

if __name__ == '__main__':
    pose = Pose('zhora')
    while True:
        pass
    
