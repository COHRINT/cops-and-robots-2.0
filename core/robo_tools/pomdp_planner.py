#!/usr/bin/env python

""" The 'pomdp' goal planner subclass of GoalPlanner
"""

__author__ = ["Ian Loefgren", "Sierra Williams"]
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "2.0.2" # Edit 1/25/18 LT [removed print statement]
#Edit 11/21/17 LT
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

from pdb import set_trace

import random
import logging
import math
from shapely.geometry import Point, LineString
import rospy
import sys
import numpy as np

from core.robo_tools.planner import GoalPlanner
from core.robo_tools.belief_handling import dehydrate_msg, rehydrate_msg, discrete_rehydrate, discrete_dehydrate

from policy_translator.srv import *
from policy_translator.msg import *

class PomdpGoalPlanner(GoalPlanner):

	def __init__(self, belief, bounds, delta, robot_name=None, robot_pose=None):

                self.belief = belief
                self.delta = delta
                
		self.shapes = [int((bounds[2]-bounds[0])/self.delta),int((bounds[3]-bounds[1])/self.delta)]

		super(PomdpGoalPlanner, self).__init__(robot_name, robot_pose)
#                set_trace()

	def get_goal_pose(self,pose=None):
		"""Find goal pose from POMDP policy translator server

		Parameters
		----------
		Returns
		--------
		goal_pose [array]
			Goal pose in the form [x,y,theta] as [m,m,degrees]
		"""
		msg = PolicyTranslatorRequest()
		msg.name = self.robot_name
           	if self.belief is not None:
			(msg.weights,msg.means,msg.variances) = dehydrate_msg(self.belief)
		else:
			msg.weights = []
			msg.means = []
			msg.variances = []
		res = None

                print("Waiting for the POMDP Policy Service")
		rospy.wait_for_service('translator')
		try:
			pt = rospy.ServiceProxy('translator',policy_translator_service)
			res = pt(msg)
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e

                if not hasattr(res, 'response'):
                        print("***ERROR***, policy translator server did not return an ideal response, check the server node for an error")
                
		self.belief = rehydrate_msg(res.response.weights_updated, res.response.means_updated, res.response.variances_updated)

		goal_pose = list(res.response.goal_pose)

#		print("NEW GOAL POSE: {}".format(goal_pose))
                
                # Round the given goal pose
                for i in range(len(goal_pose)):
                        goal_pose[i] = round(goal_pose[i], 2)
#                print("Rounded goal pose: {}".format(goal_pose))
                
		return goal_pose
