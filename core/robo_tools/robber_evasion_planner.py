#!/usr/bin/env python

'''
Robber Navigation
Escapes from cops while stealing goods
'''

__author__ = ["Sousheel Vunnam"]
__copyright__ = "Copyright 2018, COHRINT"
__credits__ = ["Nisar Ahmed", "Luke Barbier"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Sousheel"
__email__ = "sousheel.vunnam@colorado.edu"
__status__ = "Development"

from planner import GoalPlanner
import tf
import geometry_msgs.msg as geo_msgs
import math
from robber_intelligence.srv import robberEvasionGoal
import rospy

class robberEvasionGoalPlanner(GoalPlanner):

	def __init__(self, robotName=None, robot_pose=None):
		self.goal_pose = robot_pose

		super(robberEvasionGoalPlanner, self).__init__(robotName, robot_pose)

	def get_goal_pose(self,pose=None):
		"""
		Find goal pose from robber evasion server
		Parameters
		----------
		Returns
		--------
		goal_pose [array]
			Goal pose in the form [x,y,theta] as [m,m,degrees]
		"""
		rospy.wait_for_service('robberEvasionGoal')

		print('Now getting Robber evasion goal'); 

		isAtGoal = False
		if self.reached_pose(pose, self.goal_pose):
			isAtGoal = True

		try:
			getRobberGoal = rospy.ServiceProxy('robberEvasionGoal', robberEvasionGoal)
			print("Waiting for robber evasion server...")
			goalResponse = getRobberGoal(isAtGoal)
			(roll,pitch,yaw) = tf.transformations.euler_from_quaternion([goalResponse.robberGoalResponse.pose.orientation.x,
				goalResponse.robberGoalResponse.pose.orientation.y,
				goalResponse.robberGoalResponse.pose.orientation.z,
				goalResponse.robberGoalResponse.pose.orientation.w]
			)
			theta = math.degrees(yaw)
			goal_pose = [goalResponse.robberGoalResponse.pose.position.x, goalResponse.robberGoalResponse.pose.position.y, theta]
			self.goal_pose = goal_pose
			return goal_pose
		except rospy.ServiceException, e:
			print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
			print("Robber intelligence planner exception!!!!!!!!!")
			print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
			
			return pose

