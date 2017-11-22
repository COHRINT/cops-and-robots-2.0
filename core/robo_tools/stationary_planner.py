#!/usr/bin/env python

""" Type of goal planner that keeps a robot in the same position, stationary
--- subclass of "GoalPlanner"
"""

__author__ = "LT"
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = "Nisar Ahmed"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "LT"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Stable"

from core.robo_tools.planner import GoalPlanner

class StationaryGoalPlanner(GoalPlanner):

	def __init__(self, robot, type_='stationary', view_distance=0.3, use_target_as_goal=True,goal_pose_topic=None,**kwargs):

		super(StationaryGoalPlanner, self).__init__(robot=robot, type_=type_, view_distance=view_distance, use_target_as_goal=use_target_as_goal, goal_pose_topic=goal_pose_topic)

	def find_goal_pose(self,positions=None):
		"""
                Simply returns the original position

                Inputs
                -------
                pose [x,y,theta] in [m,m,degrees]

		Returns
		-------
		array_like
			A pose as [x,y,theta] in [m,m,degrees].
		"""
                
		return positions

	def update(self,pose=None):
                """
                Parameters
                ---------
                positions [x,y, degrees] floats
                """

		super(StationaryGoalPlanner, self).update(pose)
