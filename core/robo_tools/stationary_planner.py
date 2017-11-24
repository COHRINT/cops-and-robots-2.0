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
        # Uses GoalPlanner's init function
        
	def get_goal_pose(self,pose=[0,0,0]):
		"""
                Simply returns the original position, default = [0,0,0]

                Inputs
                -------
                pose [x,y,theta] in [m,m,degrees]

		Returns
		-------
		pose  [x,y,theta] in [m,m,degrees].
		
		"""
		return pose
