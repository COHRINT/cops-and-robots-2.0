#!/usr/bin/env python

""" Subclass of Goal Planner
Returns one of a list of positions, once a robot has reached the previous goal
"""

__author__ = ["LT"]
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Nick Sweet", "Nisar Ahmed", "Ian Loefgren", "Sierra Williams"]
__license__ = "GPL"
__version__ = "3.0.0"
__maintainer__ = "Luke Barbier"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"

from pdb import set_trace

import random
from core.robo_tools.planner import GoalPlanner
#from planner import GoalPlanner # if main file

class SimpleGoalPlanner(GoalPlanner):

        pose_list = [[-1.25, 2.5,0], [1,2,0], [0, -2.25,0], [3, -1.0, 0], [-8.5,-1,0], [-6,2,0]]
        
        def __init__(self, robot_name=None, robot_pose=None):
                random.seed()
#                self.goal_pose = [-5,0,0]
                self.goal_pose = self.pose_list[random.randint(0, len(self.pose_list)-1)]
                super(SimpleGoalPlanner, self).__init__(robot_name, robot_pose)
        
	def get_goal_pose(self,pose=None):
		""" Find a random goal pose among the pose_list
		Returns
		-------
		array_like
			A pose as [x,y,theta] in [m,m,degrees].
		"""
#                Check conditions for new goal_pose
                # if pose == self.goal_pose:
                #         self.goal_pose = self.pose_list[random.randint(0, len(self.pose_list)-1)]

                if self.reached_pose(pose, self.goal_pose):
                        self.goal_pose = self.pose_list[random.randint(0, len(self.pose_list)-1)]

                return self.goal_pose                

if __name__ == "__main__":
        pose = [1,1,1]
        s = SimpleGoalPlanner("zhora", pose)
        goal_pose = s.get_goal_pose(pose)
        for i in range(0,10): # Test while the robot has not reached its goal pose
                print(s.get_goal_pose(pose))
        pose = goal_pose
        for i in range(0,10): # Check that the pose has changed
                print(s.get_goal_pose(pose))
