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
#from planner import GoalPlanner # if running as the main file (for tests)

# Whether to run through the run_list of poses or pick randomly among the pose_list
RUN_THROUGH = False


class SimpleGoalPlanner(GoalPlanner):

        pose_list = [[0, -2.25,0],[-4,-1,0],[-3,2,0],[2,-2,0],[4,-1,0],[1,2,0],[2.5,2,0],[4.5,2,0],[-3,-2,0],[0,0,0],[4.5,-2,0]]
        run_list = [[2,2,0],[-3,-2.4,0],[2,-2,0]]
        
        def __init__(self, robot_name=None, robot_pose=None):
                random.seed()

                if RUN_THROUGH :
                        self.goal_pose = self.run_list[0]
                        self.goal_num = 1
                else:
                        self.goal_pose = self.pose_list[random.randint(0, len(self.pose_list)-1)]                        
                
                super(SimpleGoalPlanner, self).__init__(robot_name, robot_pose)
        
	def get_goal_pose(self,pose=None):
		""" Find a random goal pose among the pose_list
		Returns
		-------
		array_like
			A pose as [x,y,theta] in [m,m,degrees].
		"""

                if self.reached_pose(pose, self.goal_pose):
                        if RUN_THROUGH:
                                self.goal_pose = self.run_list[self.goal_num]
                                self.goal_num = (self.goal_num + 1) % len(self.run_list)
                        else:
                                self.goal_pose = self.pose_list[random.randint(0, len(self.pose_list)-1)]
#                        self.goal_pose = self.pose_list[self.goal_num]
#                        self.goal_num += 1

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
