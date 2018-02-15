#!/usr/bin/env python
"""
Generic Robot Class
1) __init__() instantiates the robot
2) update() : updates the robot's pose and its goal pose

"""
from pdb import set_trace

__author__ = "LT"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "3.0.0"
__maintainer__ = "Luke Barbier"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"

import rospy
from std_msgs.msg import Bool
from core.robo_tools.pose import Pose
from core.robo_tools.planner import GoalPlanner


class Robot(object):
    """Class definition for the generic robot object.

    Parameters
    ----------
    name : str
        The robot's name.
    goal_planner_type : str
        Which goal planner object to instantiate as self.goal_planner

    Contains
    --------
    1) __init__()
    2) update()
    """

    def __init__(self, name, goal_planner_type='stationary'):
        
        # Stop attribute
#        rospy.Subscriber('/'+name.lower()+'/stop', Bool, self.stop_callback)
        
        # Object attributes
        self.name = name.lower()
        self.Pose = Pose(self.name)

        # Select and instantiate the goal planner
        if goal_planner_type == 'stationary':
            from stationary_planner import StationaryGoalPlanner
            self.goal_planner = StationaryGoalPlanner(self.name, self.Pose.pose)

        elif goal_planner_type == 'simple':
            from simple_planner import SimpleGoalPlanner
            self.goal_planner = SimpleGoalPlanner(self.name, self.Pose.pose)
            
        elif goal_planner_type == 'pomdp':
            from pomdp_planner import PomdpGoalPlanner
            self.goal_planner = PomdpGoalPlanner(self.init_belief, self.init_map_bounds, self.init_delta, self.name, self.Pose.pose)
            # These variables are now in the goal planner, let's not store the references in two places for debugging purposes
            del self.init_belief 
            del self.init_map_bounds
            del self.init_delta
            
	elif goal_planner_type == 'rob_int':
            from robber_evasion_planner import robberEvasionGoalPlanner
            self.goal_planner = robberEvasionGoalPlanner()

        elif goal_planner_type == 'audio': # Jeremy's Audio Planner. Intergration with this goal planner class has not been set up yet
            from audio_planner import AudioGoalPlanner
            self.goal_planner = AudioGoalPlanner()
            
# Other planners not used. No integration thus far done
        elif goal_planner_type == 'trajectory': 
            from trajectory_planner import TrajectoryGoalPlanner
            self.goal_planner = TrajectoryGoalPlanner()

        elif goal_planner_type == 'particle':
            from particle_planner import ParticleGoalPlanner
            self.goal_planner = ParticleGoalPlanner()

        elif goal_planner_type == 'MAP':
            from probability_planner import PorbabilityGoalPlanner
            self.goal_planner = ProbabilityGoalPlanner()
            
        else:
            print("No goal planner selected. Check instantation of robot.py")
            raise

    def update(self): 
        """
        1 thing to update: goal pose

        Note: self.Pose.pose is being updated in the background by a callback to /robot_name/base_footprint
        """
        
        print(self.name + " pose: " + str(self.Pose.pose))
        
        # Update the robot's goal pose
        self.goal_planner.update(self.Pose.pose) # generally goes to planner.py update()

