#!/usr/bin/env python

#import logging
from pdb import set_trace

__author__ = ["LT"]
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Nick Sweet", "Nisar Ahmed", "Ian Loefgren", "Sierra Williams"]
__license__ = "GPL"
__version__ = "3.0.0"
__maintainer__ = "Luke Barbier"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"

from core.robo_tools.robot import Robot
from core.robo_tools.gaussianMixtures import GM

class Cop(Robot):
    """
    Cop Definition
    Contains
    1) self.belief, a Guassian Mixture

    Parameters
    ----------
    initial_belief : GM()
    delta : float
    map_bounds : 4 element list of the edges
         [left, bottom, right, top]
    name, goal_planner see robot.py
    """
    robot_type = 'cop'
    
    def __init__(self,
                 initial_belief, # A Gaussian
                 delta, # float
                 map_bounds, # 4 element list, corners
                 name, 
                 goal_planner_type='pomdp'):

#       set_trace()
        
       # Perform Pomdp initializations
       self.belief = initial_belief
       self.belief.normalizeWeights()
       self.belief = self.belief.discretize2D(low=[map_bounds[0], map_bounds[1]],high=[map_bounds[2], map_bounds[3]],delta=delta)

       super(Cop, self).__init__(name, goal_planner_type)

