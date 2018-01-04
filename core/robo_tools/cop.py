#!/usr/bin/env python

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
       self.init_belief = initial_belief
       self.init_map_bounds = map_bounds
       self.init_delta = delta
       self.init_belief.normalizeWeights()
       self.init_belief = self.init_belief.discretize2D(low=[self.init_map_bounds[0], self.init_map_bounds[1]],high=[self.init_map_bounds[2], self.init_map_bounds[3]],delta=self.init_delta)

       super(Cop, self).__init__(name, goal_planner_type)


