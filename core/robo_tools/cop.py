#!/usr/bin/env python

#import logging
from pdb import set_trace

__author__ = ["Ian Loefgren", "Sierra Williams"]
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "2.0.0"
__maintainer__ = "Ian Loefgren"
__email__ = "ian.loefgren@colorado.edu"
__status__ = "Development"

from core.robo_tools.robot import Robot
from core.robo_tools.planner import MissionPlanner
from core.robo_tools.gaussianMixtures import GM

class Cop(Robot):
    """The Cop subclass of the generic Robot type.
    Parameters
    ----------
    name : str
        The robot's name.
    **kwargs
        Arguments passed to the ``Cop`` superclass.
    Attributes
    ----------
    mission_statuses : {'on the run'}
    """
    mission_planner_defaults = {}
    goal_planner_defaults = {'type_': 'simple',
                             'use_target_as_goal': True}
    path_planner_defaults = {'type_': 'direct'}

    def __init__(self,
                 name,
                 pose=None,
                 pose_source='python',
                 map_cfg={},
                 mission_planner_cfg={},
                 goal_planner_cfg={},
                 path_planner_cfg={},
                 initial_belief=None,
                 map_bounds=None,
                 delta=0.1,
                 **kwargs):


       print(kwargs)
       print("CREATING A COP!")

       print("Inside the robber class")
       print("Mission planner:")
       print(mission_planner_cfg)
       print("Goal planner:")
       print(goal_planner_cfg)
       print("Path planner:")
       print(path_planner_cfg)

        # Use class defaults for kwargs not included
       mp_cfg = Cop.mission_planner_defaults.copy()
       mp_cfg.update(mission_planner_cfg)
       gp_cfg = Cop.goal_planner_defaults.copy()
       gp_cfg.update(goal_planner_cfg)
       pp_cfg = Cop.path_planner_defaults.copy()
       pp_cfg.update(path_planner_cfg)

       print("mp_cfg")
       print(mp_cfg) # mp_cfg is empty
       print("gp_cfg") 
       print(gp_cfg) # type_ stationary, use_target_as_goal: True
       print("pp_cfg")
       print(pp_cfg) # type_: direct

#       set_trace()
       
       super(Cop, self).__init__(name,
                                     pose=pose,
                                     pose_source=pose_source,
                                     goal_planner_cfg=gp_cfg,
                                     path_planner_cfg=pp_cfg,
                                     map_cfg=map_cfg,
                                     create_mission_planner=False,
                                     color_str='red',
                                     **kwargs)
       self.belief = initial_belief
       self.belief.normalizeWeights()
       self.belief = self.belief.discretize2D(low=[map_bounds[0], map_bounds[1]],high=[map_bounds[2], map_bounds[3]],delta=delta)
       self.mission_planner = CopMissionPlanner(self, **mp_cfg)

class CopMissionPlanner(MissionPlanner):
    # """The Cop subclass of the generic MissionPlanner
    # """
    # mission_statuses = ['on the run', 'captured']

    def __init__(self, robot, mission_status='on the run'):

        super(CopMissionPlanner, self).__init__(robot,
                                                   mission_status=mission_status)

    def update(self):
        """Update the robot's status
        """
        pass
        # Does not make sence anymore but still needs the update
        # if self.robot.name in self.robot.found_cop.keys():
        #     self.mission_status = 'on the run'
        # if self.mission_status is 'on the run':
        #     self.stop_all_movement()
