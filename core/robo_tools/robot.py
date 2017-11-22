#!/usr/bin/env python
"""
Generic Robot Class
1) __init__()
2) 

A robot has a planner that allows it to select goals and a map to
keep track of other robots, feasible regions to which it can move,
an occupancy grid representation of the world, and role-specific
information (such as a probability layer for the rop robot to keep
track of where robber robots may be).

"""
from pdb import set_trace

__author__ = "LT"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Luke Barbier"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"

import logging
import math
import random
import numpy as np

from shapely.geometry import Point, Polygon

from core.robo_tools.pose import Pose
from core.robo_tools.planner import (MissionPlanner,
                                             GoalPlanner,
                                             PathPlanner,
                                             Controller)
from core.map_tools.map import Map
from core.map_tools.map_elements import MapObject


class Robot(object):
    """Class definition for the generic robot object.

    .. image:: img/classes_Robot.png

    Parameters
    ----------
    name : str
        The robot's name.
    pose : array_like, optional
        The robot's initial [x, y, theta] in [m,m,degrees] (defaults to
        [0, 0.5, 0]).
    pose_source : str
        The robots pose source. Either  a rostopic name, like 'odom' or
        'tf', or 'python'
    color_str : str
        The color of the robots map object
    **kwargs
        Arguments passed to the ``MapObject`` attribute.

    """

    def __init__(self,
                 name,
                 pose=None,
                 pose_source='python',
                 color_str='darkorange',
                 map_cfg={},
                 create_mission_planner=True,
                 goal_planner_cfg={},
                 path_planner_cfg={},
                 **kwargs):

        print("ENTERING ROBOT.PY")
        print("Pose: "+ str(pose))
        print("pose_source: " + str(pose_source))
        print("map_cfg: " +str(map_cfg))
        print("create_mission_planner: "+ str(create_mission_planner))
        print("goal_planner_cfg: "+ str(goal_planner_cfg))
        print("path_planner_cfg: "+str(path_planner_cfg))
        print("kwargs: "+ str(kwargs))
        
#        set_trace()
        
        # Object attributes
        self.name = name
        self.pose_source = pose_source

        # Robot belief
        # self.belief = None

        # Setup map     COME BACK IF WE ACTUALLY USE THIS
        self.map = Map(**map_cfg)
        print(self.map)
#        set_trace()

        # We probably don't need this feasible layer stuff
        # all it influences is self.map ...
        
        # If pose is not given, randomly place in feasible layer.
        feasible_robot_generated = False
        if pose is None:
            while not feasible_robot_generated:
                x = random.uniform(self.map.bounds[0], self.map.bounds[2])
                y = random.uniform(self.map.bounds[1], self.map.bounds[3])
                if self.map.feasible_layer.pose_region.contains(Point([x, y])):
                    feasible_robot_generated = True
            theta = random.uniform(0, 359)
            pose = [x, y, theta]

        self.pose2D = Pose(self, pose, pose_source)
        self.pose_history = np.array(([0, 0, 0], self.pose2D.pose))
        if pose_source == 'python':
            self.publish_to_ROS = False
        else:
            self.publish_to_ROS = True

#        set_trace()
        # if publishing to ROS, create client for occupancy grid service
        if self.publish_to_ROS:
            import rospy
            import nav_msgs.msg
            from nav_msgs.msg import OccupancyGrid, MapMetaData
            from nav_msgs.srv import GetMap
            print("Waiting for /" + self.name.lower() + "/static_map rosservice to become available")
            rospy.wait_for_service('/' + self.name.lower() + '/static_map')
            try:
                get_map = rospy.ServiceProxy('/' + self.name.lower() + '/static_map',GetMap)
                map_msg = get_map()
                logging.info("Received new map")
                
#                set_trace()
            except rospy.ServiceException, e:
                print "Service call for map failed: %s"%e

            self.map_server_info_update(map_msg)
#        set_trace()
        # Setup planners
        if create_mission_planner: # THIS CODE DOES NOT RUN
            print("CREATING_MSSION_PLANNER")
            self.mission_planner = MissionPlanner(self)

        goal_planner_type = goal_planner_cfg['type_'] # stationary/pomdp
        print("goal_planner_type: "+str(goal_planner_type))

        if goal_planner_type == 'stationary':
            from stationary_planner import StationaryGoalPlanner
            self.goal_planner = StationaryGoalPlanner(self, **goal_planner_cfg)

        elif goal_planner_type == 'simple':
            from simple_planner import SimpleGoalPlanner
            self.goal_planner = SimpleGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'trajectory':
            from trajectory_planner import TrajectoryGoalPlanner
            self.goal_planner = TrajectoryGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'particle':
            from particle_planner import ParticleGoalPlanner
            self.goal_planner = ParticleGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'MAP':
            from probability_planner import PorbabilityGoalPlanner
            self.goal_planner = ProbabilityGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'pomdp':
            from pomdp_planner import PomdpGoalPlanner
            self.goal_planner = PomdpGoalPlanner(self,**goal_planner_cfg)

        elif goal_planner_type == 'audio':
            from audio_planner import AudioGoalPlanner
            self.goal_planner = AudioGoalPlanner(self,**goal_planner_cfg)

        # elif self.goal_planner_type == 'trajectory':
        #     self.goal_planner
        # elif self.type == 'particle':
        #     target_pose = self.find_goal_from_particles()
        # elif self.type == 'MAP':
        #     target_pose = self.find_goal_from_probability()
        #
        # self.goal_planner = GoalPlanner(self,
        #                                 **goal_planner_cfg)
        # If pose_source is python, this robot is just in simulation
        if not self.publish_to_ROS:
            print("NOT PUBLISH TO ROS?")
            self.path_planner = PathPlanner(self, **path_planner_cfg)
            self.controller = Controller(self)

        # Define MapObject
        # <>TODO: fix this horrible hack
        create_diameter = 0.34
        self.diameter = create_diameter
        if self.name == 'Deckard':
            pose = [0, 0, -np.pi / 4]
            r = create_diameter / 2
            n_sides = 4
            pose = [0, 0, -np.pi / 4]
            x = [r * np.cos(2 * np.pi * n / n_sides + pose[2]) + pose[0]
                 for n in range(n_sides)]
            y = [r * np.sin(2 * np.pi * n / n_sides + pose[2]) + pose[1]
                 for n in range(n_sides)]
            shape_pts = Polygon(zip(x, y)).exterior.coords
        else:
            shape_pts = Point([0, 0]).buffer(create_diameter / 2)\
                .exterior.coords
        self.map_obj = MapObject(self.name, shape_pts[:], has_relations=False,
                                 blocks_camera=False, color_str=color_str)

        # Move_absolute => Moving the robot to a new position
        self.update_shape()

    def map_server_info_update(self,occupancy_grid_msg):
        """Update stored info about occupancy_grid
        """
        self.occupancy_grid = occupancy_grid_msg.map.data
        self.map_resolution = occupancy_grid_msg.map.info.resolution
        self.map_height = occupancy_grid_msg.map.info.height
        self.map_width = occupancy_grid_msg.map.info.width
#        set_trace()
        logging.info("Map metadata updated")

    def update_shape(self):
        """Update the robot's map_obj.
        """
        self.map_obj.move_absolute(self.pose2D.pose)

        
    def update(self): 
        """Update the poses and goal_pose

        This includes planning and movement for both cops and robbers,
        as well as sensing and map animations for cops.

        """
        # Update the robot's pose using a tf transform
        self.pose2D.tf_update() # pose is stored in self.pose2D._pose
        
        # Update the robots goal pose
        self.goal_planner.update(self.pose2D._pose)

        print("POSE Object's Pose: " + str(self.pose2D._pose))

        # Do we really need to update the shape?
        self.update_shape()
        
        
#        set_trace()
#        if self.mission_planner.mission_status is not 'stopped':
            # Update statuses and planners
#            self.mission_planner.update() # No need for this
#            set_trace()
        

            # why do we have a pose_history?
            # Add to the pose history, update the map
            # self.pose_history = np.vstack((self.pose_history,
            #                                self.pose2D.pose[:]))
            
