#!/usr/bin/env python

'''
Cops and Robots launchig file. Contains the main update loop in the __init__ function
'''

__author__ = ["LT"]
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Ian Loefgren","Sierra Williams","Matt Aiken","Nick Sweet"]
__license__ = "GPL"
__version__ = "2.2" # for CnR 2.0
__maintainer__ = "Luke Barbier"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"


from pdb import set_trace

import sys
import os
import rospy
import yaml

from core.helpers.config import load_config
from core.robo_tools.cop import Cop
from core.robo_tools.robber import Robber
from core.robo_tools.gaussianMixtures import GM, Gaussian
from caught.msg import Caught
from std_msgs.msg import Bool

class MainTester(object):
	"""
        Starts the CnR experiment

	Methods
	----------
	1) __init__() : launches the experiment and contains the main loop
        2) init_cop_robber() : creates each robot as either a cop or robber
        3) update_cop_robber() : calls the robot.update() method of each robot
        4) end_experiment() : callback to the /caught_confirm topic and influences the self.running_experiment variable
        
	"""
        running_experiment = True

        experiment_runspeed_hz = 4
        
        map_bounds = [-5, -2.5, 5, 2.5]
        max_num_robots = 2 # Maximum number of robots our experiment is designed for

        # Related to Cop's belief 
        # cop_initial_belief = GM() # cop x, cop y, rob x, rob y, then follow the rooms
        # cop_initial_belief.addNewG([0,0,-2,2],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # kitchen
        # cop_initial_belief.addNewG([0,0,-5,0],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # hallway
        # cop_initial_belief.addNewG([0,0,0,-2.5],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # library
        # cop_initial_belief.addNewG([0,0,2,2.5],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # billiards room
        # cop_initial_belief.addNewG([0,0,-5,-2],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # study 
        # cop_initial_belief.addNewG([0,0,-8,-2],[[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]],1) # dining room 
        # cop_initial_belief.normalizeWeights()
        
        delta = 0.1

	def __init__(self, config_file='config/config.yaml'):

                print("Starting Cops and Robots")
                
		rospy.init_node("Python_Node")
                rospy.Subscriber('/caught_confirm', Caught, self.end_experiment)

                # caught_confirm topic
                
		# Create robots
		self.init_cop_robber(config_file)

                # Main Loop
                print("Entering Main Loop")
                r = rospy.Rate(self.experiment_runspeed_hz) # 1 Hz
                while self.running_experiment is True and not rospy.is_shutdown():
                        self.update_cop_robber()
                        r.sleep()
                for robot in self.robots:
                        self.robots[robot].goal_planner.return_position()
                        rospy.sleep(1)
                print("Experiment Finished")


	def init_cop_robber(self, config_file=None):
                """
                Initialize the cop and robber using the config file
                """
                if config_file != None: 
                        cfg = load_config(config_file) #load the config file as a dictionary
                else:   
                        print("No Config File. Restart and pass the config file.")
                        raise

                
		self.robots = {} # robot dictionary
                num_robots  = 0
                try:
                        for robot, kwargs in cfg['robots'].iteritems():
                                if cfg['cop_rob'][robot] != 'no':

                                        # check for bad config, too many robots selected
                                        num_robots += 1
                                        if num_robots > self.max_num_robots:
                                                print("Bad config file, More robots selected than allowed")
                                                print("Check config/config.yaml or run gui.py and reconfigure")
                                                raise

                                        # goal_planner string
                                        goal_planner = cfg['robots'][robot]['goal_planner']
                                        
                                        # Check cop or robber 
                                        # Initialize a cop
				        if cfg['cop_rob'][robot] == 'cop':
                                                with open('models/'+cfg['map']+'.yaml', 'r') as stream:
                                                        map_cfg = yaml.load(stream)
                                                cop_initial_belief = GM()
                                                for room in map_cfg['info']['rooms']:
                                                        max_x = map_cfg['info']['rooms'][room]['max_x']
                                                        max_y = map_cfg['info']['rooms'][room]['max_y']
                                                        min_x = map_cfg['info']['rooms'][room]['min_x']
                                                        min_y = map_cfg['info']['rooms'][room]['min_y']
                                                        cent_x = (max_x + min_x) / 2
                                                        cent_y = (max_y + min_y) / 2
                                                        cop_initial_belief.addG(Gaussian([0,0,cent_x,cent_y],[[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0],[0,0,0,0.5]],1))
                                                        
                                                cop_initial_belief.normalizeWeights()
                                                
					        self.robots[robot] = Cop(cop_initial_belief,
                                                                         self.delta,
                                                                         self.map_bounds,
                                                                         robot,
                                                                         goal_planner)
                                                
                                                # Initialize a robber
				        elif cfg['cop_rob'][robot] == 'rob':
					        self.robots[robot] = Robber(robot, goal_planner)
                                                
                                                
                                                print("Added: " + str(robot) + " to the experiment")
                except TypeError as ex:
                        print("***ERROR***, in config/config.yaml, add singe quotes (') around 'cop', 'rob' and 'no' ")
                        raise
                except Exception as ex:
                        template = "***ERROR*** An exception of type {0} occurred. Arguments:\n{1!r}"
                        message = template.format(type(ex).__name__, ex.args)
                        print message
                        raise
		                
                print("COP AND ROBBER INITIALIZED")

	def update_cop_robber(self):
		"""
                Updates the cop and robber: goal pose and belief (if cop)
		"""
#                set_trace()
		for robot_name, robot in self.robots.iteritems():
#                        print("UPDATING: : " + robot_name)
                        robot.update() # calls Robot.update (in Robot.py)
                        
        def end_experiment(self, msg):
                if msg.confirm is True:
                        self.running_experiment = False
                        print("*****"+ msg.robber.upper() + " CAUGHT*****")
                        print("  ENDING EXPERIMENT")
                        self.running_experiment = False

if __name__ == '__main__':
        MainTester()
