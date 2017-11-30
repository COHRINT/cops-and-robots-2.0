#!/usr/bin/env python

'''
Cops and Robots launchig file. Contains the main update loop in the __init__ function
'''

__author__ = ["LT"]
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Ian Loefgren","Sierra Williams","Matt Aiken","Nick Sweet"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Luke Barbier"
__email__ = "luke.barbier@colorado.edu"
__status__ = "Development"


from pdb import set_trace

import sys
import os
import rospy

from core.helpers.config import load_config
from core.robo_tools.cop import Cop
from core.robo_tools.robber import Robber
from core.robo_tools.gaussianMixtures import GM
from caught.msg import Caught

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

        experiment_runspeed_hz = 1
        
        map_bounds = [-9.6, -3.6, 4, 3.6]
        max_num_robots = 2 # Maximum number of robots our experiment is designed for

        # Related to Cop's belief 
        cop_initial_belief = GM([[-6,2.5],[1,0],[-4,-2]],[[[4,0],[0,4]],[[10,0],[0,4]],[[2,0],[0,4]]],[0.5,0.5,0.5])
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
		for robot, kwargs in cfg['robots'].iteritems():

                        
			if cfg['robots'][robot]['use']:

                                # check for bad config, too many robots selected
                                num_robots += 1
                                if num_robots > self.max_num_robots:
                                        print("Bad config file, More robots selected than allowed")
                                        print("Check config/config.yaml or run gui.py and reconfigure")
                                        raise

                                # goal_planner string
                                goal_planner = cfg['robots'][robot]['goal_planner_cfg']['type_']

                                # Check cop or robber 
                                # Initialize a cop
				if cfg['robots'][robot]['type_'] == 'cop':
					self.robots[robot] = Cop(self.cop_initial_belief,
                                                                 self.delta,
                                                                 self.map_bounds,
                                                                 robot,
                                                                 goal_planner)
                                        
                                # Initialize a robber
				elif cfg['robots'][robot]['type_'] == 'robber':
					self.robots[robot] = Robber(robot, goal_planner)

                                        
                                print("Added: " + str(robot) + " to the experiment")
                                
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
                        print(msg.robber + " caught")
                        # send robots to starting positions

if __name__ == '__main__':
	MainTester()
