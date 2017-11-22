#!/usr/bin/env python

'''
Cops and Robots launchig file. Contains the update loop in the __init__ function
'''

__author__ = ["LT"]
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Ian Loefgren","Sierra Williams","Matt Aiken","Nick Sweet"]
__license__ = "GPL"
__version__ = "1.0"
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

class MainTester(object):
	"""
        Starts the CnR experiment

	Methods
	----------
	1) __init__() : launches the experiment and contains the main loop
        2) create_actors() : creates each robot as either a cop or robber
        3) update_actors() : calls the robot.update() method of each robot
        
	"""
        map_bounds = [-9.6, -3.6, 4, 3.6]
        num_robots = 2 # Maximum number of robots our experiment is designed for

        # Related to Cop's belief 
        cop_initial_belief = GM([[-6,2.5],[1,0],[-4,-2]],[[[4,0],[0,4]],[[10,0],[0,4]],[[2,0],[0,4]]],[0.5,0.5,0.5])
        delta = 0.1

	def __init__(self, config_file='config/config.yaml'):

                print("Starting Cops and Robots")
                
		rospy.init_node("Python_Node")
                
		# Create robots
		self.init_cop_robber(config_file)

                # Main Loop
                print("Entering Main Loop")
                while True:
                        self.update_cop_robber()
                        

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
		for robot, kwargs in cfg['robots'].iteritems():
			if cfg['robots'][robot]['use']:
				if cfg['robots'][robot]['type_'] == 'cop':
					self.robots[robot] = Cop(robot,
                                                                 initial_belief=self.cop_initial_belief,
                                                                 map_bounds=self.map_bounds,
                                                                 delta=self.delta,
                                                                 **kwargs) 
				elif cfg['robots'][robot]['type_'] == 'robber':
					self.robots[robot] = Robber(robot, **kwargs)
                                print("Added: " + str(robot) + " to the experiment")
                                
                print("COP AND ROBBER INITIALIZED")

	def update_cop_robber(self):
		"""
                Updates the cop and robber: goal pose and belief (if cop)
		"""
                set_trace()
		for robot_name, robot in self.robots.iteritems():
                        print("UPDATING: : " + robot_name)
                        robot.update() # calls Robot.update (in Robot.py)

if __name__ == '__main__':
	MainTester()
