# Cops and Robots 2.0
## Overview
Cops and Robots 2.0 is a continuous space version of the original CnR experiment out of the University of Colorado at Boulder's Aerospace Lab, COHRINT, lead by Professor Nisar Ahmed. CnR 1.5 is functional (for issues, see **Current Issues**). The Cops and Robots experiment family (1.0, 1.5, 1.6, 2.0+) demonstate human robotic interaction through robotic probabalistic control algorithms. The basic idea is to have a "cop"/turtlebot "catch"/move close to a "robber"/turtlebot in a 2D map. A human provides observtions to the cop via a gui interface that features 3 security cameras, one cop camera, a belief map and two means of sending observations to the robot: the most valuable yes or no questions (see **Wiki/VOI**) and options for the human to push information to the robot, such as seeing the robber through a security camera. CnR 1.5 uses a 2D numpy array, a "belief" of its environment to estimate the most probable location of the robber, the MAP (maximum a posteriori). This belief becomes updated using observations given from the human through the interface and a viewcone of its own search path. This process repeats until the cop has successfully caught the robber. 
Status: **Development**
## Setup
### **For difficulties in any part see "Current Issues", first bullet point** ###
1) Install cops_and_robots/2.0 onto your local machine and catkin_make the package
2) Calibrate vicon cameras and set the origin
$ python gui.py
Select configurations
	In 1.5, the configurations were:
		In Main Tab:
			pose_source: tf
			use_ROS selected
			other configurations untouched
		In Robots Tab: (only for alternating from defaults)
			If: pris is cop, check Pris, check Roy as robber and proceed to step 3
			Else: select robots with proper planners, proper cop/robber and leave
				pose_source as tf
				Select **save** and navigate to cops-and-robots-2.0/config
				save as config.yaml
3) Click **RUN**
4) In the window that appears: type 'n' to not create a new map?
5) Enter the passwords of the robots to ssh into them
	__NOTE:__ if the ssh fails, such as the windows closing before vicon_nav is run:
		$ ssh odroid@<robot_name> and enter the password for cops and robbers
6) run $ roslaunch cops_and_robots/launch/vicon_nav.launch on each robot
7) In the terminal window prompting "When vicon_nav.launch has ....", press ENTER
8) Open a terminal window and run $ roslaunch policy_translator policy_translator.launch
9) Type '1' and hit ENTER to run the experiment

 
## For more Information
* [Observation Interface](https://github.com/COHRINT/cops-and-robots-2.0/wiki/Observation-Interface)
* [Policy Translator](https://github.com/COHRINT/cops-and-robots-2.0/wiki/Policy-Translator)
* [Value of Information](https://github.com/COHRINT/cops-and-robots-2.0/wiki/Questions-and-Value-of-Information)
	
## Current Issues

* run.sh called by gui.py sometimes fails to run a process.
	This problem has been solved by dividing the setup into multiple parts:
	Whichever part failed to open can be rerun **in a separate terminal window** while '1' has not been entered in the main terminal window
	- $ roslaunch cops-and-robots-2.0/launch/vicon_sys.launch
	- ssh into and run $ roslaunch cops_and_robots/launch/vicon_nav.launch on each robot
	- $ roslaunch policy_translator policy_translator.launch
* A caught script and topic has not been written yet
* No "robber infront of cop" observation
* cop view cone is 90 degrees offset
* The observation "robber is right of Mars Poster" is the wrong likelihood
