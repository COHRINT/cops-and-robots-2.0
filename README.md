# Cops and Robots 2.0
## Overview
Cops and Robots 2.0 is **an improved** revival of the original Cops and Robots experiment out of the University of Colorado at Boulder's Aerospace Lab, COHRINT, lead by Professor Nisar Ahmed. CnR 2.0 is functional (for issues, see **Current Issues**). The Cops and Robots experiment family (1.0, 1.5, 1.6, 2.0+) demonstrate human robotic interaction through robotic probabilistic control algorithms. The basic idea is to have a "cop"/turtlebot "catch"/move close to a "robber"/turtlebot in a 2D map. A human provides observations to the cop via a GUI interface that features 3 security cameras, one cop camera, a belief map and two means of sending observations to the robot: the most valuable yes or no questions (see **Wiki/VOI**) and options for the human to push information to the robot, such as seeing the robber through a security camera. CnR 2.0 uses a 2D numpy array, a "belief" of its environment to estimate the most probable location of the robber, the MAP (maximum a posteriori). This belief becomes updated using observations given from the human through the interface and a view-cone of its own search path. This process repeats until the cop has successfully caught the robber.

The current version under development is CnR 2.1, please visit that branch for an updated status on its development.

## Setup
### **For difficulties in any part see "Current Issues", first bullet point** ###
1) Install cops-and-robots-2.0 onto your local machine and catkin_make the package
 - In catkin_ws/src
   - $ git clone https://github.com/COHRINT/cops-and-robots-2.0.git
   - $ bash cops-and-robots-2.0/install.sh
 - In catkin_ws/
   - $ catkin_make
2) Calibrate vicon cameras and set the origin
3) Edit config/config.yaml for necessary robot and map parameters (see file for further instructions)
4) source aliases (must be done for every terminal, unless added to .bashrc)
 - $ source aliases.txt
 - OR if you want to add aliases to .bashrc
   - $ cat aliases.txt >> ~/.bashrc
5) Start vicon system
 - $ vsys
6) ssh into robots
 - Deckard: $ deck
 - Roy: $ roy
 - Pris: $ pris
 - Zhora: $ zhora
7) Configure robots:
 - On Cop: $ cop:=robber_name
   - For example, if pris is the robber: $ cop:=pris
 - On Robber: $ rob
8) Run experiment
 - Start Policy Translator: $ pol
 - Bring up interface: $ obs"first_letter_of_cop"
   - For example, if deckard is the cop: $ obsd
 - Begin experiment: $ python main.py
9) In the terminal window prompting "When vicon_nav.launch has ....", press ENTER
10) Type '1' and hit ENTER to run the experiment


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
* No "robber infront of cop" observation
