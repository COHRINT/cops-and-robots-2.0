import numpy as np 
# from gaussianMixtures import gm

# beliefs = np.load("/home/ian/catkin_ws/src/cops-and-robots-2.0/policy_translator/srcthebeliefsfile_151968402.npy")
obs = np.load("/home/ian/catkin_ws/src/cops-and-robots-2.0/policy_translator/srctheobsfile_151968458.npy")
prisLoc = np.load("/home/ian/catkin_ws/src/cops-and-robots-2.0/Pris_goal_planner_type_15196846.npy")
zhoraLoc = np.load("/home/ian/catkin_ws/src/cops-and-robots-2.0/Zhora_goal_planner_type_15196844.npy")

# print(beliefs)
print(obs)
print(prisLoc)
print(zhoraLoc)