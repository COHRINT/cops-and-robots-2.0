from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal as mvn
import random
import copy
import cProfile
import re
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import os; 
from math import sqrt
import signal
import sys
import cProfile
sys.path.append('../src/'); 
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
import matplotlib.animation as animation
from numpy import arange
import time
import matplotlib.image as mgimg
from softmaxModels import Softmax

'''
****************************************************
File: D2QuestSoftmaxModel.py
Written By: Luke Burks
Febuary 2017

Container Class for problem specific models
Model: Question asking POMDP
A single room, with a table off to the side
Softmax Observations are question dependent


Bounds from [0,10] by [0,5] on both dimensions 

Rewards are dispensed for being in the correct

****************************************************


'''

__author__ = "Luke Burks"
__copyright__ = "Copyright 2017, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"



class ModelSpec:

	def __init__(self):
		self.fileNamePrefix = 'D2QuestsSoftmax'; 
		self.walls = []; 


	#Problem specific
	def buildTransition(self):
		self.bounds = [[0,10],[0,5]]; 
		self.delAVar = (np.identity(2)*0.25).tolist(); 
		#self.delA = [[-0.5,0],[0.5,0],[0,-0.5],[0,0.5],[0,0],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5],[0.5,0.5]]; 
		delta = 0.5; 
		self.delA = [[-delta,0],[delta,0],[0,delta],[0,-delta],[0,0]]; 
		self.discount = 0.95; 

		#set wall line segments
		#self.walls = [[[2.5,-1],[2.5,2]],[[0,0],[0,5]],[[0,5],[5,5]],[[5,0],[5,5]],[[0,0],[5,0]]]; 
		#self.walls = []; 

	#Problem Specific
	def buildObs(self,gen=True):
		#cardinal + 1 model
		#left,right,up,down,near

		if(gen):
			self.pz = Softmax(); 
			self.pz.buildRectangleModel([[2,2],[3,4]],10); 
			print('Plotting Observation Model'); 
			self.pz.plot2D(low=[0,0],high=[10,5],vis=True); 
						

			f = open(self.fileNamePrefix + "OBS.npy","w"); 
			np.save(f,self.pz);
		else:
			self.pz = np.load(self.fileNamePrefix + "OBS.npy").tolist(); 
			
			
	#Problem Specific
	def buildReward(self,gen = True):
		if(gen): 

			self.r = [0]*len(self.delA);
			

			for i in range(0,len(self.r)):
				self.r[i] = GM();  

			var = (np.identity(2)*.5).tolist(); 

			for i in range(0,len(self.r)):
				self.r[i].addG(Gaussian([-self.delA[i][0],-self.delA[i][1]],var,10));

			

			print('Plotting Reward Model'); 
			for i in range(0,len(self.r)):
				self.r[i].plot2D(high = [10,5],low = [0,0],xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Reward for action: ' + str(i)); 

			print('Condensing Reward Model');
			for i in range(0,len(self.r)):
				self.r[i] = self.r[i].kmeansCondensationN(k = 5);


			print('Plotting Condensed Reward Model'); 
			for i in range(0,len(self.r)):
				#self.r[i].plot2D(xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Reward for action: ' + str(i)); 
				[x,y,c] = self.r[i].plot2D(high = [10,5],low = [0,0],vis = False);  
	
				minim = np.amin(c); 
				maxim = np.amax(c); 

				#print(minim,maxim); 
				levels = np.linspace(minim,maxim); 
				plt.contourf(x,y,c,levels = levels,vmin = minim,vmax = maxim,cmap = 'viridis');
				plt.title('Reward for action: ' + str(i));
				plt.xlabel('Robot X'); 
				plt.ylabel('Robot Y'); 
				plt.show(); 


			f = open(self.fileNamePrefix + "REW.npy","w"); 
			np.save(f,self.r);

		else:
			self.r = np.load(self.fileNamePrefix + "REW.npy").tolist();




if __name__ == '__main__':
	a = ModelSpec(); 
	a.buildTransition(); 
	a.buildReward(gen = False); 
	a.buildObs(gen = True); 

	

	



