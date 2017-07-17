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
File: D4QuestSoftmaxModel.py
Written By: Luke Burks
Febuary 2017

Container Class for problem specific models
Model: Question asking POMDP
A single room, with a bookcase in the right spot
Softmax Observations are question dependent

State specified over Cop_x, Cop_y, Robber_x, Robber_y
Bounds from [0,10] by [0,5] on x and y dimensions 

Rewards for collocation of the cop and robber
Transitions are very certain in cop dimension

So, for the COP, actually have him go to the MAP state?
And then doctor belie


****************************************************
'''

class ModelSpec:

	def __init__(self):
		self.fileNamePrefix = 'D2QuestSoftmax'; 


	#Problem specific
	def buildTransition(self):
		self.bounds = [[0,10],[0,5],[0,10],[0,5]]; 
		self.delAVar = (np.identity(4)*4).tolist(); 
		self.delAVar[0][0] = 0.00001; 
		self.delAVar[1][1] = 0.00001; 
		#self.delA = [[-0.5,0],[0.5,0],[0,-0.5],[0,0.5],[0,0],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5],[0.5,0.5]]; 
		delta = 1; 
		self.delA = [[-delta,0,0,0],[delta,0,0,0],[0,delta,0,0],[0,-delta,0,0],[0,0,0,0]]; 
		self.discount = 0.95; 


	def buildObs(self,gen=True):
		#cardinal + 1 model
		#left,right,up,down,near

		if(gen):
			self.pz = Softmax(); 
			self.pz.buildRectangleModel([[3,4.5],[5,5]],1); 
			for i in range(0,len(self.pz.weights)):
				self.pz.weights[i] = [0,0,self.pz.weights[i][0],self.pz.weights[i][1]]; 
			
			#print('Plotting Observation Model'); 
			#self.pz.plot2D(low=[0,0],high=[10,5],vis=True); 
						
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

			var = (np.identity(4)*1).tolist(); 

			#Need gaussians along the borders for positive and negative rewards
			for i in range(0,len(self.r)):
				for x1 in range(self.bounds[0][0],self.bounds[1][0]):
					for y1 in range(self.bounds[0][1],self.bounds[1][1]):
						for x2 in range(self.bounds[0][0],self.bounds[1][0]):
							for y2 in range(self.bounds[0][1],self.bounds[1][1]):
								if(math.sqrt((x1-x2)**2 + (y1-y2)**2) < 1):
									self.r[i].addG(Gaussian([x1,y1,x2,y2],var,1)); 



			f = open(self.fileNamePrefix + "REW.npy","w"); 
			np.save(f,self.r);

		else:
			self.r = np.load(self.fileNamePrefix + "REW.npy").tolist();



if __name__ == '__main__':
	a = ModelSpec(); 
	a.buildTransition(); 
	a.buildReward(gen = True); 
	#a.buildObs(gen = True); 
