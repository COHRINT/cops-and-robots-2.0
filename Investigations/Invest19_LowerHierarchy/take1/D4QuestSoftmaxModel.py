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




****************************************************
'''

class ModelSpec:

	def __init__(self):
		self.fileNamePrefix = 'D4QuestSoftmax'; 


	#Problem specific
	def buildTransition(self):
		#self.bounds = [[0,10],[0,5],[0,10],[0,5]]; 
		#Hallway
		#self.bounds = [[-9.5,4],[-1,1.4],[-9.5,4],[-1,1.4]];
		#Dining Room
		#self.bounds = [[-9.5,-7],[-3.33,-1],[-9.5,-7],[-3.33,-1]] 
		#Study
		#self.bounds = [[-7,-2],[-3.33,-1],[-7,-2],[-3.33,-1]];
		#Library
		#self.bounds = [[-2,4],[-3.33,-1],[-2,4],[-3.33,-1]];
		#Billiard
		#self.bounds = [[0,4],[1.4,3.68],[0,4],[1.4,3.68]];
		#Kitchen
		self.bounds = [[-9.5,0],[1.4,3.68],[-9.5,0],[1.4,3.68]];

		self.delAVar = (np.identity(4)*1).tolist(); 
		self.delAVar[0][0] = 0.00001; 
		self.delAVar[1][1] = 0.00001; 
		#self.delA = [[-0.5,0],[0.5,0],[0,-0.5],[0,0.5],[0,0],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5],[0.5,0.5]]; 
		delta = 1; 
		self.delA = [[-delta,0,0,0],[delta,0,0,0],[0,delta,0,0],[0,-delta,0,0],[0,0,0,0]]; 
		self.discount = 0.95; 


	def buildObs(self,gen=True):


		if(gen):
			self.pz = Softmax(); 

			#Vern
			#self.pz.buildOrientedRecModel([-2.475,1.06],270,0.5,0.5,5); 

			#Desk
			#self.pz.buildOrientedRecModel([-5.5,-2.0],0,0.61,0.99,5); 

			#bookcase
			#self.pz.buildOrientedRecModel([0,-1.1662],270,0.38,0.18,5); 

			#checkers
			#self.pz.buildOrientedRecModel([2.04,2.16],270,0.5,0.5,5); 

			#Fridge
			self.pz.buildOrientedRecModel([-9.1,3.07],315,0.46,0.46,5); 

			for i in range(0,len(self.pz.weights)):
				self.pz.weights[i] = [0,0,self.pz.weights[i][0],self.pz.weights[i][1]]; 
			
			#print('Plotting Observation Model'); 
			#self.pz.plot2D(low=[0,0],high=[10,5],vis=True); 

			f = open(self.fileNamePrefix + "OBS.npy","w"); 
			np.save(f,self.pz);

			
			self.pz2 = Softmax(); 

			#filing cabinet
			#self.pz2.buildOrientedRecModel([-3.8638,-1.3262],270,0.5,.37,5); 
			
			#chair
			#self.pz2.buildOrientedRecModel([2.975,-2.435],90,0.46,0.41,5); 

			#cassini
			#self.pz2.buildOrientedRecModel([1.38,3.475],270,0.05,0.56,5); 

			#mars
			self.pz2.buildOrientedRecModel([-4.38,3.475],270,0.05,0.84,5); 


			for i in range(0,len(self.pz2.weights)):
				self.pz2.weights[i] = [0,0,self.pz2.weights[i][0],self.pz2.weights[i][1]]; 
			



			#print('Plotting Observation Model'); 
			#self.pz.plot2D(low=[0,0],high=[10,5],vis=True); 

			f = open(self.fileNamePrefix + "OBS2.npy","w"); 
			np.save(f,self.pz2);
			



		else:
			self.pz = np.load(self.fileNamePrefix + "OBS.npy").tolist(); 
			self.pz2 = np.load(self.fileNamePrefix+ "OBS2.npy").tolist(); 


	#Problem Specific
	def buildReward(self,gen = True):
		if(gen): 

			self.r = [0]*len(self.delA);
			

			for i in range(0,len(self.r)):
				self.r[i] = GM();  

			var = (np.identity(4)*5).tolist(); 

			#Need gaussians along the borders for positive and negative rewards
			for i in range(0,len(self.r)):
				for x1 in range(int(np.floor(self.bounds[0][0]))-1,int(np.ceil(self.bounds[0][1]))+1):
					for y1 in range(int(np.floor(self.bounds[1][0]))-1,int(np.ceil(self.bounds[1][1]))+1):
						for x2 in range(int(np.floor(self.bounds[2][0]))-1,int(np.ceil(self.bounds[2][1]))+1):
							for y2 in range(int(np.floor(self.bounds[3][0]))-1,int(np.ceil(self.bounds[3][1]))+1):
								if(np.sqrt((x1-x2)**2 + (y1-y2)**2) < 1):
									self.r[i].addG(Gaussian(np.array(([x1,y1,x2,y2])-np.array(self.delA[i])).tolist(),var,10)); 
									


			for r in self.r:
				r.display(); 

			f = open(self.fileNamePrefix + "REW.npy","w"); 
			np.save(f,self.r);

		else:
			self.r = np.load(self.fileNamePrefix + "REW.npy").tolist();



if __name__ == '__main__':
	a = ModelSpec(); 
	a.buildTransition(); 
	a.buildReward(gen = True); 
	a.buildObs(gen = True); 
