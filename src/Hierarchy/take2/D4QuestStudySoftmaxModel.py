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
import yaml

'''
****************************************************
File: D4QuestStudySoftmaxModel.py
Written By: Luke Burks
January 2018


****************************************************
'''

class ModelSpec:

	def __init__(self):
		self.fileNamePrefix = 'D4QuestStudySoftmax'; 
		self.yamlFile = yaml.load(open('../../../models/mapA.yaml')); 

	#Problem specific
	def buildTransition(self):

		#Study
		#self.bounds = [[-7,-2],[-3.33,-1],[-7,-2],[-3.33,-1]];
		r = self.yamlFile['info']['rooms']['study']; 
		self.bounds = [[r['min_x'],r['max_x']],[r['min_y'],r['max_y']],[r['min_x'],r['max_x']],[r['min_y'],r['max_y']]]; 



		self.delAVar = (np.identity(4)*1).tolist(); 
		self.delAVar[0][0] = 0.001; 
		self.delAVar[1][1] = 0.001; 
		#self.delA = [[-0.5,0],[0.5,0],[0,-0.5],[0,0.5],[0,0],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5],[0.5,0.5]]; 
		delta = 1; 
		self.delA = [[-delta,0,0,0],[delta,0,0,0],[0,delta,0,0],[0,-delta,0,0],[0,0,0,0]]; 
		self.discount = 0.95; 


	def buildObs(self,gen=True):


		if(gen):
			self.pz = Softmax(); 

			#Desk
			#self.pz.buildOrientedRecModel([-5.5,-2.0],0,0.61,0.99,5); 
			desk = self.yamlFile['desk']; 
			self.pz.buildOrientedRecModel([desk['centroid_x'],desk['centroid_y']],desk['orientation']+90,desk['x_len'],desk['y_len'],5); 


			for i in range(0,len(self.pz.weights)):
				self.pz.weights[i] = [0,0,self.pz.weights[i][0],self.pz.weights[i][1]]; 
			
			#print('Plotting Observation Model'); 
			#self.pz.plot2D(low=[0,0],high=[10,5],vis=True); 

			f = open("../models/"+self.fileNamePrefix + "OBS.npy","w"); 
			np.save(f,self.pz);

			
			self.pz2 = Softmax(); 

			#filing cabinet
			#self.pz2.buildOrientedRecModel([-3.8638,-1.3262],270,0.5,.37,5); 
			filing = self.yamlFile['filing cabinet']; 
			self.pz2.buildOrientedRecModel([filing['centroid_x'],float(filing['centroid_y'])],filing['orientation']+90,filing['x_len'],filing['y_len'],5); 


			for i in range(0,len(self.pz2.weights)):
				self.pz2.weights[i] = [0,0,self.pz2.weights[i][0],self.pz2.weights[i][1]]; 
			



			#print('Plotting Observation Model'); 
			#self.pz.plot2D(low=[0,0],high=[10,5],vis=True); 

			f = open("../models/"+self.fileNamePrefix + "OBS2.npy","w"); 
			np.save(f,self.pz2);
			



		else:
			self.pz = np.load("../models/"+self.fileNamePrefix + "OBS.npy").tolist(); 
			self.pz2 = np.load("../models/"+self.fileNamePrefix+ "OBS2.npy").tolist(); 


	#Problem Specific
	def buildReward(self,gen = True):
		if(gen): 

			self.r = [0]*len(self.delA);
			

			for i in range(0,len(self.r)):
				self.r[i] = GM();  

			var = (np.identity(4)*5).tolist(); 

			cutFactor = 3;
			for i in range(0,len(self.r)):
				for x1 in range(int(np.floor(self.bounds[0][0]/cutFactor))-1,int(np.ceil(self.bounds[0][1]/cutFactor))+1):
					for y1 in range(int(np.floor(self.bounds[1][0]/cutFactor))-1,int(np.ceil(self.bounds[1][1]/cutFactor))+1):
						for x2 in range(int(np.floor(self.bounds[2][0]/cutFactor))-1,int(np.ceil(self.bounds[2][1]/cutFactor))+1):
							for y2 in range(int(np.floor(self.bounds[3][0]/cutFactor))-1,int(np.ceil(self.bounds[3][1]/cutFactor))+1):
								if(np.sqrt((x1-x2)**2 + (y1-y2)**2) < 1):
									self.r[i].addG(Gaussian(np.array(([x1*cutFactor,y1*cutFactor,x2*cutFactor,y2*cutFactor])-np.array(self.delA[i])).tolist(),var,100));



			# for r in self.r:
			# 	r.display(); 

			f = open("../models/"+self.fileNamePrefix + "REW.npy","w"); 
			np.save(f,self.r);

		else:
			self.r = np.load("../models/"+self.fileNamePrefix + "REW.npy").tolist();



if __name__ == '__main__':
	a = ModelSpec(); 
	a.buildTransition(); 
	a.buildReward(gen = True); 
	a.buildObs(gen = True); 
