

'''
######################################################

File: roomsProblemSpec.py
Author: Luke Burks
Date: July 2017

Specs out the upper level POMDP for the cops and robots
experiment


6 States: Kitchen, Dining Room, Hallway, Study,
Library, Billiard Room

Actions split into movements and questions
6 movement actions: 1 per room
6 question actions: 1 per room
no state change

2 observations: detect/no detect
action dependent
bad sensor, so like .1 chance of true detection


Rewards: 100 for taking the action associated with the 
correct state
-10 for taking every other action

rooms: 
0:Kitchen
1:Dining Room
2:Hallway
3:Study
4:Library
5:Billiard Room

######################################################
'''

import numpy as np; 
import matplotlib.pyplot as plt;

class ModelSpec:

	def __init__(self):
		self.discount = .9; 
		self.N = 6; 
		self.acts = 36; 
		self.obs = 3; 

		self.buildTransition(); 
		self.buildObservations(); 
		self.buildRewards(); 

		

	def buildTransition(self):
		self.px = np.zeros(shape=(self.acts,self.N,self.N)).tolist(); 
		'''
		rooms: 
		0:Kitchen
		1:Dining Room
		2:Hallway
		3:Study
		4:Library
		5:Billiard Room
		'''
		'''
		for i in range(0,self.N):
			for j in range(0,self.N):
				for a in range(0,self.acts):
					self.px[a][i][i] = 1; 
		'''

		for a in range(0,self.acts):
			#K|H
			self.px[a][0][2] = .2; 
			self.px[a][2][0] = .2/5;  

			#B|H
			self.px[a][5][2] = .2; 
			self.px[a][2][5] = .2/5;

			#L|H
			self.px[a][4][2] = .1; 
			self.px[a][2][4] = .2/5; 

			#L|S
			self.px[a][4][3] = .1; 
			self.px[a][3][4] = .1; 

			#S|H
			self.px[a][3][2] = .1; 
			self.px[a][2][3] = .2/5; 

			#D|H
			self.px[a][1][2] = .2; 
			self.px[a][2][1] = .2/5; 

			#selves
			self.px[a][0][0] = .8; 
			self.px[a][1][1] = .8; 
			self.px[a][2][2] = .8; 
			self.px[a][3][3] = .8; 
			self.px[a][4][4] = .8; 
			self.px[a][5][5] = .8;  


	def buildObservations(self):

		self.pz = np.zeros(shape = (self.obs,self.acts,self.N)).tolist(); 
		for i in range(0,self.N):
			for a in range(0,self.acts):
				am = a//6; 
				aq = a%6; 

				if(aq==i):
					self.pz[1][a][i] = .1; 
					self.pz[0][a][i] = .3;
					self.pz[2][a][i] = .6;  
				else:
					self.pz[1][a][i] = .01;
					self.pz[0][a][i] = .39;
					self.pz[2][a][i] = .6;  

					

	def buildRewards(self):
		self.R = np.zeros(shape=(self.acts,self.N)).tolist(); 

		for i in range(0,self.N):
			for a in range(0,self.acts):
				am = a//6; 
				aq = a%6; 
				if(i==am):
					self.R[a][i] = 10; 
				else:
					self.R[a][i] = -10; 

		


def checkReward(m):
	x = [i for i in range(0,m.N)]; 
	plt.plot(x,m.R[0],c='r',linewidth=5);
	plt.plot(x,m.R[1],c='b',linewidth=5); 
	plt.plot(x,m.R[2],c='g',linewidth=5);  
	plt.legend(['Left','Right','Stay']); 
	plt.title('Hallway Problem Reward Function'); 
	plt.show()

def checkObs(m):
	x = [i for i in range(0,m.N)]; 
	plt.plot(x,m.pz[0],c='r',linewidth=5);
	plt.plot(x,m.pz[1],c='b',linewidth=5); 
	plt.plot(x,m.pz[2],c='g',linewidth=5); 
	plt.plot(x,m.pz[3],c='k',linewidth=5); 
	plt.plot(x,m.pz[4],c='y',linewidth=5); 
	plt.legend(['Left','Right','Near','Far Left','Far Right']); 
	plt.title('Hallway Problem Observation Model'); 
	plt.show()

def checkTransition(m):
	x0 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	x1 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	x2 = [[0 for i in range(0,m.N)] for j in range(0,m.N)]; 
	
	x, y = np.mgrid[0:m.N:1, 0:m.N:1]


	for i in range(0,m.N):
		for j in range(0,m.N):
			x0[i][j] = m.px[0][i][j]; 
			x1[i][j] = m.px[1][i][j]; 
			x2[i][j] = m.px[2][i][j]; 


	fig,axarr = plt.subplots(3); 
	axarr[0].contourf(x,y,x0); 
	axarr[1].contourf(x,y,x1); 
	axarr[2].contourf(x,y,x2); 
	plt.show(); 
	

if __name__ == "__main__":
	m = ModelSpec(); 
	
	#checkReward(m);
	#checkObs(m); 
	#checkTransition(m); 

