
'''
***************************************************
File: hierarchyTesting.py

Building a toy problem to handle a new method of 
hierarchical pomdps

***************************************************
'''
from __future__ import division

__author__ = "Luke Burks"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Luke Burks"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"


import numpy as np; 
from softmaxModels import Softmax
import scipy.linalg as linalg
from gaussianMixtures import Gaussian,GM
import matplotlib.pyplot as plt

def buildRectangleModel():

	#Specify the lower left and upper right points
	recBounds = [[2,2],[3,4]]; 
	#recBounds = [[1,1],[8,4]]; 

	B = np.matrix([-1,0,recBounds[0][0],1,0,-recBounds[1][0],0,1,-recBounds[1][1],0,-1,recBounds[0][1]]).T; 
	
	M = np.zeros(shape=(12,15)); 

	#Boundry: Left|Near
	rowSB = 0; 
	classNum1 = 1; 
	classNum2 = 0; 
	for i in range(0,3):
		M[3*rowSB+i,3*classNum2+i] = -1; 
		M[3*rowSB+i,3*classNum1+i] = 1; 


	#Boundry: Right|Near
	rowSB = 1; 
	classNum1 = 2; 
	classNum2 = 0; 
	for i in range(0,3):
		M[3*rowSB+i,3*classNum2+i] = -1; 
		M[3*rowSB+i,3*classNum1+i] = 1; 


	#Boundry: Up|Near
	rowSB = 2; 
	classNum1 = 3; 
	classNum2 = 0; 
	for i in range(0,3):
		M[3*rowSB+i,3*classNum2+i] = -1; 
		M[3*rowSB+i,3*classNum1+i] = 1; 

	#Boundry: Down|Near
	rowSB = 3; 
	classNum1 = 4; 
	classNum2 = 0; 
	for i in range(0,3):
		M[3*rowSB+i,3*classNum2+i] = -1; 
		M[3*rowSB+i,3*classNum1+i] = 1; 


	A = np.hstack((M,B)); 
	#print(np.linalg.matrix_rank(A))
	#print(np.linalg.matrix_rank(M))

	Theta = linalg.lstsq(M,B)[0].tolist();  

	weight = []; 
	bias = []; 
	for i in range(0,len(Theta)//3):
		weight.append([Theta[3*i][0],Theta[3*i+1][0]]); 
		bias.append(Theta[3*i+2][0]); 

	steep = 1;
	weight = (np.array(weight)*steep).tolist(); 
	bias = (np.array(bias)*steep).tolist(); 
	pz = Softmax(weight,bias); 
	#print('Plotting Observation Model'); 
	#pz.plot2D(low=[0,0],high=[10,5],vis=True); 


	prior = GM(); 
	for i in range(0,10):
		for j in range(0,5):
			prior.addG(Gaussian([i,j],[[1,0],[0,1]],1)); 
	# prior.addG(Gaussian([4,3],[[1,0],[0,1]],1)); 
	# prior.addG(Gaussian([7,2],[[4,1],[1,4]],3))

	prior.normalizeWeights(); 

	dela = 0.1; 
	x, y = np.mgrid[0:10:dela, 0:5:dela]
	fig,axarr = plt.subplots(6);
	axarr[0].contourf(x,y,prior.discretize2D(low=[0,0],high=[10,5],delta=dela)); 
	axarr[0].set_title('Prior'); 
	titles = ['Inside','Left','Right','Up','Down']; 
	for i in range(0,5):
		post = pz.runVBND(prior,i); 
		c = post.discretize2D(low=[0,0],high=[10,5],delta=dela); 
		axarr[i+1].contourf(x,y,c,cmap='viridis'); 
		axarr[i+1].set_title('Post: ' + titles[i]); 

	plt.show(); 

def buildGeneralModel():
	dims = 2; 

	'''
	#Triangle Specs
	numClasses = 4; 
	boundries = [[1,0],[2,0],[3,0]]; 
	B = np.matrix([-1,1,-1,1,1,-1,0,-1,-1]).T; 
	'''

	
	#Rectangle Specs
	numClasses = 5; 
	boundries = [[1,0],[2,0],[3,0],[4,0]]; 
	recBounds = [[2,2],[3,4]];
	B = np.matrix([-1,0,recBounds[0][0],1,0,-recBounds[1][0],0,1,-recBounds[1][1],0,-1,recBounds[0][1]]).T; 
	

	'''
	#Pentagon Specs
	numClasses = 6; 
	boundries = [[1,0],[2,0],[3,0],[4,0],[5,0]]; 
	B = np.matrix([-1,1,-2,1,1,-2,1,0,-1,0,-1,-2,-1,0,-1]).T; 
	'''

	'''
	#Hexagon Specs
	numClasses = 7; 
	boundries = [[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]]; 
	B = np.matrix([-1,1,-1,0,2,-1,1,1,-1,1,-1,-1,0,-2,-1,-1,-1,-1]).T
	'''

	'''
	#Octogon Specs
	numClasses = 9; 
	boundries = [[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[8,0]]; 
	B = np.matrix([-1,1,-1,0,2,-1,1,1,-1,1,0,-1,1,-1,-1,0,-2,-1,-1,-1,-1,-1,0,-1]).T;
	'''

	M = np.zeros(shape=(len(boundries)*(dims+1),numClasses*(dims+1)));


	for j in range(0,len(boundries)):
		for i in range(0,dims+1):
			M[(dims+1)*j+i,(dims+1)*boundries[j][1]+i] = -1; 
			M[(dims+1)*j+i,(dims+1)*boundries[j][0]+i] = 1; 

	A = np.hstack((M,B)); 
	#print(np.linalg.matrix_rank(A))
	#print(np.linalg.matrix_rank(M))


	Theta = linalg.lstsq(M,B)[0].tolist();

	weight = []; 
	bias = []; 
	for i in range(0,len(Theta)//(dims+1)):
		weight.append([Theta[(dims+1)*i][0],Theta[(dims+1)*i+1][0]]); 
		bias.append(Theta[(dims+1)*i+dims][0]); 

	steep = 10;
	weight = (np.array(weight)*steep).tolist(); 
	bias = (np.array(bias)*steep).tolist(); 
	pz = Softmax(weight,bias); 
	print('Plotting Observation Model'); 
	#pz.plot2D(low=[2.5,2.5],high=[3.5,3.5],delta = 0.1,vis=True); 
	pz.plot2D(low=[0,0],high=[10,5],delta = 0.1,vis=True); 
	#pz.plot2D(low=[-5,-5],high=[5,5],delta = 0.1,vis=True); 

if __name__ == "__main__":
	#buildRectangleModel();
	buildGeneralModel(); 
