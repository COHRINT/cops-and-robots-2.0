
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
from numpy.linalg import svd
import math
from copy import deepcopy
import matplotlib.animation as animation
import os
import matplotlib.image as mgimg
import sys; 
import time

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

	'''
	#Rectangle Specs
	numClasses = 4; 
	boundries = [[1,0],[2,0],[3,0],[4,0]]; 
	recBounds = [[2,2],[3,4]];
	#B = np.matrix([-1,0,recBounds[0][0],1,0,-recBounds[1][0],0,1,-recBounds[1][1],0,-1,recBounds[0][1]]).T; 
	B = np.matrix([0.44721359549995826, -2.220446049250313e-16, -0.8944271909999157, -0.0, 0.24253562503633294, -0.9701425001453319, 0.316227766016838, -5.551115123125783e-17, -0.948683298050514, 0.0, -0.447213595499958, 0.8944271909999159]).T; 
	'''


	
	#Pentagon Specs
	numClasses = 6; 
	boundries = [[1,0],[2,0],[3,0],[4,0],[5,0]]; 
	B = np.matrix([-1,1,-2,1,1,-2,1,0,-1,0,-1,-2,-1,0,-1]).T; 
	

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

	steep = 5;
	weight = (np.array(weight)*steep).tolist(); 
	bias = (np.array(bias)*steep).tolist(); 
	pz = Softmax(weight,bias); 
	print('Plotting Observation Model'); 
	#pz.plot2D(low=[2.5,2.5],high=[3.5,3.5],delta = 0.1,vis=True); 
	#pz.plot2D(low=[0,0],high=[10,5],delta = 0.1,vis=True); 
	pz.plot2D(low=[-5,-5],high=[5,5],delta = 0.1,vis=True); 


def nullspace(A,atol=1e-13,rtol=0):
	A = np.atleast_2d(A)
	u, s, vh = svd(A)
	tol = max(atol, rtol * s[0])
	nnz = (s >= tol).sum()
	ns = vh[nnz:].conj().T
	return ns;

def distance(x1,y1,x2,y2):
	dist = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2); 
	dist = math.sqrt(dist); 
	return dist; 


def buildPointsModel():
	#Ok, so the idea here is that given some points you can 
	#find the normal for each edge. 
	dims = 2;
	#points = [[2,2],[2,4],[3,4],[3,2]]; 
	#points = [[-2,-2],[-2,-1],[0,1],[2,-1],[2,-2]];
	#points = [[1,1],[1,2],[3,2],[6,1],[4,-1]];  
	points = [[1,1],[3,5],[4,1],[3,0],[4,-2]]; 
	pointsx = [p[0] for p in points]; 
	pointsy = [p[1] for p in points]; 
	centroid = [sum(pointsx)/len(points),sum(pointsy)/len(points)];

	#for each point to the next, find the normal  between them.
	B = []; 
	for i in range(0,len(points)):
		p1 = points[i]; 
		 
		if(i == len(points)-1): 
			p2 = points[0]; 
		else:
			p2 = points[i+1]; 
		mid = []; 
		for i in range(0,len(p1)):
			mid.append((p1[i]+p2[i])/2)

		H = np.matrix([[p1[0],p1[1],1],[p2[0],p2[1],1],[mid[0],mid[1],1]]); 

		#print(H);
		#print(nullspace(H).T[0]); 
		#print("");  
		Hnull = (nullspace(H)).tolist();
		distMed1 = distance(mid[0]+Hnull[0][0],mid[1]+Hnull[1][0],centroid[0],centroid[1]); 
		distMed2 = distance(mid[0]-Hnull[0][0],mid[1]-Hnull[1][0],centroid[0],centroid[1]);
		if(distMed1 < distMed2):
			Hnull[0][0] = -Hnull[0][0];
			Hnull[1][0] = -Hnull[1][0];
			Hnull[2][0] = -Hnull[2][0]; 

		for j in Hnull:
			B.append(j[0]);
 	
	B = np.matrix(B).T;  
	 
	numClasses = len(points)+1; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	#boundries = [[1,0],[2,0],[3,0],[4,0],[5,0]];
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

	steep = 5;
	weight = (np.array(weight)*steep).tolist(); 
	bias = (np.array(bias)*steep).tolist(); 
	pz = Softmax(weight,bias); 
	print('Plotting Observation Model'); 
	#pz.plot2D(low=[2.5,2.5],high=[3.5,3.5],delta = 0.1,vis=True); 
	#pz.plot2D(low=[0,0],high=[10,5],delta = 0.1,vis=True); 
	#pz.plot2D(low=[-5,-5],high=[5,5],delta = 0.1,vis=True); 	
	pz.plot2D(low=[-10,-10],high=[10,10],delta = 0.1,vis=True); 	


def stretch1DModel():
	steep = 5; 
	weight = (np.array([-3,-2,-1,0])*steep).tolist(); 
	bias = (np.array([6,5,3,0])*steep).tolist(); 

	low = 0; 
	high = 5; 
	res = 100; 

	#Define Likelihood Model
	a = Softmax(weight,bias);
	a.plot1D(low = low, high =high); 

	#weight = (np.array([[-1,0],[0,0]])*steep).tolist(); 
	#bias = (np.array([3,0])*steep).tolist(); 
	for i in range(0,len(weight)):
		weight[i] = [weight[i],0];

	b = Softmax(weight,bias); 
	b.plot2D(low=[0,0],high=[5,5]); 



def slice3DModel():
	steep = 1; 

	dims = 3;

	
	#Trapezoidal Pyramid Specs
	numClasses = 7; 
	boundries = [[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]]; 
	B = np.matrix([0,0,-1,-2,-1,0,.5,-2,0,1,.5,-2,1,0,.5,-2,0,-1,.5,-2,0,0,1,-1]).T; 
	
	'''
	#Octohedron Specs
	numClasses = 9; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	B = np.matrix([-1,-1,0.5,-1,-1,1,0.5,-1,1,1,0.5,-1,1,-1,0.5,-1,-1,-1,-0.5,-1,-1,1,-0.5,-1,1,1,-0.5,-1,1,-1,-0.5,-1]).T; 
	'''

	M = np.zeros(shape=(len(boundries)*(dims+1),numClasses*(dims+1)));


	for j in range(0,len(boundries)):
		for i in range(0,dims+1):
			M[(dims+1)*j+i,(dims+1)*boundries[j][1]+i] = -1; 
			M[(dims+1)*j+i,(dims+1)*boundries[j][0]+i] = 1; 

	A = np.hstack((M,B)); 

	Theta = linalg.lstsq(M,B)[0].tolist();

	weight = []; 
	bias = []; 
	for i in range(0,len(Theta)//(dims+1)):
		weight.append([Theta[(dims+1)*i][0],Theta[(dims+1)*i+1][0],Theta[(dims+1)*i+2][0]]); 
		bias.append(Theta[(dims+1)*i+dims][0]); 

	steep = 10;
	weight = (np.array(weight)*steep).tolist(); 
	bias = (np.array(bias)*steep).tolist(); 
	pz = Softmax(weight,bias); 

	print('Plotting Observation Model'); 
	#pz.plot2D(low=[2.5,2.5],high=[3.5,3.5],delta = 0.1,vis=True); 
	#pz.plot2D(low=[0,0],high=[10,5],delta = 0.1,vis=True); 
	#pz.plot2D(low=[-5,-5],high=[5,5],delta = 0.1,vis=True); 

	pz2 = Softmax(deepcopy(weight),deepcopy(bias));
	pz3 = Softmax(deepcopy(weight),deepcopy(bias));
	pz4 = Softmax(deepcopy(weight),deepcopy(bias));

	for i in range(0,len(pz2.weights)):
		pz2.weights[i] = [pz2.weights[i][0],pz2.weights[i][2]]

	for i in range(0,len(pz3.weights)):
		pz3.weights[i] = [pz3.weights[i][1],pz3.weights[i][2]]

	for i in range(0,len(pz4.weights)):
		pz4.weights[i] = [pz4.weights[i][0],pz4.weights[i][1]]

	fig = plt.figure(); 
	[x,y,c] = pz2.plot2D(low=[-5,-5],high=[5,5],vis = False); 
	plt.contourf(x,y,c); 
	plt.xlabel('X Axis'); 
	plt.ylabel('Z Axis'); 
	plt.title('Slice Across Y Axis')

	fig = plt.figure(); 
	[x,y,c] = pz3.plot2D(low=[-5,-5],high=[5,5],vis = False); 
	plt.contourf(x,y,c); 
	plt.xlabel('Y Axis'); 
	plt.ylabel('Z Axis');
	plt.title('Slice Across X axis')

	fig = plt.figure(); 
	[x,y,c] = pz4.plot2D(low=[-5,-5],high=[5,5],vis = False); 
	plt.contourf(x,y,c); 
	plt.xlabel('X Axis'); 
	plt.ylabel('Y Axis');
	plt.title('Slice Across Z Axis'); 


	fig = plt.figure(); 
	ax = fig.add_subplot(111,projection='3d'); 
	ax.set_xlabel('X Axis'); 
	ax.set_ylabel('Y Axis'); 
	ax.set_zlabel('Z Axis'); 
	ax.set_xlim([-5,5]); 
	ax.set_ylim([-5,5]); 
	ax.set_zlim([-5,5]); 
	ax.set_title("3D Scatter of Softmax Class Dominance Regions")

	
	for clas in range(1,numClasses):
		shapeEdgesX = []; 
		shapeEdgesY = [];
		shapeEdgesZ = []; 
		#-5 to 5 on all dims
		data = np.zeros(shape=(21,21,21)); 
		for i in range(0,21):
			for j in range(0,21):
				for k in range(0,21):
					data[i][j][k] = pz.pointEvalND(clas,[(i-10)/2,(j-10)/2,(k-10)/2]);
					if(data[i][j][k] > 0.1):
						shapeEdgesX.append((i-10)/2); 
						shapeEdgesY.append((j-10)/2); 
						shapeEdgesZ.append((k-10)/2);   

		ax.scatter(shapeEdgesX,shapeEdgesY,shapeEdgesZ); 
	

	plt.show(); 
	#fig = plt.figure(); 


def buildRecFromCentroidOrient():
	centroid = [5,4]; 
	orient = 0;
	length = 2; 
	width = 4; 
	theta1 = orient*math.pi/180;  
	

	h = math.sqrt((width/2)*(width/2) + (length/2)*(length/2)); 
	theta2 = math.asin((width/2)/h); 
	 
	s1 = h*math.sin(theta1+theta2); 
	s2 = h*math.cos(theta1+theta2); 

	s3 = h*math.sin(theta1-theta2); 
	s4 = h*math.cos(theta1-theta2); 


	pz = Softmax(); 

	points = [];
	points = [[centroid[0]+s2,centroid[1]+s1],[centroid[0]+s4,centroid[1]+s3],[centroid[0]-s2,centroid[1]-s1],[centroid[0]-s4,centroid[1]-s3]]; 
	
	for p in points:
		plt.scatter(p[0],p[1]); 
	plt.show(); 

	pz.buildPointsModel(points,steepness=5); 
	pz.plot2D(low=[0,0],high=[10,10]); 


def buildTriView():

	pose = [2,1,165];
	l = 3;
	#Without Cutting
	triPoints = [[pose[0],pose[1]],[pose[0]+l*math.cos(2*-0.261799+math.radians(pose[2])),pose[1]+l*math.sin(2*-0.261799+math.radians(pose[2]))],[pose[0]+l*math.cos(2*0.261799+math.radians(pose[2])),pose[1]+l*math.sin(2*0.261799+math.radians(pose[2]))]];
	
	#With Cutting
	lshort = 0.5
	triPoints = [[pose[0]+lshort*math.cos(2*0.261799+math.radians(pose[2])),pose[1]+lshort*math.sin(2*0.261799+math.radians(pose[2]))],[pose[0]+lshort*math.cos(2*-0.261799+math.radians(pose[2])),pose[1]+lshort*math.sin(2*-0.261799+math.radians(pose[2]))],[pose[0]+l*math.cos(2*-0.261799+math.radians(pose[2])),pose[1]+l*math.sin(2*-0.261799+math.radians(pose[2]))],[pose[0]+l*math.cos(2*0.261799+math.radians(pose[2])),pose[1]+l*math.sin(2*0.261799+math.radians(pose[2]))]];


	pz = Softmax(); 
	pz.buildPointsModel(triPoints,steepness=10); 
	pz.plot2D(low=[-10,-10],high=[10,10]); 

def make3DSoftmaxAnimation():
	dims = 3;

	
	#Trapezoidal Pyramid Specs
	numClasses = 7; 
	boundries = [[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]]; 
	B = np.matrix([0,0,-1,-2,-1,0,.5,-2,0,1,.5,-2,1,0,.5,-2,0,-1,.5,-2,0,0,1,-2]).T/4; 
	
	'''
	#Octohedron Specs
	numClasses = 9; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	B = np.matrix([-1,-1,0.5,-1,-1,1,0.5,-1,1,1,0.5,-1,1,-1,0.5,-1,-1,-1,-0.5,-1,-1,1,-0.5,-1,1,1,-0.5,-1,1,-1,-0.5,-1]).T; 
	'''

	M = np.zeros(shape=(len(boundries)*(dims+1),numClasses*(dims+1)));


	for j in range(0,len(boundries)):
		for i in range(0,dims+1):
			M[(dims+1)*j+i,(dims+1)*boundries[j][1]+i] = -1; 
			M[(dims+1)*j+i,(dims+1)*boundries[j][0]+i] = 1; 

	A = np.hstack((M,B)); 

	Theta = linalg.lstsq(M,B)[0].tolist();

	weight = []; 
	bias = []; 
	for i in range(0,len(Theta)//(dims+1)):
		weight.append([Theta[(dims+1)*i][0],Theta[(dims+1)*i+1][0],Theta[(dims+1)*i+2][0]]); 
		bias.append(Theta[(dims+1)*i+dims][0]); 

	steep = 10;
	weight = (np.array(weight)*steep).tolist(); 
	bias = (np.array(bias)*steep).tolist(); 
	pz = Softmax(weight,bias); 

	fig = plt.figure(); 
	ax = fig.add_subplot(111,projection='3d'); 
	ax.set_xlabel('X Axis'); 
	ax.set_ylabel('Y Axis'); 
	ax.set_zlabel('Z Axis'); 
	ax.set_xlim([-5,5]); 
	ax.set_ylim([-5,5]); 
	ax.set_zlim([-5,5]); 
	#ax.set_title("3D Scatter of Softmax Class Dominance Regions")



	dataClear = np.zeros(shape=(numClasses,21,21,21)); 
	edgesClear = np.empty(shape=(numClasses,3,0)).tolist(); 
	for clas in range(0,numClasses):
		#-5 to 5 on all dims
		data = np.zeros(shape=(21,21,21)); 
		for i in range(0,21):
			for j in range(0,21):
				for k in range(0,21):
					dataClear[clas][i][j][k] = pz.pointEvalND(clas,[(i-10)/2,(j-10)/2,(k-10)/2]);
					if(dataClear[clas][i][j][k] > 0.0001):
						edgesClear[clas][0].append((i-10)/2); 
						edgesClear[clas][1].append((j-10)/2); 
						edgesClear[clas][2].append((k-10)/2);   

	


	count = 0; 

	allCols = ['b','g','r','c','m','y','w','#eeefff','#ffa500']; 

	edgesZ = np.empty(shape=(numClasses,20,3,0)).tolist();
	#Z Axis


	for slic in range(-10,10):
		for clas in range(0,1):
			#-5 to 5 on all dims
			data = np.zeros(shape=(101,101)); 
			for i in range(0,101):
				for j in range(0,101):
						data[i][j] = pz.pointEvalND(clas,[(i-50)/10,(j-50)/10,slic/5]);
						if(data[i][j] > 0.1):
							edgesZ[clas][slic+10][0].append((i-50)/10); 
							edgesZ[clas][slic+10][1].append((j-50)/10); 
							edgesZ[clas][slic+10][2].append(slic/5);   
			ax.cla(); 
			ax.set_xlabel('X Axis'); 
			ax.set_ylabel('Y Axis'); 
			ax.set_zlabel('Z Axis'); 
			ax.set_xlim([-5,5]); 
			ax.set_ylim([-5,5]); 
			ax.set_zlim([-5,5]); 
			#ax.set_title("3D Scatter of Softmax Class Dominance Regions")
		for i in range(0,numClasses):
			#ax.scatter(edgesClear[i][0],edgesClear[i][1],edgesClear[i][2],alpha=0.15,color=allCols[i]); 
			ax.scatter(edgesZ[i][slic+10][0],edgesZ[i][slic+10][1],edgesZ[i][slic+10][2],color=allCols[i]); 
			
		fig.savefig(os.path.dirname(__file__) + '/tmp/img'+str(count)+".png",bbox_inches='tight',pad_inches=0);
		count+=1;  
		plt.pause(0.1); 
	'''
	#X axis
	for slic in range(-10,10):
		shapeEdgesX = []; 
		shapeEdgesY = [];
		shapeEdgesZ = [];
		#-5 to 5 on all dims
		data = np.zeros(shape=(101,101)); 
		for i in range(0,101):
			for j in range(0,101):
					data[i][j] = pz.pointEvalND(0,[slic/5,(i-50)/10,(j-50)/10]);
					if(data[i][j] > 0.0001):
						shapeEdgesY.append((i-50)/10); 
						shapeEdgesZ.append((j-50)/10); 
						shapeEdgesX.append(slic/5);   
		ax.cla(); 
		ax.set_xlabel('X Axis'); 
		ax.set_ylabel('Y Axis'); 
		ax.set_zlabel('Z Axis'); 
		ax.set_xlim([-5,5]); 
		ax.set_ylim([-5,5]); 
		ax.set_zlim([-5,5]); 
		#ax.set_title("3D Scatter of Softmax Class Dominance Regions")
		ax.scatter(shapeEdgesXClear,shapeEdgesYClear,shapeEdgesZClear,alpha=0.01,color='b')
		ax.scatter(shapeEdgesX,shapeEdgesY,shapeEdgesZ,color='k'); 
		fig.savefig(os.path.dirname(__file__) + '/tmp/img'+str(count)+".png",bbox_inches='tight',pad_inches=0);
		count+=1;  
		plt.pause(0.1); 

	#Y axis
	for slic in range(-10,10):
		shapeEdgesX = []; 
		shapeEdgesY = [];
		shapeEdgesZ = [];
		#-5 to 5 on all dims
		data = np.zeros(shape=(101,101)); 
		for i in range(0,101):
			for j in range(0,101):
					data[i][j] = pz.pointEvalND(0,[(i-50)/10,slic/5,(j-50)/10]);
					if(data[i][j] > 0.0001):
						shapeEdgesX.append((i-50)/10); 
						shapeEdgesZ.append((j-50)/10); 
						shapeEdgesY.append(slic/5);   
		ax.cla(); 
		ax.set_xlabel('X Axis'); 
		ax.set_ylabel('Y Axis'); 
		ax.set_zlabel('Z Axis'); 
		ax.set_xlim([-5,5]); 
		ax.set_ylim([-5,5]); 
		ax.set_zlim([-5,5]); 
		#ax.set_title("3D Scatter of Softmax Class Dominance Regions")
		ax.scatter(shapeEdgesXClear,shapeEdgesYClear,shapeEdgesZClear,alpha=0.01,color='b')
		ax.scatter(shapeEdgesX,shapeEdgesY,shapeEdgesZ,color='k'); 
		fig.savefig(os.path.dirname(__file__) + '/tmp/img'+str(count)+".png",bbox_inches='tight',pad_inches=0);
		count+=1;  
		plt.pause(0.1); 
	'''
	#Animate Results
	fig,ax=plt.subplots()
	images=[]
	for k in range(0,count):
		fname=os.path.dirname(__file__) + '/tmp/img%d.png' %k
		img=mgimg.imread(fname)
		imgplot=plt.imshow(img)
		plt.axis('off')
		images.append([imgplot])
	ani=animation.ArtistAnimation(fig,images,interval=20);
	ani.save('trapezoidalAllClass3.gif',fps=3,writer='animation.writer')



def buildRadialSoftmaxModels():
	dims = 2;


	#Target Model
	steep = 2; 
	weight = (np.array([-2,-1,0])*steep).tolist(); 
	bias = (np.array([6,4,0])*steep).tolist(); 

	for i in range(0,len(weight)):
		weight[i] = [weight[i],0];
	pz = Softmax(weight,bias); 
	obsOffset = [-7,-4];
	observation = 2; 

	#pz.plot2D(low=[0,0],high=[1,6.28]); 
	'''
	H = np.matrix([[2,math.pi*2,1],[2,math.pi*3/4,1],[2,math.pi/2,1]]); 
	print(nullspace(H)); 


	#Modified Target Model
	#def buildGeneralModel(self,dims,numClasses,boundries,B,steepness=1):
	B = np.matrix([0.447,0,-0.8944,0.447,0,-0.894]).T;
	boundries = [[1,0],[2,0]]; 
	pz = Softmax(); 
	pz.buildGeneralModel(2,3,boundries,B,steepness=2); 
	'''

	'''
	cent = [0,0]; 
	length = 3; 
	width = 2; 
	orient = 0; 

	pz = Softmax(); 
	pz.buildOrientedRecModel(cent,orient,length,width,steepness=5); 
	'''

	print('Plotting Observation Model'); 
	[xobs,yobs,domObs] = plot2DPolar(pz,low=[-10,-10],high=[10,10],delta=0.1,offset=obsOffset,vis=False);  
	[xobsPol,yobsPol,domObsPol] = pz.plot2D(low=[0,-3.14],high=[10,3.14],delta=0.1,vis=False); 


	# fig = plt.figure()
	# ax = fig.gca(projection='3d');
	# colors = ['b','g','r','c','m','y','k','w','b','g']; 
	# for i in range(0,len(model)):
	# 	ax.plot_surface(x,y,model[i],color = colors[i]); 
	
	# plt.show(); 

	#pz.plot2D(low=[-10,-10],high=[10,10]); 

	scaling = o1
	bcart = GM();
	for i in range(-10,11): 
		for j in range(-10,11):
			# if(i != 0 or j != 0):
			bcart.addG(Gaussian([i,j],[[scaling,0],[0,scaling]],1)); 

	bcart.normalizeWeights();
	[xpri,ypri,cpri] = bcart.plot2D(low=[-10,-10],high=[10,10],vis = False);
	bpol = transformCartToPol(bcart,obsOffset); 

	for i in range(0,3):
		bpolPrime = pz.runVBND(bpol,i); 
		bcartPrime = transformPolToCart(bpolPrime,obsOffset); 
		bcartPrime.normalizeWeights()
		[xpos,ypos,cpos] = bcartPrime.plot2D(low=[-10,-10],high=[10,10],vis = False); 

	
		fig,axarr = plt.subplots(3);
		axarr[0].contourf(xpri,ypri,cpri); 
		axarr[0].set_ylabel("Prior"); 
		axarr[1].contourf(xobs,yobs,domObs);
		axarr[1].set_ylabel("Observation: " + str(i)); 
		axarr[2].contourf(xpos,ypos,cpos); 
		axarr[2].set_ylabel("Posterior");

		plt.show(); 
	

	fig,axarr = plt.subplots(1,2); 
	axarr[0].contourf(xobs,yobs,domObs); 
	axarr[1].contourf(xobsPol,yobsPol,domObsPol); 
	axarr[0].set_title("Cartesian Observations");
	axarr[1].set_title("Polar Observations"); 
	axarr[0].set_xlabel("X"); 
	axarr[0].set_ylabel("Y"); 
	axarr[1].set_xlabel("Radius (r)"); 
	axarr[1].set_ylabel("Angle (theta)"); 
	plt.show(); 



	bTestCart = GM(); 
	bTestCart.addG(Gaussian([1,2],[[1,0],[0,1]],.25)); 
	bTestCart.addG(Gaussian([-3,1],[[3,0],[0,1]],.25)); 
	bTestCart.addG(Gaussian([1,-4],[[1,0],[0,2]],.25)); 
	bTestCart.addG(Gaussian([-3,-3],[[2,1.2],[1.2,2]],.25)); 
	bTestPol = transformCartToPol(bTestCart,[0,0]); 



	[xTestCart,yTestCart,cTestCart] = bTestCart.plot2D(low=[-10,-10],high=[10,10],vis=False); 
	[xTestPol,yTestPol,cTestPol] = bTestPol.plot2D(low=[0,-3.14],high=[10,3.14],vis=False); 

	fig,axarr = plt.subplots(1,2); 
	axarr[0].contourf(xTestCart,yTestCart,cTestCart); 
	axarr[1].contourf(xTestPol,yTestPol,cTestPol); 
	axarr[0].set_title("Cartesian Gaussians");
	axarr[1].set_title("Polar Gaussians"); 
	axarr[0].set_xlabel("X"); 
	axarr[0].set_ylabel("Y"); 
	axarr[1].set_xlabel("Radius (r)"); 
	axarr[1].set_ylabel("Angle (theta)"); 
	plt.show(); 

	'''
	#particle filter
	xcur = []; 
	numParticles = 1000; 

	initPart = []; 
	for i in range(0,int(np.sqrt(numParticles))):
		for j in range(0,int(np.sqrt(numParticles))):
			initPart.append([np.random.random()*20-10,np.random.random()*20-10]); 
	# for i in range(-10,10):
	# 	for j in range(-10,10):
	# 		initPart.append([i,j]); 

	fig,axarr = plt.subplots(2); 
	for i in range(0,len(initPart)):
		axarr[0].scatter(initPart[i][0],initPart[i][1],c='b'); 

	axarr[0].set_xlim([-10,10]); 
	axarr[0].set_ylim([-10,10]); 
	axarr[0].set_title('1000 Particles Randomly Scattered'); 

	xnew = particleFilter(initPart,observation,pz,obsOffset);
	#for i in range(0,10):
		#xnew = particleFilter(xnew,observation,pz,obsOffset);



	for i in range(0,len(xnew)):
		axarr[1].scatter(xnew[i][0],xnew[i][1],c='b')
	axarr[1].set_xlim([-10,10]); 
	axarr[1].set_ylim([-10,10]); 
	axarr[1].set_title('1000 Particles after a Particle Filter Update'); 
	plt.show(); 


	priorXs = []; 
	priorYs = []; 
	postXs = []; 
	postYs = []; 



	for i in initPart:
		priorXs.append(i[0]); 
		priorYs.append(i[1]);
	for i in range(0,len(xnew)): 
		postXs.append(xnew[i][0]); 
		postYs.append(xnew[i][1]); 


	xedgespri = [i for i in range(-10,10)]; 
	yedgespri = [i for i in range(-10,10)]; 

	xedgespost = [i for i in range(-10,10)]; 
	yedgespost = [i for i in range(-10,10)]; 

	Hpri,xedgespri,yedgespri = np.histogram2d(priorXs,priorYs,bins=(xedgespri,yedgespri)); 
	Hpost,xedgespost,yedgespost = np.histogram2d(postXs,postYs,bins=(xedgespost,yedgespost)); 
	

	fig,axarr = plt.subplots(1,2); 
	Xpri,Ypri = np.meshgrid(xedgespri,yedgespri); 
	axarr[0].pcolormesh(Xpri,Ypri,Hpri); 
	Xpost,Ypost = np.meshgrid(xedgespost,yedgespost); 
	axarr[1].pcolormesh(Xpost,Ypost,Hpost);
	axarr[0].set_title("Prior Histogram"); 
	axarr[1].set_title("Posterior Histogram"); 
	plt.suptitle("Particle Filter Update with Observation: " + str(observation)); 
	plt.show(); 
	'''

def transformCartToPol(bcart,offset = [0,0]):
	bpol = GM(); 
		
	for g in bcart:
		m = g.mean;
		m1 = [0,0]; 
		m1[0] = m[0] - offset[0]; 
		m1[1] = m[1] - offset[1]; 
		mPrime = [np.sqrt(m1[0]**2+m1[1]**2),np.arctan2(m1[1],m1[0])];  

		
		if(m1[0]**2+m1[1]**2 == 0):
			m1[0] = 0.0001; 
			m1[1] = 0.0001; 
		

		J11 = m1[0]/np.sqrt(m1[0]**2+m1[1]**2); 
		J12 = m1[1]/np.sqrt(m1[0]**2+m1[1]**2); 
		J21 = -m1[1]/(m1[0]**2+m1[1]**2); 
		J22 = m1[0]/(m1[0]**2+m1[1]**2); 

		JCarPol = np.matrix([[J11,J12],[J21,J22]]); 

		var = np.matrix(g.var);
		varPrime = (JCarPol*var*JCarPol.T).tolist();

		bpol.addG(Gaussian(mPrime,varPrime,g.weight));

	return bpol;  


def transformPolToCart(bpol,offset=[0,0]):
	bcart = GM(); 
	
	for g in bpol:
		m = g.mean; 
		mPrime = [m[0]*math.cos(m[1])+offset[0],m[0]*math.sin(m[1])+offset[1]]; 
		J11 = math.cos(m[1]); 
		J12 = -m[0]*math.sin(m[1]); 
		J21 = math.sin(m[1]); 
		J22 = m[0]*math.cos(m[1]); 

		JPolCar = np.matrix([[J11,J12],[J21,J22]]); 
		var = np.matrix(g.var);
		varPrime = (JPolCar*var*JPolCar.T).tolist();

		bcart.addG(Gaussian(mPrime,varPrime,g.weight)); 
	return bcart; 

def plot2DPolar(pz,low = [0,0],high = [10,10],offset = [0,0],delta = 0.1,vis = True):
	x, y = np.mgrid[low[0]:high[0]:delta, low[1]:high[1]:delta]
	pos = np.dstack((x, y))  
	resx = int((high[0]-low[0])//delta)+1;
	resy = int((high[1]-low[1])//delta)+1; 

	model = [[[0 for i in range(0,resy)] for j in range(0,resx)] for k in range(0,len(pz.weights))];
	

	for m in range(0,len(pz.weights)):
		for i in range(0,resx):
			xx = (i*(high[0]-low[0])/resx + low[0])-offset[0];
			for j in range(0,resy):
				yy = (j*(high[1]-low[1])/resy + low[1])-offset[1]; 
				rcord = math.sqrt(xx**2 + yy**2); 

				thetacord = np.arctan2(yy,xx); 
				dem = 0; 
				for k in range(0,len(pz.weights)):
					dem+=np.exp(pz.weights[k][0]*rcord + pz.weights[k][1]*thetacord + pz.bias[k]);
				model[m][i][j] = np.exp(pz.weights[m][0]*rcord + pz.weights[m][1]*thetacord + pz.bias[m])/dem;

	dom = [[0 for i in range(0,resy)] for j in range(0,resx)]; 
	for m in range(0,len(pz.weights)):
		for i in range(0,resx):
			for j in range(0,resy):
				dom[i][j] = np.argmax([model[h][i][j] for h in range(0,len(pz.weights))]);

	if(vis==True):
		plt.contourf(x,y,dom); 
	else:
		return [x,y,dom]; 


#Algorithm in Probabilistic Robotics by Thrun, page 98
def particleFilter(XprevIn,o,pz,offset=[0,0]):
	Xcur = []; 
	Xprev = deepcopy(XprevIn); 
	delA = [-1,1,0]; 
	delAVar = 0.5;
	allW = []; 
	allxcur = []; 
	for xprev in Xprev:
			
		xprev[0] = xprev[0]-offset[0]; 
		xprev[1] = xprev[1]-offset[1]; 

		xprevprime = [0,0]; 
		xprevprime[0] = np.sqrt(xprev[0]**2+xprev[1]**2); 
		xprevprime[1] = np.arctan2(xprev[1],xprev[0]); 

		w = pz.pointEval2D(o,xprevprime); 
 		xprev = [0,0]; 
 		xprev[0] = xprevprime[0]*math.cos(xprevprime[1])+offset[0]; 
 		xprev[1] = xprevprime[0]*math.sin(xprevprime[1])+offset[1]; 

		allW.append(w);
		allxcur.append(xprev); 

	#normalize weights for kicks
	suma = 0; 
	for w in allW:
		suma+=w; 
	for i in range(0,len(allW)):
		allW[i] = allW[i]/suma; 

	allIndexes = [i for i in range(0,len(allxcur))]; 
	for m in range(0,len(Xprev)):

		c = np.random.choice(allIndexes,p=allW);
		c = allxcur[c];  
		Xcur.append(deepcopy(c)); 
	return Xcur; 


def testInvertedSoftmaxModels():

	b = GM(); 
	b.addG(Gaussian([2,2],[[1,0],[0,1]],1)); 
	b.addG(Gaussian([4,2],[[1,0],[0,1]],1)); 
	b.addG(Gaussian([2,4],[[1,0],[0,1]],1)); 
	b.addG(Gaussian([3,3],[[1,0],[0,1]],1)); 
	b.normalizeWeights(); 

	b.plot2D(); 


	pz = Softmax(); 
	pz.buildOrientedRecModel([2,2],0,1,1,5); 
	#pz.plot2D(); 

	startTime = time.clock(); 
	b2 = GM(); 
	for i in range(1,5):
		b2.addGM(pz.runVBND(b,i)); 
	print(time.clock()-startTime); 
	b2.plot2D(); 

	startTime = time.clock(); 
	b3 = GM(); 
	b3.addGM(b); 
	tmpB = pz.runVBND(b,0); 
	tmpB.normalizeWeights(); 
	tmpB.scalerMultiply(-1); 
	

	b3.addGM(tmpB);

	
	tmpBWeights = b3.getWeights(); 
	mi = min(b3.getWeights()); 
	#print(mi); 
	for g in b3.Gs:
		g.weight = g.weight-mi; 
	

	b3.normalizeWeights(); 
	print(time.clock()-startTime); 
	#b3.display();  
	b3.plot2D(); 


if __name__ == "__main__":
	#buildRectangleModel();
	#buildGeneralModel(); 
	#buildPointsModel();
	#stretch1DModel();  
	#slice3DModel(); 
	#buildRecFromCentroidOrient(); 
	#buildTriView(); 
	#make3DSoftmaxAnimation(); 
	#buildRadialSoftmaxModels(); 
	testInvertedSoftmaxModels(); 








