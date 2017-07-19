
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

	'''
	#Trapezoidal Pyramid Specs
	numClasses = 7; 
	boundries = [[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]]; 
	B = np.matrix([0,0,-1,-1,-1,0,.5,-1,0,1,.5,-1,1,0,.5,-1,0,-1,.5,-1,0,0,1,-1]).T; 
	'''
	
	#Octohedron Specs
	numClasses = 9; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	B = np.matrix([-1,-1,0.5,-1,-1,1,0.5,-1,1,1,0.5,-1,1,-1,0.5,-1,-1,-1,-0.5,-1,-1,1,-0.5,-1,1,1,-0.5,-1,1,-1,-0.5,-1]).T; 
	

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
	B = np.matrix([0,0,-1,-1,-1,0,.5,-1,0,1,.5,-1,1,0,.5,-1,0,-1,.5,-1,0,0,1,-1]).T; 
	
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



	dataClear = np.zeros(shape=(101,101,101)); 

	shapeEdgesXClear = []; 
	shapeEdgesYClear = [];
	shapeEdgesZClear = []; 
	#-5 to 5 on all dims
	data = np.zeros(shape=(101,101,101)); 
	for i in range(0,101):
		for j in range(0,101):
			for k in range(0,101):
				dataClear[i][j][k] = pz.pointEvalND(0,[(i-50)/10,(j-50)/10,(k-50)/10]);
				if(dataClear[i][j][k] > 0.0001):
					shapeEdgesXClear.append((i-50)/10); 
					shapeEdgesYClear.append((j-50)/10); 
					shapeEdgesZClear.append((k-50)/10);   

	


	count = 0; 
	#Z Axis
	for slic in range(-10,10):
		shapeEdgesX = []; 
		shapeEdgesY = [];
		shapeEdgesZ = [];
		#-5 to 5 on all dims
		data = np.zeros(shape=(101,101)); 
		for i in range(0,101):
			for j in range(0,101):
					data[i][j] = pz.pointEvalND(0,[(i-50)/10,(j-50)/10,slic/5]);
					if(data[i][j] > 0.0001):
						shapeEdgesX.append((i-50)/10); 
						shapeEdgesY.append((j-50)/10); 
						shapeEdgesZ.append(slic/5);   
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
	ani.save('trapezoidalInteriorClass.gif',fps=3,writer='animation.writer')



if __name__ == "__main__":
	#buildRectangleModel();
	#buildGeneralModel(); 
	#buildPointsModel();
	#stretch1DModel();  
	#slice3DModel(); 
	#buildRecFromCentroidOrient(); 
	#buildTriView(); 
	make3DSoftmaxAnimation(); 


