from __future__ import division
'''
************************************************************************************************************************************************************
File: POMDPTranslator.py
************************************************************************************************************************************************************
'''

__author__ = "Luke Burks"
__copyright__ = "Copyright 2017, Cohrint"
__credits__ = ["Luke Burks","LT","Ian Loefgren"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"


from gaussianMixtures import Gaussian, GM 
from map_maker import Map
from copy import deepcopy
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches
import math
import matplotlib.tri as tri;
import os
import matplotlib.pyplot as plt 

class POMDPTranslator(object):


	def __init__(self):
		self.map2 = Map('map2.yaml');
		self.bounds = [-9.6, -3.6, 4, 3.6]
		self.delta = 0.1; 

	def getNextPose(self,belief,obs,copPoses=None):
		pass;

	def beliefUpdate(self, belief, responses = None,copPoses = None):
		#1. partition means into separate GMs, 1 for each room
		allBels = []; 
		weightSums = []; 
		for room in self.map2.rooms:
			tmp = GM(); 
			tmpw = 0; 
			for g in belief:
				m = [g.mean[2],g.mean[3]];
				if(m[0] <= self.map2.rooms[room]['upper_r'][0] and m[0] >= self.map2.rooms[room]['lower_l'][0] and m[1] <= self.map2.rooms[room]['upper_r'][1] and m[1] >= self.map2.rooms[room]['lower_l'][1]):
					tmp.addG(deepcopy(g)); 
					tmpw+=g.weight; 
			tmp.normalizeWeights(); 
			allBels.append(tmp);
			weightSums.append(tmpw); 

		#2. use queued observations to update appropriate rooms GM
		

		#3. recombine beliefs
		newBelief = GM(); 
		for g in allBels:
			g.scalerMultiply(weightSums[allBels.index(g)]); 
			newBelief.addGM(g); 
		newBelief.normalizeWeights(); 
		
		#4. fix cops position in belief
		for g in newBelief:
			g.mean = [copPoses[0][0],copPoses[0][1],g.mean[2],g.mean[3]]; 

		#5. add uncertainty for robber position
		for g in newBelief:
			g.var[2][2] += 0.25; 
			g.var[3][3] += 0.25; 

		return newBelief; 

	def makeBeliefMap(self,belief,copPose = [0,0,0]):

		print("MAKING NEW BELIEF MAP!")
		fig = Figure()
		canvas = FigureCanvas(fig)
		ax = fig.add_subplot(111)

		x_space,y_space = np.mgrid[self.bounds[0]:self.bounds[2]:self.delta,self.bounds[1]:self.bounds[3]:self.delta];
		bcut = self.cutGMTo2D(belief,dims=[2,3]); 
		bel = bcut.discretize2D(low = [self.bounds[0],self.bounds[1]],high=[self.bounds[2],self.bounds[3]],delta=self.delta); 
		ax.contourf(x_space,y_space,bel,cmap="viridis");
		m = self.map2;
		for obj in m.objects:
		    cent = m.objects[obj].centroid;
		    x = m.objects[obj].length;
		    y = m.objects[obj].width;
		    theta = m.objects[obj].orient;
		    col = m.objects[obj].color
		    if(m.objects[obj].shape == 'oval'):
		        tmp = patches.Ellipse((cent[0] - x/2,cent[1]-y/2),width = x, height=y,angle=theta,fc=col,ec='black');
		    else:
		        tmp = patches.Rectangle(self.findLLCorner(m.objects[obj]),width = x, height=y,angle=theta,fc=col,ec='black');
		    ax.add_patch(tmp)

		bearing = -90;
		l = 1;
		triang=tri.Triangulation([copPose[0],copPose[0]+l*math.cos(2*-0.261799+math.radians(copPose[2]+(bearing)+90)),copPose[0]+l*math.cos(2*0.261799+math.radians(copPose[2]+(bearing)+90))],[copPose[1],copPose[1]+l*math.sin(2*-0.261799+math.radians(copPose[2]+(bearing)+90)),copPose[1]+l*math.sin(2*0.261799+math.radians(copPose[2]+(bearing)+90))])

		levels = [i/250 + 1 for i in range(0,250)]

		tpl = ax.tricontourf(triang,[2,1,1],cmap="inferno",alpha=0.5,levels=levels);

		cop = patches.Circle((copPose[0],copPose[1]),radius=0.2,fc = 'white',ec='black');
		ax.add_patch(cop)

		ax.axis('scaled')
		print('about to save plot')
		canvas.print_figure(os.path.abspath(os.path.dirname(__file__) + '../tmp/tmpBelief.png'),bbox_inches='tight',pad_inches=0)
		#canvas.print_figure('tmpBelief.png',bbox_inches='tight',pad_inches=0)
		

	def cutGMTo2D(self,mix,dims = [2,3]):
		newer = GM(); 
		for g in mix:
			newer.addG(Gaussian([g.mean[dims[0]],g.mean[dims[1]]],[[g.var[dims[0]][dims[0]],g.var[dims[0]][dims[1]]],[g.var[dims[1]][dims[0]],g.var[dims[1]][dims[1]]]],g.weight)); 
		return newer; 

	def findLLCorner(self, obj):
		""" Returns a 2x1 tuple of x and y coordinate of lower left corner """
		length = obj.length
		width = obj.width

		theta1 = obj.orient*math.pi/180;
		h = math.sqrt((width/2)*(width/2) + (length/2)*(length/2));
		theta2 = math.asin((width/2)/h);

		s1 = h*math.sin(theta1+theta2);
		s2 = h*math.cos(theta1+theta2);

		return (obj.centroid[0]-s2, obj.centroid[1]-s1)


def testGetNextPose():
	pass;

def testBeliefUpdate():
	translator = POMDPTranslator(); 

	b = GM(); 
	b.addG(Gaussian([3,2,1,0],np.identity(4).tolist(),1)); 
	bcut = cutGMTo2D(b,dims=[0,1]); 
	bcut.plot2D(low=[0,0],high=[10,5]); 
	b = translator.beliefUpdate(b,2,[[8,5]]); 
	bcut = cutGMTo2D(b,dims=[0,1]); 
	bcut.plot2D(low=[0,0],high=[10,5]); 

def testMakeMap():
	translator = POMDPTranslator(); 
	b = GM(); 
	b.addG(Gaussian([3,2,1,0],np.identity(4).tolist(),1)); 
	
	translator.makeBeliefMap(b,[0,0,0]); 





if __name__ == '__main__':
    #testGetNextPose();
    #testBeliefUpdate();
    testMakeMap();