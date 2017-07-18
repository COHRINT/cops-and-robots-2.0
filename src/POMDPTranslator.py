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


class POMDPTranslator(object):


	def __init__(self):
		self.map2 = Map('map2.yaml'); 

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
					tmp2+=g.weight; 
			tmp.normalizeWeights(); 
			allBels.append(tmp);
			weightSums.append(tmpw); 

		#2. use queued observations to update appropriate rooms GM
		

		#3. recombine beliefs
		newBelief = GM(); 
		for g in allBels:
			g.scalarMultiply(weightSums[allBels.index(g)]); 
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
		pass; 


def testGetNextPose():
	pass;

def testBeliefUpdate():
	translator = POMDPTranslator(); 

	b = GM(); 
	b.addG(Gaussian([3,2,1,0],np.identity(4).tolist(),1)); 
	b = translator.beliefUpdate(b,2,[[8,5]]); 
	b.display(); 

def testMakeMap():
	pass; 

if __name__ == '__main__':
    #testGetNextPose();
    testBeliefUpdate();
    #testMakeMap();