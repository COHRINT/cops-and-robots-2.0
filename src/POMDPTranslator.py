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



class POMDPTranslator(object):


	def __init__(self):
		pass;

	def getNextPose(self,belief,obs,copPoses=None):
		pass;

	def beliefUpdate(self, belief, responses = None,copPoses = None):
		pass;

	def makeBeliefMap(self,belief,copPose = [0,0,0]):
		pass; 


def testGetNextPose():
	pass;

def testBeliefUpdate():
	pass;

def testMakeMap():
	pass; 

if __name__ == '__main__':
    #testGetNextPose();
    #testBeliefUpdate();
    testMakeMap();