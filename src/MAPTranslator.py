from __future__ import division


# DISCRETIZED MAP TRANSLATOR

from gaussianMixtures import Gaussian, GM

import random                   # testing getNextPose
import matplotlib.pyplot as plt # testing getNextPose
import sys
import numpy as np
import os;
import copy;
from map_maker import MAP

__author__ = "LT"
__copyright__ = "Copyright 2017, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"


class MAPTranslator(object):


    def __init__(self):
        self.delta = 0.1
        f = self.findFile('likelihoods.npy','../');
        self.likelihoods = np.load(f);
        self.bounds = [-9.6, -3.6, 4, 3.6]

    def findFile(self,name,path):
    	for root,dirs,files in os.walk(path):
    		if name in files:
    			return os.path.join(root,name);


    """
    Takes a belief and returns the goal
    pose (MAP coordinate list [x.y] in [m, m])
    (orientation is calculated in
    the Policy Translator Service)

    -Calls the findMAP function to find the [x,y] MAP
    coordinate.

    Parameters
    ----------
    belief : belief object

    Returns
    -----------
    goal_pose : [x, y] in [m,m]

    """

    # belief is a 2D grid because of the discretization
    def getNextPose(self, belief):

        #grid from plot 2D function
        max_coord = self._find_array2D_max(belief)
        goal_pose = self._grid_coord_to_world_coord(max_coord)

        # TODO call belief update method here
        # belief is a GM instance, find MAP coords [x,y]
        return [belief, belief.findMAPN()]

    """ Locates the max coord of the 2D array
        and returns its index"""
    def _find_array2D_max(self, array2D):
        max_num = 0
        max_index = [0,0]
        size = array2D.shape # tuple of numpy array size

        for i in range(size[0]):
            for j in range(size[1]):
                if (array2D[i,j] > max_num):
                    max_index = [i,j]
                    max_num = array2D[i,j]  # set new max num
        return max_index

    """ Inputs the max grid coord and returns the
        world map coord equivalent"""
    def _grid_coord_to_world_coord(self, coord, world_min_x_y=[0,0]):
        d = self.delta
        world_coord = [0,0]
        world_coord[0] = world_min_x_y[0] + d * coord[0] # world x coord
        world_coord[1] = world_min_x_y[1] + d * coord[1] # world y coord
        return world_coord

    def unFlatten(self,arr,shapes):
    	newArr = np.zeros((shapes[0],shapes[1]));
    	for i in range(0,shapes[0]):
    		for j in range(0,shapes[1]):
    			newArr[i][j] = arr[i*shapes[1]+j];
    	return newArr;

    def normalize(self,arr):
   		suma = sum(arr);
   		for i in range(0,len(arr)):
   			arr[i] = arr[i]/suma;
   		return arr;


    def beliefUpdate(self, belief, responses):

    	flatBelief = belief.flatten();
    	post = flatBelief;
    	for res in responses:
    	    if(res[1] == True):
    	        like = self.likelihoods[res[0]][1];
            else:
                like = 1-self.likelihoods[res[0]][1];
            print(self.likelihoods[res[0]][0],res[1])
            posterior = np.multiply(post,like);
            post = self.normalize(posterior);
        post = self.unFlatten(post,belief.shape);

       	return post;


    def makeBeliefMap(self, belief):
        pass

""" Creates a belief, call getNextPose to find the MAP
    verifies the coord returned is the actual MAP
        -Features: will plot the given MAP using plt.contourf """
def testGetNextPose(rndm=None):
    print "Testing MAP Translator!"
    MAP = MAPTranslator()


    if (rndm):
        random.seed()
        means = [[rdm(), rdm()], [rdm(),rdm()], [rdm(), rdm()],[rdm(), rdm()]]
        variances = [[1,0], [0,1]]
        weights = [1.5, 2, rdm(), rdm()]
        pos = [1,1]

    else:
        print("Determined Test")
        means = [[2,3],[2,4],[3,1],[3,4]]               # Also the MAP location
        variances = [[1,0], [0,1]]
        weights = [1.5, 2, 1.5, 0.7]
        pos = [1,1]

    # create the belief
    b = GM()
    b.addG(Gaussian(means[0], variances, weights[0]))
    b.addG(Gaussian(means[1], variances, weights[1]))
    b.addG(Gaussian(means[2], variances, weights[2]))
    b.addG(Gaussian(means[3], variances, weights[3]))
    b.normalizeWeights()            # all GaussianMixtures must be normalized

    min_x_y = [-6,-6]
    max_x_y = [8,8]
    d = 0.1
    grid = b.discretize2D(low=min_x_y, high=max_x_y, delta=d)

    max_pt = MAP._find_array2D_max(grid)

    # TODO why do these need to be reversed?
    # switch x and y
    x = max_pt[1]
    y = max_pt[0]
    max_pt = [x,y]

    MAP.delta = d
    print("In Luke's Discretized coords, the MAX is: "+ str(max_pt))
    print("In World Coords from "+ str(min_x_y) + " to " + str(max_x_y) + " with delta: "+str(MAP.delta)+ ". The MAP would be: " + str(MAP._grid_coord_to_world_coord(max_pt, min_x_y)))

    plt.contourf(grid, cmap='viridis')
    plt.pause(0.1)
    raw_input("Show MAP?")
    plt.scatter(max_pt[0], max_pt[1])
    plt.show()

def testBeliefUpdate():
    print "Testing Belief Update!"
    MAP = MAPTranslator();
    belief = GM();
    belief.addG(Gaussian([0,0],[[8,0],[0,8]],0.5));
    belief.addG(Gaussian([-8,-1],[[4,0],[0,4]],0.5));
    db = belief.discretize2D(low=[MAP.bounds[0],MAP.bounds[1]],high=[MAP.bounds[2],MAP.bounds[3]],delta=MAP.delta);

    responses = [[50,False],[3,True],[15,False]];

    post = MAP.beliefUpdate(db,responses);
    # print(sum(sum(post)));
    # print(db.shape);


    # like = MAP.unFlatten(MAP.likelihoods[questionNum][1],db.shape);
    #like = MAP.likelihoods[questionNum][1];
    # print(like.shape);
    # print(post.shape);

    x_space,y_space = np.mgrid[MAP.bounds[0]:MAP.bounds[2]:MAP.delta,MAP.bounds[1]:MAP.bounds[3]:MAP.delta];

    fig,axarr = plt.subplots(2);
    axarr[0].contourf(x_space,y_space ,db);
    axarr[0].set_title('Prior');
    # axarr[1].contourf(x_space,y_space ,like);
    # axarr[1].set_title('Likelihood');
    axarr[1].contourf(x_space,y_space ,post);
    axarr[1].set_title('Posterior');
    #plt.suptitle('Belief Update for question:' + str(MAP.likelihoods[questionNum][0]));
    plt.show();


def rdm():
    return random.randint(0, 5)

if __name__ == '__main__':
    #testGetNextPose();
    testBeliefUpdate();
    #testMakeMap();
    #m = Map('map1.yaml');
