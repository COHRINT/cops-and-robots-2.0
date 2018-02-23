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

from pdb import set_trace

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
import re
from softmaxModels import Softmax
from scipy.stats import multivariate_normal as mvn
from obs_q_map import gen_questions


class POMDPTranslator(object):


	def __init__(self, map_yaml='mapA.yaml'):
		self.map_ = Map(map_yaml);
		self.bounds = self.map_.bounds
		self.delta = 0.1;
		self.upperPolicy = np.load(os.path.dirname(__file__) + '/../policies/upperPolicy1.npy');

		roomNames = ['Hallway','Billiard','Study','Library','Dining','Kitchen'];
		self.lowerPolicys = [];
		for i in range(0,len(roomNames)):
			self.lowerPolicys.append(np.load(os.path.dirname(__file__) + '/../policies/' + roomNames[i]+'AlphasFull.npy'));

		self.question_list = gen_questions(map_yaml)
		print self.question_list
                # set_trace()

		# TODO: Find the correct mappings to the rooms

		hall = 5
		bill = 4
		libb = 2
		dinn = 3
		kitt = 1
		stud = 0
                
		self.rooms_map = {hall:'hallway',bill:'billiard room',stud:'study',libb:'library',dinn:'dining room',kitt:'kitchen'}
		self.rooms_map_inv = {'hallway':hall,'billiard room':bill,'study':stud,'library':libb,'dining room':dinn,'kitchen':kitt}

	def getNextPose(self,belief,obs=None,copPoses=None):
		print('GETTING NEW POSE')
		#0. update belief
		newBel = self.beliefUpdate(belief,obs,copPoses);
		#1. partition means into separate GMs, 1 for each room
		allBels = [];
		weightSums = [];
		for room in self.map_.rooms:
			tmp = GM();
			tmpw = 0;
			for g in belief:
				m = [g.mean[2],g.mean[3]];
                                if(m[0] < self.map_.rooms[room]['max_x'] and m[0] > self.map_.rooms[room]['min_x'] and m[1] < self.map_.rooms[room]['max_y'] and m[1] > self.map_.rooms[room]['min_y']):
					tmp.addG(deepcopy(g));
					tmpw+=g.weight;
			tmp.normalizeWeights();
			allBels.append(tmp);
			weightSums.append(tmpw);
		#2. find action from upper level pomdp
		[room,questsHigh,weightsHigh] = self.getUpperAction(weightSums);
		# print(questsHigh);
		# print(weightsHigh);
		questHighConversion = [5,4,0,2,3,1]
		questHighNew = [];
		for i in range(0,len(questsHigh)):
			questHighNew.append(questHighConversion[int(questsHigh[i])])
		questsHigh = questHighNew;
		questHighNew = [];
		for i in range(0,len(questsHigh)):
			questHighNew.append([questsHigh[i],0]);
		questsHigh = questHighNew;

		#3. find position and questions from lower level pomdp for that room


		roomConversion = [5,4,0,2,3,1];
		room_conv = roomConversion[room];

		[movement,questsLow,weightsLow] = self.getLowerAction(belief,room_conv);

		# TODO: Fake Questions and goal pose
		# goal_pose = allBels[room].findMAPN();
		# goal_pose = [goal_pose[2],goal_pose[3]];



		pose = copPoses[-1];
		roomCount = 0;
		copRoom = 7;
		for room in self.map_.rooms:  
                        if(pose[0] < self.map_.rooms[room]['max_x'] and pose[0] > self.map_.rooms[room]['min_x'] and pose[1] < self.map_.rooms[room]['max_y'] and pose[1] > self.map_.rooms[room]['min_y']):
				copRoom = self.rooms_map_inv[room]
				break;
			roomCount+=1;

		if(copRoom == room_conv):
			displacement = [0,0,0];
			dela = 1.0
			if(movement ==0):
				displacement = [-dela,0,0];
			elif(movement == 1):
				displacement = [dela,0,0];
			elif(movement == 2):
				displacement = [0,dela,0];
			elif(movement==3):
				displacement = [0,-dela,0];
			goal_pose = np.array(copPoses[-1]) + np.array(displacement);
			goal_pose = goal_pose.tolist();
		else:
			xpose = (self.map_.rooms[self.rooms_map[room_conv]]['max_x'] + self.map_.rooms[self.rooms_map[room_conv]]['min_x'])/2;
			ypose = (self.map_.rooms[self.rooms_map[room_conv]]['max_y'] + self.map_.rooms[self.rooms_map[room_conv]]['min_y'])/2;
			goal_pose = [xpose,ypose,0];

		for i in range(0,len(questsLow)):
			questsLow[i] = [room_conv,questsLow[i]+1];


		#questsLow = [18,43,21,33,58];
		#weightsLow = [24,54,23,48,53];
		#questsLow = [];
		#weightsLow = [];


		suma = sum(weightsLow);

		for i in range(0,len(weightsLow)):
			weightsLow[i] = weightsLow[i]/suma;

		#4. weight questions accordingly
		h = [];
		for i in range(0,len(questsHigh)):
			h.append([weightsHigh[i],questsHigh[i]])
		for i in range(0,len(questsLow)):
			h.append([weightsLow[i],questsLow[i]])
		# print('H: {}'.format(h))
		h = sorted(h,key = self.getKey,reverse=True);
		questsFull = [];
		weightsFull = [];
		for i in h:
			questsFull.append(i[1]);
			weightsFull.append(i[0]);



		#5. use questioner function to publish questions
		#questions = self.getQuestionStrings(questsFull);
		questions = [];
		for i in range(0,len(questsFull)):
			print questsFull
			print i
			print questsFull[i][0]
			print questsFull[i][1]
			questions.append(self.question_list[questsFull[i][0]][questsFull[i][1]]);

                        

		#6. return new belief and goal pose
		return [newBel,goal_pose,[questions,questsFull]];



	def getQuestionStrings(self,questIds):

		strings = ['Is Roy in the Kitchen','Is Roy in the Dining Room','Is Roy in the Hallway','Is Roy in the Study','Is Roy in the Library','Is Roy in the Billiard Room'];
		questStrings = [];

		for i in range(0,len(questIds)):
			if(questIds[i] < 6):
				questStrings.append(strings[int(questIds[i])]);
		return questStrings;



	def getUpperAction(self,b):
		worstVal = 1000000000;
		bestVal = -100000;
		bestInd = 0;
		Gamma = self.upperPolicy;

		questList = [];

		for j in range(0,len(Gamma)):
			tmp = self.dotProduct(b,Gamma[j]);
			questList.append([tmp,Gamma[j][-1]]);
			if(tmp>bestVal):
				bestVal = tmp;
				bestInd = j;
			if(tmp < worstVal):
				worstVal=tmp;

		for i in range(0,len(questList)):
			questList[i][0] = questList[i][0]-worstVal;

		a = sorted(questList,key=self.getKey,reverse=True);
		b = [];
		weights = [];
		for i in a:
			if(i[1]%6 not in b):
				b.append(i[1]%6);
				weights.append(i[0]);

		mi = min(weights);
		suma = 0;
		for i in range(0,len(weights)):
			weights[i] = weights[i] - mi;
			suma+=weights[i];
		for i in range(0,len(weights)):
			weights[i] = weights[i]/suma;

		return [int(Gamma[bestInd][-1])//6,b,weights];

	def getKey(self,item):
		return item[0];

	def getLowerAction(self,b,room):
		#act = Gamma[np.argmax([self.continuousDot(j,b) for j in Gamma])].action;
		Gamma = self.lowerPolicys[room];
		valActs = [];
		for j in range(0,len(Gamma)):
			valActs.append([self.continuousDot(Gamma[j],b),Gamma[j].action]);
		a = sorted(valActs,key=self.getKey,reverse=True);
		quests = [];
		questWeights = [];

		for i in a:
			if(i[1][1] not in quests):
				quests.append(i[1][1]);
				questWeights.append(i[0]);
		mi = min(questWeights);
		for i in range(0,len(questWeights)):
			questWeights[i] = questWeights[i] -mi;

		suma = sum(questWeights);

		for i in range(0,len(questWeights)):
			questWeights[i] = questWeights[i]/suma;

		print('LOWER ACTION: {}'.format(a[0][1][0]))
		return [a[0][1][0],quests,questWeights];

	def dotProduct(self,a,b):
		suma = 0;
		for i in range(0,len(a)):
			suma+=a[i]*b[i];
		return suma;

	def continuousDot(self,a,b):
		suma = 0;

		if(isinstance(a,np.ndarray)):
			a = a.tolist();
			a = a[0];

		if(isinstance(a,list)):
			a = a[0];

		a.clean();
		b.clean();

		for k in range(0,a.size):
			for l in range(0,b.size):
				suma += a.Gs[k].weight*b.Gs[l].weight*mvn.pdf(b.Gs[l].mean,a.Gs[k].mean, np.matrix(a.Gs[k].var)+np.matrix(b.Gs[l].var));
		return suma;

	def beliefUpdate(self, belief, responses = None,copPoses = None):
		print('UPDATING BELIEF')
		#1. partition means into separate GMs, 1 for each room
		allBels = [];
		allBounds = [];
		copBounds = [];
		weightSums = [];
		for room in self.map_.rooms:
			tmp = GM();
			tmpw = 0;
                        
			allBounds.append([self.map_.rooms[room]['min_x'],self.map_.rooms[room]['min_y'],self.map_.rooms[room]['max_x'],self.map_.rooms[room]['max_y']]);
			for g in belief:
				m = [g.mean[2],g.mean[3]];
                                # if mean is inside the room
				if(m[0] < self.map_.rooms[room]['max_x'] and m[0] > self.map_.rooms[room]['min_x'] and m[1] < self.map_.rooms[room]['max_y'] and m[1] > self.map_.rooms[room]['min_y']):
					tmp.addG(deepcopy(g));
					tmpw+=g.weight;
                                      
			tmp.normalizeWeights();
			allBels.append(tmp);

			weightSums.append(tmpw);

		pose = copPoses[-1];
		roomCount = 0;
		copBounds = 0;
		for room in self.map_.rooms:
			if(pose[0] < self.map_.rooms[room]['max_x'] and pose[0] > self.map_.rooms[room]['min_x'] and pose[1] < self.map_.rooms[room]['max_y'] and pose[1] > self.map_.rooms[room]['min_y']):
				copBounds = self.rooms_map_inv[room]
			roomCount+=1;


		viewCone = Softmax();
		viewCone.buildTriView(pose,length=1,steepness=10);
		for i in range(0,len(viewCone.weights)):
			viewCone.weights[i] = [0,0,viewCone.weights[i][0],viewCone.weights[i][1]];
		
		#Only update room that cop is in with view cone update
		#Make sure to renormalize that room
		# newerBelief = GM();
		# for i in range(1,5):
		# 	tmpBel = viewCone.runVBND(allBels[copBounds],i);
		# 	newerBelief.addGM(tmpBel);
		# allBels[copBounds] = newerBelief;

		#Update all rooms
		for i in range(0,len(allBels)):
			newerBelief = GM(); 
			for j in range(1,5):
				newerBelief.addGM(viewCone.runVBND(allBels[i],j)); 
			allBels[i]=newerBelief;

		for i in range(0,len(allBels)):
			allBels[i].normalizeWeights();
		#allBels[copBounds].normalizeWeights();

		print('allBels LENGTH: {}'.format(len(allBels)))

		#2. use queued observations to update appropriate rooms GM
		if(responses is not None):
			for res in responses:
				roomNum = res[0];
				mod = res[1];
				clas = res[2];
				sign = res[3];

				if(roomNum == 0):
					#apply to all
					for i in range(0,len(allBels)):
						if(sign==True):
							allBels[i] = mod.runVBND(allBels[i],0);
						else:
							tmp = GM();
							for j in range(1,mod.size):
								tmp.addGM(mod.runVBND(allBels[i],j));
							allBels[i] = tmp;

				# else:
				# 	print('ROOM NUM: {}'.format(roomNum))
				# 	#apply to roomNum-1;
				# 	if(sign == True):
				# 		allBels[roomNum-1] = mod.runVBND(allBels[roomNum-1],clas);
				# 	else:
				# 		tmp = GM();
				# 		for i in range(1,mod.size):
				# 			if(i!=clas):
				# 				tmp.addGM(mod.runVBND(allBels[roomNum-1],i));
				# 		allBels[roomNum-1] = tmp;
				else:
					print('ROOM NUM: {}'.format(roomNum))
					#apply to all rooms
					for i in range(0,len(allBels)):
						if(sign == True):
							allBels[i] = mod.runVBND(allBels[i],clas);
						else:
							tmp = GM();
							for j in range(1,mod.size):
								if(j!=clas):
									tmp.addGM(mod.runVBND(allBels[i],j));
							allBels[i] = tmp;

		#2.5. Make sure all GMs stay within their rooms bounds:
		#Also condense each mixture
		for gm in allBels:
			for g in gm:
				g.mean[2] = max(g.mean[2],allBounds[allBels.index(gm)][0]-0.01);
				g.mean[2] = min(g.mean[2],allBounds[allBels.index(gm)][2]+0.01);
				g.mean[3] = max(g.mean[3],allBounds[allBels.index(gm)][1]-0.01);
				g.mean[3] = min(g.mean[3],allBounds[allBels.index(gm)][3]+0.01);

		for i in range(0,len(allBels)):
                        allBels[i].condense(10);
#			allBels[i] = allBels[i].kmeansCondensationN(6)


		#3. recombine beliefs
		newBelief = GM();
		for g in allBels:
			g.scalerMultiply(weightSums[allBels.index(g)]);
			newBelief.addGM(g);
		newBelief.normalizeWeights();

		#4. fix cops position in belief
		for g in newBelief:
			g.mean = [copPoses[0][0],copPoses[0][1],g.mean[2],g.mean[3]];
			g.var[0][0] = 0.1;
			g.var[0][1] = 0;
			g.var[1][0] = 0;
			g.var[1][1] = 0.1;

		#5. add uncertainty for robber position
		for g in newBelief:
			g.var[2][2] += 1
			g.var[3][3] += 1

		# newBelief.normalizeWeights();

		if copPoses is not None:
			pose = copPoses[len(copPoses)-1]
			print("MAP COP POSE TO PLOT: {}".format(pose))
			self.makeBeliefMap(newBelief,pose)



		return newBelief;

	def makeBeliefMap(self,belief,copPose = [0,0,0]):

		print("MAKING NEW BELIEF MAP!")
		fig = Figure()
		canvas = FigureCanvas(fig)
		ax = fig.add_subplot(111)


		x_space,y_space = np.mgrid[self.bounds[0]:self.bounds[2]:self.delta,self.bounds[1]:self.bounds[3]:self.delta];
		bcut = self.cutGMTo2D(belief,dims=[2,3]);
		bel = bcut.discretize2D(low = [self.bounds[0],self.bounds[1]],high=[self.bounds[2],self.bounds[3]],delta=self.delta);
                print(np.amax(bel))
		#ax.contourf(x_space,y_space,bel,cmap="viridis",vmin=0,vmax=0.0004);
                
		ax.contourf(x_space,y_space,bel,cmap="viridis");

		m = self.map_;
		for obj in m.objects:
		    cent = m.objects[obj].centroid;
		    x = m.objects[obj].x_len;
		    y = m.objects[obj].y_len;
		    theta = m.objects[obj].orient;
		    col = m.objects[obj].color
		    tmp = patches.Ellipse((cent[0] - x/2,cent[1]-y/2),width = x, height=y,angle=theta,fc=col,ec='black');
		  #   if(m.objects[obj].shape == 'oval'):
		  #       # tmp = patches.Ellipse((cent[0] - x/2,cent[1]-y/2),width = x, height=y,angle=theta,fc=col,ec='black');
		  #   else:
				# tmp = patches.Rectangle((cent[0]- x/2,cent[1]-y/2),width = x, height=y,angle=theta,fc=col,ec='black');
				# #tmp = patches.Rectangle(self.findLLCorner(m.objects[obj]),width = x, height=y,angle=theta,fc=col,ec='black');
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
		canvas.print_figure(os.path.abspath(os.path.dirname(__file__) + '/../tmp/tmpBelief.png'),bbox_inches='tight',pad_inches=0)
		#canvas.print_figure('tmpBelief.png',bbox_inches='tight',pad_inches=0)


	def cutGMTo2D(self,mix,dims = [2,3]):
		newer = GM();
		for g in mix:
			newer.addG(Gaussian([g.mean[dims[0]],g.mean[dims[1]]],[[g.var[dims[0]][dims[0]],g.var[dims[0]][dims[1]]],[g.var[dims[1]][dims[0]],g.var[dims[1]][dims[1]]]],g.weight));
		return newer;

	def findLLCorner(self, obj):
		""" Returns a 2x1 tuple of x and y coordinate of lower left corner """
                # LOL the x and y dimensions are defined to be length and width in map_maker...
                # Below they are used oppositely
		length = obj.length
		width = obj.width

		theta1 = obj.orient*math.pi/180;
		h = math.sqrt((length/2)*(length/2) + (width/2)*(width/2));
		theta2 = math.asin((length/2)/h);

		s1 = h*math.sin(theta1+theta2);
		s2 = h*math.cos(theta1+theta2);

		return (obj.centroid[0]-s2, obj.centroid[1]-s1)

	def obs2models(self,obs):
		"""Map received observation to the appropriate softmax model and class.
		Observation may be a str type with a pushed observation or a list with
		question and answer.
		"""
		print(obs)
		sign = None
		model = None
		model_name = None
		room_num = None
		class_idx = None
		# check if observation is statement (str) or question (list)
		if type(obs) is str:
			# obs = obs.split()
			if 'not' in obs:
				sign = False
			else:
				sign = True
		else:
			sign = obs[1]
			obs = obs[0]

		# find map object mentioned in statement
		for obj in self.map_.objects:
			if re.search(obj,obs.lower()):
				model = self.map_.objects[obj].softmax
				model_name = self.map_.objects[obj].name
				for i, room in enumerate(self.map_.rooms):
					if obj in self.map_.rooms[room]['objects']: # potential for matching issues if obj is 'the <obj>', as only '<obj>' will be found in room['objects']
						room_num = i+1
						print self.map_.rooms[room]['objects']
						print room_num
				break
		# if no model is found, try looking for room mentioned in observation
		if model is None:
			for room in self.map_.rooms:
				if re.search(room,obs.lower()):
					model = self.map_.rooms[room]['softmax']
					room_num = 0
					break

		# find softmax class index
		if re.search('inside',obs.lower()) or re.search('in',obs.lower()):
			class_idx = 0
		if re.search('front',obs.lower()):
			class_idx = 1
		elif re.search('right',obs.lower()):
			class_idx = 2
		elif re.search('behind',obs.lower()):
			class_idx = 3
		elif re.search('left',obs.lower()):
			class_idx = 4
		# elif 'near' in obs:
		# 	class_idx = 5

		print(room_num,model,model_name,class_idx,sign)
		return room_num, model, class_idx, sign


def testGetNextPose():
	translator = POMDPTranslator();
	b = GM();
	b.addG(Gaussian([3,2,-2,2],np.identity(4).tolist(),1));
	b.addG(Gaussian([3,2,-8,-2],np.identity(4).tolist(),1));
	b.addG(Gaussian([3,2,-4,-2],np.identity(4).tolist(),1));
	b.addG(Gaussian([0,0,2,2],(np.identity(4)*6).tolist(),1));

	b.normalizeWeights();
	#for i in range(-8,3):
		#for j in range(-1,2):
			#b.addG(Gaussian([3,2,i,j],np.identity(4).tolist(),1));
	translator.cutGMTo2D(b,dims=[2,3]).plot2D(low=[-9.6,-3.6],high=[4,3.6]);
	[bnew,goal_pose,qs] = translator.getNextPose(b,None,[[0,0,-15.3]]);
	bnew = translator.cutGMTo2D(bnew,dims=[2,3]);
	bnew.plot2D(low=[-9.6,-3.6],high=[4,3.6]);
	#print(qs)
	'''
	b2 = GM();
	b2.addG(Gaussian([-8,2,-8,-2],np.identity(4).tolist(),1));
	translator.cutGMTo2D(b2,dims=[2,3]).plot2D(low=[-9.6,-3.6],high=[4,3.6]);
	[bnew,goal_pose,qs] = translator.getNextPose(b2,None,[[1,2,15.3]]);
	print(qs);

	bnew = translator.cutGMTo2D(bnew,dims=[2,3]);
	bnew.plot2D(low=[-9.6,-3.6],high=[4,3.6]);
	'''

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
   	testGetNextPose();
    #testBeliefUpdate();
    #testMakeMap();
