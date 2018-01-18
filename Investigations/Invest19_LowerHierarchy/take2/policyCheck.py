import numpy as np
import matplotlib.pyplot as plt
from gaussianMixtures import GM,Gaussian

def cutGMTo2D(mix,dims = [2,3]):
	newer = GM();
	for g in mix:
		newer.addG(Gaussian([g.mean[dims[0]],g.mean[dims[1]]],[[g.var[dims[0]][dims[0]],g.var[dims[0]][dims[1]]],[g.var[dims[1]][dims[0]],g.var[dims[1]][dims[1]]]],g.weight));
	return newer;



def getGreedyAction(bel):

	MAP = bel.findMAPN();
	
	if(abs(MAP[0]-MAP[2])>abs(MAP[1]-MAP[3])):
		if(MAP[0]-MAP[2] > 0):
			act = 0; 
		else:
			act = 1; 
	else:
		if(MAP[1]-MAP[3] < 0):
			act = 2; 
		else:
			act = 3; 

	return act; 

gamma = np.load('../policies/D4QuestStudySoftmaxAlphas0.npy'); 

#Hallway: Good
#Billiard: Good, but some odd coordinates
#Dining: Good
#Kitchen: Good
#Library: Good
#Study: Good

# fig,axarr = plt.subplots(len(gamma));
#minim = -100
#maxim = 100 

moveLabels = ['Left','Right','Up','Down','Stay']; 

#levels = np.linspace(minim,maxim);  
for i in range(0,len(gamma)):
	#gamma[i].display(); 
	greed = getGreedyAction(gamma[i]); 
	if(greed != gamma[i].action[0]):
		print('Policy Movement: {}, Greedy Movement: {}'.format(moveLabels[gamma[i].action[0]],moveLabels[greed])); 
		print(gamma[i].findMAPN()); 
	# gammaPrime = cutGMTo2D(gamma[i],dims=[0,1]); 
	# x,y,c = gammaPrime.plot2D(low=[-9.5,-1],high=[4,1.4],vis=False); 
	# axarr[i].contourf(x,y,c); 
	# axarr[i].set_title(str(gamma[i].action)); 
# plt.show(); 




