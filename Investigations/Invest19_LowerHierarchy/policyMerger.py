
import numpy as np; 

#fileNames = ['HallwayAlphas_0','HallwayAlphas_1','HallwayAlphas_21','HallwayAlphas_22','HallwayAlphas_23','HallwayAlphas_31','HallwayAlphas_32','HallwayAlphas_33']; 
fileNames = ['DiningAlphas_0','DiningAlphas_1','DiningAlphas_2','DiningAlphas_3'];


allAlphas = []; 

for i in range(0,len(fileNames)):
	gamma = np.load(fileNames[i]+'.npy'); 
	allAlphas.append(gamma[0]);

f = open('DiningAlphasFull.npy','w');
np.save(f,allAlphas);