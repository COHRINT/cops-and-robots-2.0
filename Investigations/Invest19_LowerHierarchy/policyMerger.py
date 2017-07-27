
import numpy as np; 

roomName = 'Kitchen'; 

#fileNames = ['HallwayAlphas_0','HallwayAlphas_1','HallwayAlphas_2','HallwayAlphas_3']; 
#fileNames = ['DiningAlphas_0','DiningAlphas_1','DiningAlphas_2','DiningAlphas_3'];
#fileNames = ['StudyAlphas_0','StudyAlphas_1','StudyAlphas_2','StudyAlphas_3']
fileNames = []; 
for i in range(0,4):
	fileNames.append(roomName+'Alphas_'+str(i)); 


allAlphas = []; 

for i in range(0,len(fileNames)):
	gamma = np.load(fileNames[i]+'.npy'); 
	allAlphas.append(gamma[0]);

f = open(roomName+'AlphasFull.npy','w');
np.save(f,allAlphas);