import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from Multi_noise_for_statestics import statistics_test

rands_seeds = 11245
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}



#x=np.load('./DATA_LMGPKNN_5_10_20_30percent.npy', allow_pickle=True)
x=np.load('D:\Pytone\LMGP\lmgp-pmacs-NN_latent\lmgp-pmacs-Multiple_noise_estimate\lmgp-pmacs-Multiple_noise_estimate/DATA_LMGPKNN_300_training_points_borhole.npy', allow_pickle=True)





out = {'missing 5 %':x.tolist()['missing 5percent'], 'missing 10 %':x.tolist()['missing 10percent'],'missing 20 %':x.tolist()['missing 20percent'],'missing 30 %':x.tolist()['missing 30percent'],'missing 50 %':x.tolist()['missing 50percent'],'missing 70 %':x.tolist()['missing 70percent'],'missing 80 %':x.tolist()['missing 80percent']}
#out = {'missing from source h':x.tolist()['missing from source h'], 'missing from source l1':x.tolist()['missing from source l1'],'missing from source l2':x.tolist()['missing from source l2'],'missing from source l3':x.tolist()['missing from source l3']}
#out_mse_h_L =['missing 5percent', 'missing 10percent', 'missing 20percent','missing 30percent','missing 50percent','missing 70percent','missing 80percent']

plt.rcParams.update({'font.size': 19})
plt.figure(figsize=(8, 6))
plt.boxplot(out.values(), labels = out.keys())
plt.title('Total Number of data=300 ')
plt.ylabel('Accuracy')
#plt.xlabel('Mapping method ')
plt.show()

