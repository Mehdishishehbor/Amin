import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from LMGP_KNN_percent_test import LMGP_KNN_percent_test

rands_seeds = 175
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['5percent', '10percent', '20percent','30percent']
# out_mse_h = {'5percent':[], '10percent':[], '20percent':[],'30percent':[]}

layers = ['missing 5percent', 'missing 10percent', 'missing 20percent','missing 30percent','missing 50percent','missing 70percent','missing 80percent']
out_mse_h = {'missing 5percent':[], 'missing 10percent':[], 'missing 20percent':[],'missing 30percent':[],'missing 50percent':[],'missing 70percent':[],'missing 80percent':[]}

####################################################################################################################################

####################################################################################################################################
out_mse_h_L =['missing 5percent', 'missing 10percent', 'missing 20percent','missing 30percent','missing 50percent','missing 70percent','missing 80percent']


Percent_of_missing_data=[5,10,20,30,50,70,80]
rep=20
for j in range(len(layers)):

    temp_mse_h = []
    
    for i in range(rep):

        #######################################
        MSE_values=LMGP_KNN_percent_test(randseed=rands_seeds * i,Percent_of_missing=Percent_of_missing_data[j])
#       ######################################

        temp_mse_h.append(MSE_values)
        

 
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h


DATA_LMGPKNN_100_training_points_borhole=out_mse_h

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_LMGPKNN_100_training_points_borhole.npy', 'wb') as f:
    np.save(f, DATA_LMGPKNN_100_training_points_borhole)

