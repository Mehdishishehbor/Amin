import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from LMGP_KNN_TEST import LMGPKNN_test

rands_seeds = 175
# layers = ['High fidelity', 'Low fidelity 1', 'Low fidelity 2','Low fidelity 3']
out = {'High fidelity':[], 'Low fidelity 1':[], 'Low fidelity 2':[],'Low fidelity 3':[]}

# layers = ['5percent', '10percent', '20percent','30percent']
# out_mse_h = {'5percent':[], '10percent':[], '20percent':[],'30percent':[]}

layers = ['missing from source h', 'missing from source l1', 'missing from source l2','missing from source l3']
out_mse_h = {'missing from source h':[], 'missing from source l1':[], 'missing from source l2':[],'missing from source l3':[]}

####################################################################################################################################

####################################################################################################################################
out_mse_h_L =['missing from source h', 'missing from source l1', 'missing from source l2','missing from source l3']

Percent_of_missing_data=[5,10,20,30]
rep=20
for j in range(len(layers)):

    temp_mse_h = []
    
    for i in range(rep):

        #######################################
        MSE_values=LMGPKNN_test(randseed=rands_seeds * i,Percent_of_missing=Percent_of_missing_data[j])
#       ######################################

        temp_mse_h.append(MSE_values)
        

 
    out_mse_h[str(out_mse_h_L[j])] = temp_mse_h


DATA_LMGPKNN_5_10_20_30percent_only_One_source=out_mse_h

#DATA_one.to_csv("DATA_one",index=False)
with open('DATA_LMGPKNN_5_10_20_30percent_only_One_source.npy', 'wb') as f:
    np.save(f, DATA_LMGPKNN_5_10_20_30percent_only_One_source)

