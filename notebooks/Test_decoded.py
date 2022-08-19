#!/usr/bin/env python
from lmgp_pytorch.models import LMGP
from lmgp_pytorch.test_functions.physical import borehole_mixed_variables
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
from lmgp_pytorch.utils import set_seed
from lmgp_pytorch.optim import fit_model_scipy, noise_tune2
from lmgp_pytorch.visual import  plot_latenth_position
import pandas as pd
import numpy as np
import torch
from lmgp_pytorch.preprocessing.numericlevels import setlevels
import random

############################### Paramter of the model #########################
##__###
random_state = 1
set_seed(random_state)
qual_index = []#[74]#{0:5, 1:6, 3:5, 5:5}
lv_columns = [0, 5]
level_set=None
############################ Generate Data #########################################
#X, y = borehole_mixed_variables(n = 10000, qual_ind_val= qual_index, random_state = random_state)
############################## train test split ####################################
data = pd.read_csv('./KCPCombo5_data.csv')
y = torch.tensor(data.iloc[:,-2].to_numpy().reshape(-1,),dtype = torch.double)
X_NN = torch.tensor(data.iloc[:,2:-2].to_numpy(),dtype = torch.double)
#X_NN=torch.cat((torch.tensor(data.iloc[:,2:-2].to_numpy(),dtype = torch.double), torch.tensor(data.iloc[:,-1].to_numpy().reshape(-1,1),dtype = torch.double)), 1)

X_numerical=torch.cat((X_NN[:,0:40], X_NN[:,76:]),1).clone()

meanx = X_numerical.mean(dim=-2, keepdim=True)
stdx = X_numerical.std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
X = (X_numerical- meanx) / stdx

#################################################################################################################   
X_Cavity=X_NN[:,40:60]
X_MachineID=X_NN[:,60:72]
X_Model=X_NN[:,72:76]
X_Cavity_original=torch.FloatTensor([np.where(r==1)[0][0] for r in X_Cavity])
X_MachineID_original=torch.FloatTensor([np.where(r==1)[0][0] for r in X_MachineID])
X_Model_original=torch.FloatTensor([np.where(r==1)[0][0] for r in X_Model])
#################################################################################################################

X_decoded=torch.cat((X, X_Cavity_original.reshape(-1,1),X_MachineID_original.reshape(-1,1),X_Model_original.reshape(-1,1)),1).clone()

#######################################################################################

# qual_index=[74]
# if level_set is None:
#     level_set = [len(set(X[...,jj].numpy())) for jj in qual_index]
    
# if len(qual_index) > 0:
#     X= setlevels(X.numpy(), qual_index)
#     #X = standard(X, quant_index)

# qual_index = {74:len(set(X[:,74].numpy()))}
# Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X_decoded, y, test_size = 0.1, 
#     qual_index_val= qual_index)

# #y = torch.from_numpy(y)

# #quant_index = list(range(0,7))

##################################################################################### Genarate missing data ###############################################################

# Percent_of_missing_data=0.1
# Number_of_missing_data=np.floor(np.array(ytrain.shape)*Percent_of_missing_data/100)
# random.seed(12)
# NAN_Lis=random.sample(range(0, int(np.array(ytrain.shape))), int(Number_of_missing_data))
# Target=Xtrain[:,74]
# X_Y_test_lable=Target[NAN_Lis]
# ii=int(torch.max(X[:74]))+1
# for k in NAN_Lis:
#     Xtrain[k,74]=ii
#     ii+=1
# level_set_final =[level_set[0]+len(NAN_Lis)]
# qual_index_lev_final = {i:j for i,j in zip(qual_index, level_set_final)}

############################### Model ##############################################
#model = LMGP(Xtrain, ytrain, qual_ind_lev=qual_index)

######################################################################
# qual_index = {42:20,43:12,44:4}
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.1)
model = LMGP(Xtrain,ytrain)
# model = LMGP(Xtrain, ytrain, qual_ind_lev=qual_index)
############################### Fit Model ##########################################
_ = fit_model_scipy(model,num_restarts=12)
#_ = noise_tune2(model)
####################################################################################
# positions=model.visualize_latent_position(model)
# # #print(f'positions= {positions}')
# # ##########################################################################################################
# from sklearn.neighbors import KNeighborsClassifier
# X_TRAIN_KNN=np.array(positions[0:level_set[0],:].data)#positions[0:level_set[0],:]
# Y_TRAIN_KNN=range(0, level_set[0])
# neigh_new = KNeighborsClassifier(n_neighbors=1)
# neigh_new.fit(X_TRAIN_KNN, Y_TRAIN_KNN)
# predict_based_on_latent_map=neigh_new.predict(np.array(positions[level_set[0]:level_set[0]+int(Number_of_missing_data),:].data))
# accuracy_final= np.sum(np.array(X_Y_test_lable) == predict_based_on_latent_map)/len(X_Y_test_lable)
# print(f'accuracy rate new method ={accuracy_final}')
#############################################################################################################
# LMGP.show()
############################### Score ##############################################
# model.score(Xtest, ytest, plot_MSE=True)
model.score(Xtest, ytest, plot_MSE=True)
############################### latent space ########################################
# _ = model.visualize_latent()
model.show()