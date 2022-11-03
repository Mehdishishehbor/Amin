#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
from lmgp_pytorch.models.lmgp import LMGP
from lmgp_pytorch.optim.mll_scipy import fit_model_scipy
from lmgp_pytorch.bayesian_optimizations.acquisition import EI_fun  
import pandas as pd
import random
from lmgp_pytorch.preprocessing.numericlevels import setlevels
from lmgp_pytorch.preprocessing import standard
import pickle
import time
from lmgp_pytorch.utils import set_seed

from lmgp_pytorch.preprocessing import train_test_split_normalizeX

###############Parameters########################
add_prior_flag = False
num_minimize_init = 16
level_set = None
quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

#################################################
set_seed(1)
###################################################
data = pd.read_csv('./datasets/sample.csv')
X = data.iloc[:,:3].to_numpy()
y = data.iloc[:,-1].to_numpy().astype(np.float64).reshape(-1,)
y = torch.from_numpy(y)
qual_index = [0,1,2]
quant_index = []

if level_set is None:
    level_set = [len(set(X[...,jj])) for jj in qual_index]

########### Make it standard ######################
if len(qual_index) > 0:
    X, labels = setlevels(X, qual_index, return_label=True)
    X = standard(X, quant_index)

qual_index_lev = {i:j for i,j in zip(qual_index, level_set)}

if len(quant_index) == 0:
    X = X[...,qual_index]

############################## train test split ####################################
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.7, 
    qual_index_val= qual_index_lev)
####################################################################################

model = LMGP(
    train_x=Xtrain,
    train_y=ytrain,
    qual_ind_lev= qual_index_lev,
).double()

# optimize noise successively
_= fit_model_scipy(model)

_ = model.score(Xtest, ytest, plot_MSE=True)

_ = model.visualize_latent(labels = labels)
LMGP.show()


