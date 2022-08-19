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
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
import pickle
import time
from lmgp_pytorch.utils import set_seed



###############Parameters########################
add_prior_flag = False
num_minimize_init = 16
level_set = None
quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

#################################################
set_seed(1)
###################################################
data = pd.read_csv('./datasets/data_M.csv')
X = data.iloc[:,3:-3].to_numpy()
y = data.iloc[:,-3].to_numpy().astype(np.float64).reshape(-1,)
y = torch.from_numpy(y)
############################## train test split ####################################
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.2, shuffle=True)
############################### Model ##############################################
model = LMGP(Xtrain, ytrain)
############################### Fit Model ##########################################
_ = fit_model_scipy(model, num_restarts= 16)
############################### Score ##############################################
model.score(Xtest, ytest, plot_MSE=True)
model.show()
LMGP.show()

