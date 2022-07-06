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



###############Parameters########################
add_prior_flag = False
num_minimize_init = 16
level_set = None
quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

#################################################
set_seed(1)
###################################################
data = pd.read_csv('./datasets/data_M.csv')
X = data.iloc[:,:3].to_numpy()
y = data.iloc[:,-3].to_numpy().astype(np.float64).reshape(-1,)
y = torch.from_numpy(y)
qual_index = [0,1,2]
quant_index = []

if level_set is None:
    level_set = [len(set(X[...,jj])) for jj in qual_index]

########### Make it standard ######################
if len(qual_index) > 0:
    X = setlevels(X, qual_index)
    X = standard(X, quant_index)

qual_index_lev = {i:j for i,j in zip(qual_index, level_set)}

if len(quant_index) == 0:
    X = X[...,qual_index]
########### Select few random choice ################
model = LMGP(
    train_x=X,
    train_y=y,
    qual_ind_lev= qual_index_lev,
).double()

# optimize noise successively
_= fit_model_scipy(model)

#_ = model.score(X, y, plot_MSE=True)

_ = model.visualize_latent()
LMGP.show()


