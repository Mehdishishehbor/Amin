#!/usr/bin/env python
# coding: utf-8

from lmgp_pytorch.models import LMGP
from lmgp_pytorch.test_functions.multi_fidelity import multi_fidelity_wing
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
from lmgp_pytorch.utils import set_seed
from lmgp_pytorch.optim import fit_model_scipy

###############Parameters########################
random_state = 4
set_seed(random_state)
qual_index = {10:4}
num={'0': 5000, '1': 10000, '2': 10000, '3': 10000}
noise_std={'0': 0.5, '1': 1.0, '2': 1.5, '3': 2.0}
############################ Generate Data #########################################
X, y = multi_fidelity_wing(n = num, noise_std= noise_std, random_state = random_state)
############################## train test split ####################################
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.99, 
    qual_index_val= qual_index, stratify= X[...,list(qual_index.keys())])
############################### Model ##############################################
model_single_noise = LMGP(Xtrain, ytrain, qual_ind_lev=qual_index, multiple_noise= False)
model_multiple_noise = LMGP(Xtrain, ytrain, qual_ind_lev=qual_index, multiple_noise= True)
############################### Fit Model ##########################################
_ = fit_model_scipy(model_single_noise, num_restarts= 4)
_ = fit_model_scipy(model_multiple_noise, num_restarts= 4)
############################### Score ##############################################
model_single_noise.score(Xtest, ytest, plot_MSE=True, title = 'Single Noise')
model_multiple_noise.score(Xtest, ytest, plot_MSE=True, title = 'Multiple Noise')
############################### Latent Map ##############################################
_ = model_single_noise.visualize_latent(suptitle='Single Noise')
_ = model_multiple_noise.visualize_latent(suptitle='Multiple Noise')
LMGP.show()