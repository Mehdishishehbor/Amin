#!/usr/bin/env python
import torch
from lmgp_pytorch.models import LMGP
from lmgp_pytorch.test_functions.physical import borehole_mixed_variables
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
from lmgp_pytorch.utils import set_seed


random_state = 12345
set_seed(random_state)
qual_index = {0:5, 6:5}

X, y = borehole_mixed_variables(n = 1000, qual_ind_val= qual_index, random_state = random_state)
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.9, qual_index = qual_index)

model = LMGP(Xtrain, ytrain, qual_ind_lev=qual_index)
model.fit(n_jobs=1)
model.score(Xtest, ytest, plot_MSE=True)

model.get_params()

model.log_marginal_likelihood(X = Xtest[0,:], y = ytest[0])

_ = model.sample_y()

_ = model.sample_y(size = 1000, plot=True)

_ = model.visualize_latent()

print(model.get_latent_space())



################################# prediction on test set ########################################

