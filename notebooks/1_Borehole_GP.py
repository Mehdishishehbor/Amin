#!/usr/bin/env python
import torch
from lmgp_pytorch.models import LMGP
from lmgp_pytorch.test_functions.physical import Borehole
from lmgp_pytorch.preprocessing import train_test_split_normalizeX

X, y = Borehole(n = 10000, random_state= 12345)
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.99, random_state = 123456)

model = LMGP(Xtrain, ytrain)
model.fit()
model.score(Xtest, ytest, plot_MSE=True)

model.get_params()

model.log_marginal_likelihood(X = Xtest[0,:], y = ytest[0])

_ = model.sample_y()

_ = model.sample_y(size = 1000, plot=True)


# # This will print all the infor about noise, mll and etc
# model.print_stats()

# # plot both visualizations for latent map and mse
# model.visualize()

# model.get_paramters()

################################# prediction on test set ########################################

