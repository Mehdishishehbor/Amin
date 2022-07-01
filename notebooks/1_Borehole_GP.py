#!/usr/bin/env python
import torch
from lmgp_pytorch.models import LMGP
from lmgp_pytorch.test_functions.physical import borehole
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
from lmgp_pytorch.utils import set_seed
from lmgp_pytorch.optim import fit_model_scipy

random_state = 12345
set_seed(random_state)


X, y = borehole(n = 10000, random_state= 12345)
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.99)

model = LMGP(Xtrain, ytrain)
model.reset_parameters
_ = fit_model_scipy(model, num_restarts= 12)

model.score(Xtest, ytest, plot_MSE=True)

# model.get_params()

# model.log_marginal_likelihood(X = Xtest[0,:], y = ytest[0])

# _ = model.sample_y()

# _ = model.sample_y(size = 1000, plot=True)

# _ = model.get_latent_space()

# # This will print all the infor about noise, mll and etc
# model.print_stats()


################################# prediction on test set ########################################

