#!/usr/bin/env python
# coding: utf-8

# # LMGP for multifidelity regresssion demonstration


from pdb import Restart
from statistics import mode
import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from lmgp_pytorch.models import LMGP
from lmgp_pytorch.optim import fit_model_scipy,noise_tune
from lmgp_pytorch.utils.variables import NumericalVariable,CategoricalVariable
from lmgp_pytorch.utils.input_space import InputSpace

from typing import Dict

from lmgp_pytorch.visual import plot_latent


noise_flag = 0

# start timing
start_time = time.time()


#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi']=150
plt.rcParams['font.family']='serif'

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

# ## save .mat files
from scipy.io import loadmat
multifidelity_big_noise = loadmat('./multifidelity_no_noise.mat')

train_x, train_y, test_x, test_y = multifidelity_big_noise['x_train_all'], multifidelity_big_noise['y_train_all'], multifidelity_big_noise['x_val'], multifidelity_big_noise['y_val']


train_x = torch.from_numpy(np.float64(train_x))
train_y = torch.from_numpy(np.float64(train_y))
test_x  = torch.from_numpy(np.float64(test_x))
test_y  = torch.from_numpy(np.float64(test_y))


meanx = train_x[:,:-1].mean(dim=-2, keepdim=True)
stdx = train_x[:,:-1].std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
train_x[:,:-1] = (train_x[:,:-1] - meanx) / stdx
test_x[:,:-1] = (test_x[:,:-1] - meanx) / stdx

train_y = train_y.reshape(-1,)


index = torch.where(test_x[:,-1] == 2)
test_x = test_x[index]
test_y = test_y[index]

set_seed(4)
model2 = LMGP(
    train_x=train_x,
    train_y=train_y,
    qual_index= [10],
    quant_index= list(range(10)),
    num_levels_per_var=[4],
    quant_correlation_class="RBFKernel",
    NN_layers= []
).double()


model2.reset_parameters

# optimize noise successively
nll_inc_tuned,opt_history = noise_tune(
    model2, 
    num_restarts = 1,
    add_prior = False # number of restarts in the initial iteration
)

# 
print('NLL obtained from noise tuning strategy.......: %6.2f'%nll_inc_tuned)

# prediction on test set
with torch.no_grad():
    # set return_std = False if standard deviation is not needed
    test_mean2 = model2.predict(test_x,return_std=False, include_noise = True)
    


print('######################################')
noise = model2.likelihood.noise_covar.noise.item() * train_y.std()**2
print(f'The estimated noise parameter is {noise}')

# print MSE
mse = ( (test_y.reshape(-1,)-test_mean2)**2).mean()
print('MSE : %5.3f'%mse.item())


# print RRMSE
rrmse = torch.sqrt(((test_y-test_mean2)**2).mean()/((test_y-test_y.mean())**2).mean())
print('Test RRMSE with noise-tuning strategy : %5.3f'%rrmse.item())




# ending timing
end_time = time.time()
print(f'The total time in second is {end_time - start_time}')


#--------
plt.plot(test_y, test_mean2, 'ro')
plt.plot(test_y, test_y, 'b')


# plot latent values
plot_latent.plot_ls(model2, constraints_flag= True)

plt.show()