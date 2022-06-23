#!/usr/bin/env python
# coding: utf-8

# # LMGP regresssion demonstration
# 


from tenacity import before_sleep_nothing
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

from lmgp_pytorch.models import LVGPR, LMGP
from lmgp_pytorch.optim import fit_model_scipy,noise_tune
from lmgp_pytorch.utils.variables import NumericalVariable,CategoricalVariable
from lmgp_pytorch.utils.input_space import InputSpace

from typing import Dict

from lmgp_pytorch.visual import plot_latenth

from lmgp_pytorch.optim import noise_tune

from lmgp_pytorch.test_functions.multi_fidelity import  multi_fidelity_wing
###############Parameters########################
noise_flag = 1
noise_std = 3.0
add_prior_flag = True
num_minimize_init = 12
qual_index = [10]
quant_index= list(range(10))
level_sets = [4]
predict_fidelity = 1
save_mat_flag = False

quant_kernel = 'Rough_RBF' #'RBFKernel' #'Rough_RBF'

plt.rcParams['figure.dpi']=150
plt.rcParams['font.family']='serif'
#################################################
# start timing
start_time = time.time()



def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


#----------------------------- Read Data-----------------------
# ## save .mat files

train_x, train_y = multi_fidelity_wing({'h': 50,'l1': 100,'l2': 100,'l3': 100 })
test_x, test_y = multi_fidelity_wing({'h': 10000,'l1': 10000,'l2': 10000,'l3': 10000 })


train_x = torch.from_numpy(np.float64(train_x))
train_y = torch.from_numpy(np.float64(train_y))
test_x  = torch.from_numpy(np.float64(test_x))
test_y  = torch.from_numpy(np.float64(test_y))

meanx = train_x[:,:-1].mean(dim=-2, keepdim=True)
stdx = train_x[:,:-1].std(dim=-2, keepdim=True) + 1e-6 # prevent dividing by 0
train_x[:,:-1] = (train_x[:,:-1] - meanx) / stdx
test_x[:,:-1] = (test_x[:,:-1] - meanx) / stdx

train_y = train_y.reshape(-1,)
test_y = test_y.reshape(-1,)
#################### What fidelity to use for prediction ##################
index = torch.where(test_x[:,-1] == predict_fidelity)
test_x = test_x[index]
test_y = test_y[index]


set_seed(4)
model2 = LMGP(
    train_x=train_x,
    train_y=train_y,
    qual_index= qual_index,
    quant_index= quant_index,
    num_levels_per_var= level_sets,
    quant_correlation_class= quant_kernel,
    NN_layers= [],
    fix_noise= False
).double()

LMGP.reset_parameters

# optimize noise successively
reslist,opt_history = fit_model_scipy(
    model2, 
    num_restarts = num_minimize_init,
    add_prior=add_prior_flag # number of restarts in the initial iteration
)


################################# prediction on test set ########################################
with torch.no_grad():
    # set return_std = False if standard deviation is not needed
    test_mean2 = model2.predict(test_x,return_std=False, include_noise = True)
    


print('######################################')
noise = model2.likelihood.noise_covar.noise.item() * model2.y_std**2
print(f'The estimated noise parameter is {noise}')

# print MSE
mse = ( (test_y-test_mean2)**2).mean()
print('MSE : %5.3f'%mse.item())


# print RRMSE
rrmse = torch.sqrt(((test_y-test_mean2)**2).mean()/((test_y-test_y.mean())**2).mean())
print('Test RRMSE with noise-tuning strategy : %5.3f'%rrmse.item())




# ending timing
end_time = time.time()
print(f'The total time in second is {end_time - start_time}')
#######################################################################

zeta = torch.tensor(model2.zeta, dtype = torch.float64)
positions = model2.nn_model(zeta)

print('####################################################################')

print(f'The positions from nn_model are \n {positions}')

# #################-----------plot latent values------------####################
plot_latenth.plot_ls(model2, constraints_flag= True)

plt.figure(figsize=(8, 6))
plt.plot(test_y, test_mean2, 'ro')
plt.plot(test_y, test_y, 'b')
plt.xlabel(r'Y_True')
plt.ylabel(r'Y_predict')
plt.show()


