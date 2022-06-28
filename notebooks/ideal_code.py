#!/usr/bin/env python
# coding: utf-8

# # LMGP regresssion demonstration
# 


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

from lmgp_pytorch.visual import plot_latent

from lmgp_pytorch.preprocessing import standard
from lmgp_pytorch.preprocessing import setlevels


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}

from sklearn.model_selection import train_test_split
#####################################################
from lmgp_pytorch.test_functions.physical import Borehole
from lmgp_pytorch.preprocessing import train_test_split_normalizeX
#####################################################
# start timing
start_time = time.time()

X, y = Borehole(n = 10000, random_state= 12345)
Xtrain, Xtest, ytrain, ytest = train_test_split_normalizeX(X, y, test_size = 0.99)

model = LMGP(quant =  'Rough_RBF', random_state = 0)

model.fit()

model.score(Xtest, ytest)

model.predict(Xtest, y_std = True)

# This will print all the infor about noise, mll and etc
model.print_stats()

# plot both visualizations for latent map and mse
model.visualize()

model.get_paramters()

#################################################
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi']=150
plt.rcParams['font.family']='serif'

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def borehole(params:Dict)->float:
    numerator = 2*math.pi*params['T_u']*(params['H_u']-params['H_l'])
    den_term1 = math.log(params['r']/params['r_w'])
    den_term2 = 1+ 2*params['L']*params['T_u']/(den_term1*params['r_w']**2*params['K_w']) +         params['T_u']/params['T_l']
    
    return numerator/den_term1/den_term2


# configuration space
config = InputSpace()
r = NumericalVariable(name='r',lower=100,upper=50000)
Tu = NumericalVariable(name='T_u',lower=63070,upper=115600)
Hu = NumericalVariable(name='H_u',lower=990,upper=1110)
Tl = NumericalVariable(name='T_l',lower=63.1,upper=116)
L = NumericalVariable(name='L',lower=1120,upper=1680)
K_w = NumericalVariable(name='K_w',lower=9855,upper=12045)

#L = CategoricalVariable(name='L',levels=np.linspace(1120,1680,5))
#K_w = CategoricalVariable(name='K_w', levels=np.linspace(9855,12045,5))

r_w = CategoricalVariable(name='r_w',levels=np.linspace(0.05,0.15,5))
H_l = CategoricalVariable(name='H_l',levels=np.linspace(700,820,5))
config.add_inputs([r,Tu,Hu,Tl,L,K_w,r_w,H_l])

config



#######################################################################
# generate 100 samples
set_seed(1)
train_x = torch.from_numpy(
    config.random_sample(np.random,num_samples_train)
)
train_y = [None]*num_samples_train

for i,x in enumerate(train_x):
    train_y[i] = borehole(config.get_dict_from_array(x.numpy()))

train_y = torch.tensor(train_y).double()

if noise_flag == 1:
    train_y += torch.randn(train_y.size()) * noise_std

# generate 1000 test samples
test_x = torch.from_numpy(config.random_sample(np.random,num_samples_test))
test_y = [None]*num_samples_test

for i,x in enumerate(test_x):
    test_y[i] = borehole(config.get_dict_from_array(x.numpy()))
    
# create tensor objects
test_y = torch.tensor(test_y).to(train_y)

if noise_flag == 1:
    test_y += torch.randn(test_y.size()) * noise_std

# ## save .mat files
if save_mat_flag:
    from scipy.io import savemat
    savemat('borehole_100.mat',{'Xtrain':train_x.numpy(), 'Xtest':test_x.numpy(), 'ytrain':train_y.numpy(), 'ytest':test_y.numpy()})


train_x = setlevels(train_x, config.qual_index)
test_x = setlevels(test_x, config.qual_index)
train_x, test_x = standard(train_x, config.quant_index, test_x)

train_x = train_x.to(**tkwargs)
train_y = train_y.to(**tkwargs)
test_x = test_x.to(**tkwargs)
test_y = test_y.to(**tkwargs)

model2 = LMGP(
    train_x=train_x.to(**tkwargs),
    train_y=train_y.to(**tkwargs),
    qual_index=config.qual_index,
    quant_index=config.quant_index,
    num_levels_per_var=list(config.num_levels.values()),
    quant_correlation_class= quant_kernel,
    NN_layers= [],
    fix_noise= False
).to(**tkwargs)

model2.reset_parameters

# optimize noise successively
reslist,opt_history = fit_model_scipy(
    model2, 
    num_restarts = num_minimize_init,
    add_prior=add_prior_flag # number of restarts in the initial iteration
    , n_jobs = 8
)

# 
#print(reslist)
#print(opt_history[-1])
################################# prediction on test set ########################################
with torch.no_grad():
    # set return_std = False if standard deviation is not needed
    test_mean2, std = model2.predict(test_x,return_std=True, include_noise = True)
    


# print('######################################')
# print(f'The value of the interval score is {model2.interval_alpha}')

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

# plot latent values
plot_latent.plot_ls(model2, constraints_flag= True)

plt.figure(figsize=(8,6))
plt.plot(test_y.cpu().numpy(), test_mean2.cpu().numpy(), 'ro')
plt.plot(test_y.cpu().numpy(), test_y.cpu().numpy(), 'b')
plt.xlabel(r'Y_True')
plt.ylabel(r'Y_predict')
plt.show()


