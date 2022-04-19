#!/usr/bin/env python
# coding: utf-8

# # LVGP regresssion demonstration
# 
# In this example, we will demonstrate training and analyzing standard LVGP models on the borehole dataset. The metric used to assess the test performance is the relative root mean squared error (RRMSE), which is given by
# 
# $$
# \mathrm{RRMSE} = \sqrt{
#    \frac{\sum_{i=1}^{N}\left(y_i-\widehat{y}_i\right)^2}{\sum_{i=1}^{N} \left(y_i-\overline{y}\right)^2}
# },   
# $$
# 
# where $y_i$ and $\widehat{y}_i$ are respectively the true and predicted response for the $i^\mathrm{th}$ test sample, and $\overline{y}$ is the mean of the true test responses.

# In[1]:


from pdb import Restart
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
from lmgp_pytorch.optim import fit_model_torch
from lmgp_pytorch.utils.variables import NumericalVariable,CategoricalVariable
from lmgp_pytorch.utils.input_space import InputSpace

from typing import Dict

from lmgp_pytorch.visual import plot_latent

noise_flag = 1

# start timing
start_time = time.time()


#get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.dpi']=150
plt.rcParams['font.family']='serif'

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


# ## Generating the training and test sets
# 
# The borehole function is given by
# 
# $$
# 2\pi T_u\left(H_u-H_l\right)\left(
#     \log\left(\frac{r}{r_w}\right)\left(
#         1+ 2\frac{LT_u}{\log\left(r/r_w\right)r_w^2K_w} + \frac{T_u}{T_l}
#     \right)
# \right)^{-1}.
# $$

# In[2]:


def borehole(params:Dict)->float:
    numerator = 2*math.pi*params['T_u']*(params['H_u']-params['H_l'])
    den_term1 = math.log(params['r']/params['r_w'])
    den_term2 = 1+ 2*params['L']*params['T_u']/(den_term1*params['r_w']**2*params['K_w']) +         params['T_u']/params['T_l']
    
    return numerator/den_term1/den_term2


# All 8 inputs are numerical. Similar to [Zhang et al. (2020)](https://doi.org/10.1080/00401706.2019.1638834), we discretize $r_w$ and $H_l$ over their domains to have 5 levels each.
# 
# We will be using the `lvgp_pytorch.utils.InputSpace` utility class to 
# 
# 1. generating training and test data
# 2. transforming the inputs into the format required by `LVGPR`:
#     - numerical/integer inputs are scaled to [0,1] (with/without log transform). 
#     - categorical inputs are encoded as integers 0,...,L-1
# 3. transforming the array format back to the original scale for obtaining the response
# 
# 
# We will now create the `InputSpace` object and add the associated variables.

# In[3]:


# configuration space
config = InputSpace()
r = NumericalVariable(name='r',lower=100,upper=50000)
Tu = NumericalVariable(name='T_u',lower=63070,upper=115600)
Hu = NumericalVariable(name='H_u',lower=990,upper=1110)
Tl = NumericalVariable(name='T_l',lower=63.1,upper=116)
L = NumericalVariable(name='L',lower=1120,upper=1680)
K_w = NumericalVariable(name='K_w',lower=9855,upper=12045)

r_w = CategoricalVariable(name='r_w',levels=np.linspace(0.05,0.15,5))
H_l = CategoricalVariable(name='H_l',levels=np.linspace(700,820,5))
config.add_inputs([r,Tu,Hu,Tl,L,K_w,r_w,H_l])

config


# We will generate 100 random samples to be used as training data and separately 1000 random samples to be used as test data.

# In[4]:


# generate 100 samples
set_seed(1)
num_samples = 100
train_x = torch.from_numpy(
    config.random_sample(np.random,num_samples)
)
train_y = [None]*num_samples

for i,x in enumerate(train_x):
    train_y[i] = borehole(config.get_dict_from_array(x.numpy()))

train_y = torch.tensor(train_y).double()

if noise_flag == 1:
    train_y += torch.randn(train_y.size()) * 3.0**2

# generate 1000 test samples
num_samples = 10000
test_x = torch.from_numpy(config.random_sample(np.random,num_samples))
test_y = [None]*num_samples

for i,x in enumerate(test_x):
    test_y[i] = borehole(config.get_dict_from_array(x.numpy()))
    
# create tensor objects
test_y = torch.tensor(test_y).to(train_y)

if noise_flag == 1:
    test_y += torch.randn(test_y.size()) * 3.0**2

# ## save .mat files
from scipy.io import savemat
savemat('borehole_100_noise.mat',{'Xtrain':train_x.numpy(), 'Xtest':test_x.numpy(), 'ytrain':train_y.numpy(), 'ytest':test_y.numpy()})


set_seed(4)
model3 = LMGP(
    train_x=train_x,
    train_y=train_y,
    qual_index=config.qual_index,
    quant_index=config.quant_index,
    num_levels_per_var=list(config.num_levels.values()),
    quant_correlation_class="RBFKernel",
).double()


# optimize noise successively
loss_f, loss_hist = fit_model_torch(
    model3, 
    num_iter= 10000,
    num_restarts= 0,
    lr_default=0.005,
    break_steps=2000
)

# 

print('loss is.......: %6.2f'%loss_f)


# prediction on test set
with torch.no_grad():
    # set return_std = False if standard deviation is not needed
    test_mean3 = model3.predict(test_x,return_std=False)
    


print('######################################')
noise = model3.likelihood.noise_covar.noise.item() * train_y.std()**2
print(f'The estimated noise parameter is {noise}')

# print MSE
mse = ( (test_y-test_mean3)**2).mean()
print('MSE : %5.3f'%mse.item())


# print RRMSE
rrmse = torch.sqrt(((test_y-test_mean3)**2).mean()/((test_y-test_y.mean())**2).mean())
print('Test RRMSE with torch adam optimizer is: %5.3f'%rrmse.item())




# ending timing
end_time = time.time()
print(f'The total time in second is {end_time - start_time}')

#
for loss in loss_hist:
    plt.plot(loss[500:])
plt.show()

# plot latent values
plot_latent.plot_ls(model3, constraints_flag= True)

plt.plot(test_y, test_mean3, 'ro')
plt.plot(test_y, test_y, 'b')
plt.show()

